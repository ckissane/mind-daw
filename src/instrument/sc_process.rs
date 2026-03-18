//! SuperCollider subprocess management and OSC receiver.
//!
//! `ScProcess` owns the sclang child process — spawning, polling, and killing it.
//! Stdout and stderr are captured in background threads and pushed into `SharedScStatus`.
//!
//! `OscReceiver` binds a UDP socket on :57111 and listens for messages SC sends back:
//!   /sc/meter/left   f  0.0–1.0   (audio level, linear)
//!   /sc/meter/right  f  0.0–1.0
//!   /sc/status       s  "booted" | "error: ..."

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::net::UdpSocket;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use rosc::decoder;

// ── ScState ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ScState {
    Stopped,
    Running,
    Error(String),
}

impl Default for ScState {
    fn default() -> Self {
        Self::Stopped
    }
}

impl ScState {
    pub fn label(&self) -> &str {
        match self {
            Self::Stopped  => "stopped",
            Self::Running  => "running",
            Self::Error(_) => "error",
        }
    }

    pub fn is_running(&self) -> bool {
        *self == Self::Running
    }
}

// ── ScStatus ──────────────────────────────────────────────────────────────────

/// Shared state written by subprocess reader threads and OSC receiver thread,
/// read by the GPUI render function.
#[derive(Debug)]
pub struct ScStatus {
    pub state: ScState,
    /// Most recent sclang stdout / stderr lines (capped at 20).
    pub log_lines: VecDeque<String>,
    /// Audio meter — left channel (linear 0–1, from SC metering OSC).
    pub meter_left: f32,
    /// Audio meter — right channel (linear 0–1).
    pub meter_right: f32,
}

impl Default for ScStatus {
    fn default() -> Self {
        Self {
            state: ScState::Stopped,
            log_lines: VecDeque::with_capacity(20),
            meter_left: 0.0,
            meter_right: 0.0,
        }
    }
}

impl ScStatus {
    pub fn push_log(&mut self, line: String) {
        if self.log_lines.len() >= 20 {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }
}

pub type SharedScStatus = Arc<Mutex<ScStatus>>;

// ── ScProcess ─────────────────────────────────────────────────────────────────

/// Manages the sclang subprocess lifetime.
pub struct ScProcess {
    child: Option<Child>,
    pub status: SharedScStatus,
}

impl ScProcess {
    pub fn new() -> (Self, SharedScStatus) {
        let status: SharedScStatus = Arc::new(Mutex::new(ScStatus::default()));
        let proc = Self {
            child: None,
            status: Arc::clone(&status),
        };
        (proc, status)
    }

    /// Start sclang with the given .scd file.
    /// Tries `sclang` on PATH first, then the standard macOS app bundle path.
    pub fn spawn(&mut self, scd_path: &str) {
        self.kill(); // stop any already-running instance

        let sclang = Self::find_sclang();
        let args = [scd_path];

        match Command::new(&sclang)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                // ── stdout reader thread ──────────────────────────────────
                if let Some(stdout) = child.stdout.take() {
                    let s = Arc::clone(&self.status);
                    std::thread::Builder::new()
                        .name("sclang-stdout".into())
                        .spawn(move || {
                            for line in BufReader::new(stdout).lines().flatten() {
                                if let Ok(mut st) = s.lock() {
                                    st.push_log(line);
                                }
                            }
                        })
                        .ok();
                }

                // ── stderr reader thread ──────────────────────────────────
                if let Some(stderr) = child.stderr.take() {
                    let s = Arc::clone(&self.status);
                    std::thread::Builder::new()
                        .name("sclang-stderr".into())
                        .spawn(move || {
                            for line in BufReader::new(stderr).lines().flatten() {
                                if let Ok(mut st) = s.lock() {
                                    st.push_log(format!("[err] {line}"));
                                }
                            }
                        })
                        .ok();
                }

                if let Ok(mut st) = self.status.lock() {
                    st.state = ScState::Running;
                    st.push_log(format!("▶  sclang {scd_path}"));
                }
                self.child = Some(child);
            }
            Err(e) => {
                if let Ok(mut st) = self.status.lock() {
                    st.state = ScState::Error(e.to_string());
                    st.push_log(format!("✗  could not start '{sclang}': {e}"));
                    st.push_log("   Is SuperCollider installed?".into());
                }
            }
        }
    }

    /// Kill the running sclang process AND the scsynth audio server it spawned.
    ///
    /// Killing only sclang leaves scsynth running in the background, causing
    /// audio to continue after "stop" is pressed. We pkill scsynth explicitly.
    pub fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();

            // scsynth is a separate child process spawned by sclang.
            // Kill it so audio stops immediately (macOS/Linux).
            let _ = std::process::Command::new("pkill")
                .args(["-x", "scsynth"])
                .status();

            if let Ok(mut st) = self.status.lock() {
                st.state = ScState::Stopped;
                st.push_log("■  sclang + scsynth stopped".into());
                st.meter_left  = 0.0;
                st.meter_right = 0.0;
            }
        }
    }

    /// Poll the child process status — call from the UI update loop.
    /// Detects unexpected exits and updates state.
    pub fn poll(&mut self) {
        if let Some(ref mut child) = self.child {
            match child.try_wait() {
                Ok(Some(code)) => {
                    self.child = None;
                    if let Ok(mut st) = self.status.lock() {
                        if code.success() {
                            st.state = ScState::Stopped;
                            st.push_log("■  sclang exited normally".into());
                        } else {
                            let msg = format!("sclang crashed ({})", code);
                            st.state = ScState::Error(msg.clone());
                            st.push_log(format!("✗  {msg}"));
                        }
                    }
                }
                Ok(None) => {} // still running
                Err(e) => {
                    if let Ok(mut st) = self.status.lock() {
                        st.state = ScState::Error(e.to_string());
                    }
                }
            }
        }
    }

    /// True if sclang process is alive.
    pub fn is_running(&self) -> bool {
        self.status
            .lock()
            .map(|s| s.state.is_running())
            .unwrap_or(false)
    }

    /// Find the sclang executable: PATH first, then macOS app bundle.
    fn find_sclang() -> String {
        // Check PATH
        if std::process::Command::new("which")
            .arg("sclang")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return "sclang".into();
        }
        // macOS app bundle
        let mac = "/Applications/SuperCollider.app/Contents/MacOS/sclang";
        if std::path::Path::new(mac).exists() {
            return mac.into();
        }
        "sclang".into() // let the error surface naturally
    }
}

impl Default for ScProcess {
    fn default() -> Self {
        Self::new().0
    }
}

// ── OscReceiver ───────────────────────────────────────────────────────────────

/// Listens on a UDP port for OSC messages from SuperCollider.
/// Runs entirely in a background thread.
pub struct OscReceiver;

impl OscReceiver {
    /// Bind to `port` (default: 57111) and start listening.
    /// Updates `status` with meter and status messages from SC.
    pub fn spawn(port: u16, status: SharedScStatus) {
        std::thread::Builder::new()
            .name("osc-receiver".into())
            .spawn(move || {
                let bind = format!("0.0.0.0:{port}");
                let socket = match UdpSocket::bind(&bind) {
                    Ok(s) => s,
                    Err(e) => {
                        if let Ok(mut st) = status.lock() {
                            st.push_log(format!("[OSC recv] bind :{port} failed: {e}"));
                        }
                        return;
                    }
                };

                if let Ok(mut st) = status.lock() {
                    st.push_log(format!("[OSC recv] listening on :{port}"));
                }

                let mut buf = [0u8; 4096];
                loop {
                    match socket.recv_from(&mut buf) {
                        Ok((n, _)) => {
                            if let Ok((_, packet)) = decoder::decode_udp(&buf[..n]) {
                                Self::handle(&packet, &status);
                            }
                        }
                        Err(e) => {
                            eprintln!("[OscReceiver] recv error: {e}");
                            break;
                        }
                    }
                }
            })
            .ok();
    }

    fn handle(packet: &rosc::OscPacket, status: &SharedScStatus) {
        use rosc::{OscPacket, OscType};
        match packet {
            OscPacket::Message(msg) => {
                let Ok(mut st) = status.lock() else { return };
                match msg.addr.as_str() {
                    "/sc/meter/left" => {
                        if let Some(OscType::Float(v)) = msg.args.first() {
                            st.meter_left = v.clamp(0.0, 1.0);
                        }
                    }
                    "/sc/meter/right" => {
                        if let Some(OscType::Float(v)) = msg.args.first() {
                            st.meter_right = v.clamp(0.0, 1.0);
                        }
                    }
                    "/sc/status" => {
                        if let Some(OscType::String(s)) = msg.args.first() {
                            st.push_log(format!("[SC] {s}"));
                            if s.contains("booted") || s.contains("ready") {
                                st.state = ScState::Running;
                            }
                        }
                    }
                    other => {
                        st.push_log(format!("[SC←] {other}"));
                    }
                }
            }
            OscPacket::Bundle(b) => {
                for p in &b.content {
                    Self::handle(p, status);
                }
            }
        }
    }
}
