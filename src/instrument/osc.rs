//! OSC sender — packages live EEG signals through `NeuralPatch` mappings
//! and fires them at SuperCollider over UDP every ~50 ms.
//!
//! Continuous parameters → `/instrument/<module>/<param>  f  <value>`
//! Event triggers        → `/instrument/<module>/<param>  i  1`

use std::net::UdpSocket;
use std::time::{Duration, Instant};

use rosc::{encoder, OscMessage, OscPacket, OscType};

use super::mapping::NeuralPatch;
use super::signals::SharedSignals;

// ── OscSender ────────────────────────────────────────────────────────────────

/// Wraps a UDP socket and sends OSC packets to a SuperCollider instance.
pub struct OscSender {
    pub socket: UdpSocket,
    pub target: String,
}

impl OscSender {
    pub fn new(host: &str, port: u16) -> anyhow::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        Ok(Self {
            socket,
            target: format!("{}:{}", host, port),
        })
    }

    /// Send a single float value to an OSC address.
    pub fn send_f32(&self, address: &str, value: f32) -> anyhow::Result<()> {
        let msg = OscPacket::Message(OscMessage {
            addr: address.to_string(),
            args: vec![OscType::Float(value)],
        });
        let buf = encoder::encode(&msg)?;
        self.socket.send_to(&buf, &self.target)?;
        Ok(())
    }

    /// Send a trigger (integer 1) to an OSC address.
    pub fn send_trigger(&self, address: &str) -> anyhow::Result<()> {
        let msg = OscPacket::Message(OscMessage {
            addr: address.to_string(),
            args: vec![OscType::Int(1)],
        });
        let buf = encoder::encode(&msg)?;
        self.socket.send_to(&buf, &self.target)?;
        Ok(())
    }
}

// ── OscLoop ──────────────────────────────────────────────────────────────────

/// Drives the OSC output loop.
/// Call `tick()` from the GPUI update loop or a background timer.
pub struct OscLoop {
    pub sender: Option<OscSender>,
    last_tick: Instant,
    /// How often to send continuous parameter updates.
    interval: Duration,
}

impl OscLoop {
    pub fn new() -> Self {
        Self {
            sender: None,
            last_tick: Instant::now(),
            interval: Duration::from_millis(50), // 20 Hz
        }
    }

    /// (Re)connect to a SuperCollider instance.
    pub fn connect(&mut self, host: &str, port: u16) {
        match OscSender::new(host, port) {
            Ok(s) => self.sender = Some(s),
            Err(e) => eprintln!("[OSC] connect failed: {e}"),
        }
    }

    /// Returns true if enough time has elapsed and OSC is connected.
    pub fn ready(&self) -> bool {
        self.sender.is_some() && self.last_tick.elapsed() >= self.interval
    }

    /// Run one OSC tick: read live signals, apply all mappings, send OSC.
    ///
    /// `dt_ms` = elapsed ms since last call (for the slew limiter inside each mapping).
    pub fn tick(&mut self, patch: &mut NeuralPatch, signals: &SharedSignals, dt_ms: f32) {
        let Some(ref sender) = self.sender else { return };

        // Snapshot live signals (brief lock)
        let snap = match signals.lock() {
            Ok(s) => s.clone(),
            Err(_) => return,
        };

        // Clear one-shot event flags after reading
        if let Ok(mut s) = signals.lock() {
            s.clear_events();
        }

        for mapping in &mut patch.mappings {
            if !mapping.enabled {
                continue;
            }

            let raw = snap.get(&mapping.source);
            let out = mapping.process(raw, dt_ms);
            let addr = mapping.target.osc_address();

            let result = if mapping.target.is_continuous() {
                sender.send_f32(addr, out)
            } else {
                // Event: only send when the signal fires (raw > 0.5)
                if raw > 0.5 {
                    sender.send_trigger(addr)
                } else {
                    continue;
                }
            };

            if let Err(e) = result {
                eprintln!("[OSC] send error on {addr}: {e}");
            }
        }

        self.last_tick = Instant::now();
    }
}

impl Default for OscLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ── ScParams ──────────────────────────────────────────────────────────────────

/// Non-EEG synthesis parameters manually controlled by the performer.
/// Each change immediately sends an OSC message to SuperCollider.
#[derive(Debug, Clone)]
pub struct ScParams {
    /// Master output volume (0.0–1.0).
    pub master_volume: f32,
    /// Global reverb send (0.0–1.0).
    pub reverb: f32,
    /// Rhythmic tempo in BPM (60–200).
    pub bpm: f32,
    /// A4 tuning frequency in Hz (430–460).
    pub tuning: f32,
}

impl Default for ScParams {
    fn default() -> Self {
        Self {
            master_volume: 0.7,
            reverb: 0.2,
            bpm: 120.0,
            tuning: 440.0,
        }
    }
}

impl ScParams {
    /// Send all parameters to SuperCollider at once (e.g. after reconnect).
    pub fn send_all(&self, sender: &OscSender) {
        let _ = sender.send_f32("/sc/master/volume", self.master_volume);
        let _ = sender.send_f32("/sc/master/reverb", self.reverb);
        let _ = sender.send_f32("/sc/rhythm/bpm", self.bpm);
        let _ = sender.send_f32("/sc/master/tuning", self.tuning);
    }

    /// Read a parameter by its short ID string ("vol", "rev", "bpm", "tune").
    pub fn by_id(&self, id: &str) -> f32 {
        match id {
            "vol"  => self.master_volume,
            "rev"  => self.reverb,
            "bpm"  => self.bpm,
            "tune" => self.tuning,
            _      => 0.0,
        }
    }

    /// Write a parameter by its short ID string ("vol", "rev", "bpm", "tune").
    pub fn set_by_id(&mut self, id: &str, value: f32) {
        match id {
            "vol"  => self.master_volume = value,
            "rev"  => self.reverb = value,
            "bpm"  => self.bpm = value,
            "tune" => self.tuning = value,
            _      => {}
        }
    }
}
