//! SuperCollider (scsynth) integration via sorceress + rosc.
//!
//! Uses `sorceress` for typed SynthDef building and server connection,
//! with `rosc` as fallback for commands sorceress doesn't support (e.g. GroupNew).
//!
//! ## How to run SuperCollider
//!
//! 1. Install SuperCollider: `sudo apt install supercollider` (Linux)
//!    or download from <https://supercollider.github.io>
//! 2. Open `synthdefs.scd` (in the project root) in the SuperCollider IDE
//! 3. Select all (Ctrl+A) and evaluate (Ctrl+Enter)
//!    — this boots the server and loads all SynthDefs automatically
//! 4. In mind-daw, click "Connect SC" in the Tonnetz tab

use rosc::{OscMessage, OscPacket, OscType};
use sorceress::server::{AddAction, Control, Server, SynthNew, NodeSet, NodeFree};
use std::net::UdpSocket;
use std::sync::mpsc;

// ── Constants ────────────────────────────────────────────────────────────────

const SCSYNTH_ADDR: &str = "127.0.0.1:57110";
const NODE_ID_BASE: i32 = 1000;
const GROUP_SOURCES: i32 = 100;
const GROUP_FX: i32 = 200;

// ── Raw OSC for commands sorceress doesn't support ──────────────────────────

fn raw_osc_send(addr: &str, port: u16, msg_addr: &str, args: Vec<OscType>) {
    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
        let _ = sock.connect(format!("{addr}:{port}"));
        let msg = OscPacket::Message(OscMessage {
            addr: msg_addr.to_string(),
            args,
        });
        if let Ok(bytes) = rosc::encoder::encode(&msg) {
            let _ = sock.send(&bytes);
        }
    }
}

fn create_groups() {
    // sorceress doesn't implement GroupNew, so we use raw OSC
    raw_osc_send("127.0.0.1", 57110, "/g_new",
        vec![OscType::Int(GROUP_SOURCES), OscType::Int(0), OscType::Int(0)]);
    raw_osc_send("127.0.0.1", 57110, "/g_new",
        vec![OscType::Int(GROUP_FX), OscType::Int(3), OscType::Int(GROUP_SOURCES)]);
}

// ── SynthDef source (evaluate in SuperCollider IDE) ─────────────────────────

/// Evaluate this string in the SuperCollider IDE after `s.boot;`.
pub const SYNTHDEFS_SCLANG: &str = r#"
// ── mind-daw SynthDefs ──────────────────────────────────────────────────

// Warm ambient pad: detuned sines + filter
SynthDef(\mpad, { |out=0, freq=220, amp=0.15, gate=1,
                   ffreq=1200, rq=0.6, detune=0.008,
                   atk=0.8, rel=2.0, pan=0|
    var sig, env, n=7;
    env = EnvGen.kr(Env.asr(atk, 1, rel), gate, doneAction: 2);
    sig = Mix.fill(n, { |i|
        var dt = detune * (i - (n-1)/2) / (n-1);
        SinOsc.ar(freq * (1 + dt), {2pi.rand}!2)
    }) / n;
    sig = BLowPass4.ar(sig, ffreq.clip(80, 18000), rq.clip(0.1, 2));
    sig = sig * env * amp;
    sig = Balance2.ar(sig[0], sig[1], pan);
    Out.ar(out, sig);
}).add;

// Rival-Consoles-inspired pluck: saw + vibrato + 12dB filter envelope + saturation
SynthDef(\mrcpluck, { |out=0, freq=440, amp=0.25, vel=0.8,
                       fbase=112, fdecay=0.3, vdecay=0.85,
                       vibrate=3.1, vibdepth=0.0081, pan=0|
    var sig, fenv, aenv, vib;
    vib = SinOsc.kr(vibrate, {2pi.rand}) * vibdepth * freq;
    sig = LFSaw.ar(freq + vib, 0, 0.7) + WhiteNoise.ar(0.03);
    fenv = EnvGen.kr(Env.perc(0, fdecay * (0.5 + vel)), doneAction: 0);
    sig = BLowPass.ar(sig, (fbase + (4000 * fenv * vel)).clip(20, 18000), 0.7);
    aenv = EnvGen.kr(Env.perc(0, vdecay * (0.5 + vel * 0.5)), doneAction: 2);
    sig = Pan2.ar(sig * aenv * amp * vel, pan);
    sig = (sig * 1.5).tanh * 0.7;
    Out.ar(out, sig);
}).add;

// Plucked string (Karplus-Strong)
SynthDef(\mpluck, { |out=0, freq=440, amp=0.3, dur=2, coef=0.1, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.linen(0.001, dur, 0.1), doneAction: 2);
    sig = Pluck.ar(PinkNoise.ar(amp), Impulse.kr(0), 0.2,
                   freq.reciprocal, dur, coef);
    sig = Pan2.ar(sig * env, pan);
    Out.ar(out, sig);
}).add;

// Bell/chime
SynthDef(\mbell, { |out=0, freq=880, amp=0.2, dur=3, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.perc(0.005, dur), doneAction: 2);
    sig = SinOsc.ar(freq) * env * amp;
    sig = sig + (SinOsc.ar(freq * 2.01, 0, amp * 0.3) * env);
    sig = sig + (SinOsc.ar(freq * 3.03, 0, amp * 0.1) * env);
    sig = Pan2.ar(sig, pan);
    Out.ar(out, sig);
}).add;

// Bass drone
SynthDef(\mdrone, { |out=0, freq=55, amp=0.12, gate=1, ffreq=400|
    var sig, env;
    env = EnvGen.kr(Env.asr(1.5, 1, 2), gate, doneAction: 2);
    sig = LFSaw.ar([freq, freq*1.002]) * 0.5;
    sig = sig + (SinOsc.ar(freq * 0.5) * 0.5);
    sig = BLowPass4.ar(sig, ffreq.clip(60, 5000), 0.5);
    sig = sig * env * amp;
    Out.ar(out, sig);
}).add;

// Percussive pulse
SynthDef(\mpulse, { |out=0, freq=440, amp=0.15, dur=0.1, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.perc(0.002, dur), doneAction: 2);
    sig = SinOsc.ar(freq) * env * amp;
    sig = sig + (WhiteNoise.ar(amp * 0.1) * EnvGen.kr(Env.perc(0.001, 0.02)));
    sig = Pan2.ar(sig, pan);
    Out.ar(out, sig);
}).add;

// Master reverb
SynthDef(\mreverb, { |out=0, in=0, mix=0.35, room=0.7, damp=0.5, amp=1|
    var dry, wet;
    dry = In.ar(in, 2);
    wet = FreeVerb2.ar(dry[0], dry[1], mix, room, damp);
    ReplaceOut.ar(out, wet * amp);
}).add;

"mind-daw SynthDefs loaded.".postln;
"#;

// ── Sound Profiles ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SoundProfile {
    pub name: &'static str,
    pub description: &'static str,
    pub voice: Voice,
    pub params: ChordParams,
    pub reverb_mix: f32,
    pub reverb_room: f32,
    pub reverb_damp: f32,
    pub bpm: f32,
    pub rhythm_pattern: Vec<f32>,
    pub velocity_pattern: Vec<f32>,
    pub allowed_chord_types: Vec<&'static str>,
    pub random_pan: bool,
}

impl Default for SoundProfile {
    fn default() -> Self {
        Self {
            name: "Default",
            description: "Basic pad, no rhythm",
            voice: Voice::Pad,
            params: ChordParams::default(),
            reverb_mix: 0.3,
            reverb_room: 0.7,
            reverb_damp: 0.5,
            bpm: 0.0,
            rhythm_pattern: Vec::new(),
            velocity_pattern: Vec::new(),
            allowed_chord_types: Vec::new(),
            random_pan: false,
        }
    }
}

pub fn builtin_profiles() -> Vec<SoundProfile> {
    vec![
        SoundProfile {
            name: "Soundprofile 1",
            description: "Dark ethereal, hexatonic PL cycle",
            voice: Voice::LoFi,
            params: ChordParams {
                amp: 0.18, duration: 0.6, filter_freq: 200.0,
                detune: 0.0, attack: 0.0, release: 0.6,
            },
            reverb_mix: 0.30, reverb_room: 0.85, reverb_damp: 0.35,
            bpm: 118.0,
            rhythm_pattern: vec![0.25, 0.25, 0.5, 0.25, 0.25, 0.75, 0.5, 0.25],
            velocity_pattern: vec![0.85, 0.5, 0.7, 0.6, 0.4, 0.8, 0.55, 0.65],
            // Hexatonic PL cycle Hex(0): C, Cm, Ab, Abm, E, Em
            // These 6 triads are connected by maximally smooth voice leadings
            // (every step moves exactly 1 voice by 1 semitone).
            // Includes both consonant stability (C, Em) and dark chromatic
            // mediants (Ab, Abm) for an uncanny, cinematic quality.
            allowed_chord_types: vec!["min", "Maj"],
            random_pan: true,
        },
        SoundProfile {
            name: "Soundprofile 2",
            description: "Evolving detuned pad, slow",
            voice: Voice::Pad,
            params: ChordParams {
                amp: 0.12, duration: 4.0, filter_freq: 800.0,
                detune: 0.012, attack: 1.5, release: 3.0,
            },
            reverb_mix: 0.4, reverb_room: 0.85, reverb_damp: 0.6,
            bpm: 60.0,
            rhythm_pattern: vec![4.0],
            velocity_pattern: vec![0.7],
            allowed_chord_types: vec!["Maj", "min", "Maj7", "sus2", "sus4"],
            random_pan: false,
        },
        SoundProfile {
            name: "Soundprofile 3",
            description: "Bell tones, arpeggiated",
            voice: Voice::Bell,
            params: ChordParams {
                amp: 0.18, duration: 2.5, filter_freq: 4000.0,
                detune: 0.0, attack: 0.0, release: 2.5,
            },
            reverb_mix: 0.35, reverb_room: 0.8, reverb_damp: 0.4,
            bpm: 90.0,
            rhythm_pattern: vec![0.333, 0.333, 0.334, 1.0],
            velocity_pattern: vec![0.8, 0.6, 0.5, 0.7],
            allowed_chord_types: Vec::new(),
            random_pan: true,
        },
        SoundProfile {
            name: "Soundprofile 4",
            description: "Bass drone, minimal",
            voice: Voice::Drone,
            params: ChordParams {
                amp: 0.1, duration: 8.0, filter_freq: 300.0,
                detune: 0.002, attack: 2.0, release: 4.0,
            },
            reverb_mix: 0.25, reverb_room: 0.9, reverb_damp: 0.7,
            bpm: 0.0,
            rhythm_pattern: Vec::new(),
            velocity_pattern: Vec::new(),
            allowed_chord_types: vec!["P5", "P4"],
            random_pan: false,
        },
    ]
}

// ── Types ────────────────────────────────────────────────────────────────────

pub enum ScCommand {
    PlayChord {
        midi_notes: Vec<u8>,
        voice: Voice,
        params: ChordParams,
        random_pan: bool,
    },
    /// Start the sequencer with a profile and initial chord.
    StartSequencer {
        profile: SoundProfile,
        midi_notes: Vec<u8>,
    },
    /// Update the sequencer's chord (from tonnetz navigation).
    UpdateChord {
        midi_notes: Vec<u8>,
    },
    /// Feed EEG state for real-time modulation of the sequencer.
    UpdateEeg {
        tension: f32,
        stability: f32,
        motion_x: f32,
        motion_y: f32,
        confidence: f32,
    },
    /// Stop the sequencer.
    StopSequencer,
    ReleaseAll,
    SetReverb { mix: f32, room: f32, damp: f32 },
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Voice {
    Pad,
    RcPluck,
    Pluck,
    Bell,
    Drone,
    Pulse,
    LoFi,
    Glass,
}

impl Voice {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pad => "Pad", Self::RcPluck => "RC Pluck", Self::Pluck => "Pluck",
            Self::Bell => "Bell", Self::Drone => "Drone", Self::Pulse => "Pulse",
            Self::LoFi => "Lo-Fi", Self::Glass => "Glass",
        }
    }
    pub fn sc_name(self) -> &'static str {
        match self {
            Self::Pad => "mpad", Self::RcPluck => "mrcpluck", Self::Pluck => "mpluck",
            Self::Bell => "mbell", Self::Drone => "mdrone", Self::Pulse => "mpulse",
            Self::LoFi => "mlofi", Self::Glass => "mglass",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChordParams {
    pub amp: f32,
    pub duration: f32,
    pub filter_freq: f32,
    pub detune: f32,
    pub attack: f32,
    pub release: f32,
}

impl Default for ChordParams {
    fn default() -> Self {
        Self { amp: 0.15, duration: 2.0, filter_freq: 1200.0,
               detune: 0.008, attack: 0.5, release: 1.5 }
    }
}

// ── Sequencer ───────────────────────────────────────────────────────────────

/// Internal state for the rhythm sequencer running in the SC worker thread.
/// Section variations for evolving arrangement.
const SECTION_DURATION_SECS: f32 = 30.0;
const NUM_SECTIONS: usize = 4;

struct SequencerState {
    profile: SoundProfile,
    midi_notes: Vec<u8>,
    step_idx: usize,
    note_idx: usize, // arpeggio index within the chord
    last_fire: std::time::Instant,
    started_at: std::time::Instant,
    /// Beat duration derived from BPM.
    beat_dur: std::time::Duration,
    /// Accumulated beats for the kick/hat pattern.
    beat_counter: f32,
    /// Current section (changes every SECTION_DURATION_SECS).
    section: usize,
    /// EEG modulation inputs (0.0-1.0).
    tension: f32,     // beta+gamma activation → energy, density, brightness
    stability: f32,   // alpha dominance → filter openness, consonance
    motion_x: f32,    // alpha-beta balance → tempo variation
    motion_y: f32,    // theta → depth, reverb, octave register
    confidence: f32,  // overall signal quality → how much EEG drives vs autopilot
    /// Node IDs for the sustained shimmer pad layer.
    pad_nodes: Vec<i32>,
    rng: u32,
}

impl SequencerState {
    fn new(profile: SoundProfile, midi_notes: Vec<u8>) -> Self {
        let bpm = profile.bpm.max(20.0);
        Self {
            profile,
            midi_notes,
            step_idx: 0,
            note_idx: 0,
            last_fire: std::time::Instant::now(),
            started_at: std::time::Instant::now(),
            beat_dur: std::time::Duration::from_secs_f64(60.0 / bpm as f64),
            beat_counter: 0.0,
            section: 0,
            tension: 0.3,
            stability: 0.5,
            motion_x: 0.0,
            motion_y: 0.0,
            confidence: 0.0,
            pad_nodes: Vec::new(),
            rng: 0xCAFE_BABE,
        }
    }

    /// Current bar number (4 beats per bar).
    fn bar(&self) -> u32 {
        (self.beat_counter / 4.0).floor() as u32
    }

    /// Position within the current 4-bar phrase (0..15 beats).
    fn phrase_beat(&self) -> f32 {
        self.beat_counter % 16.0
    }

    fn step_duration(&self) -> std::time::Duration {
        let pat = &self.profile.rhythm_pattern;
        let pat_len = pat.len().max(1);
        let frac = pat[self.step_idx % pat_len];

        // ── EEG: tension drives density (high tension = fast notes) ──
        // Range: 0.5x speed at tension=0 to 1.5x at tension=1
        let tension_scale = 1.3 - self.tension * 0.8;

        // ── EEG: motion_x drifts the tempo ±15% ──
        let tempo_drift = 1.0 + self.motion_x * 0.15;

        // Phrase fill on 4th bar
        let fill_factor = if self.phrase_beat() >= 12.0 { 0.6 } else { 1.0 };

        // Micro-swing
        let swing = if self.step_idx % 2 == 1 { 1.08 } else { 0.92 };

        // Blend EEG influence with autopilot based on confidence
        let eeg_scale = tension_scale * tempo_drift;
        let scale = 1.0 * (1.0 - self.confidence) + eeg_scale * self.confidence;

        self.beat_dur.mul_f32(frac * scale * fill_factor * swing)
    }

    fn current_velocity(&self) -> f32 {
        let pat = &self.profile.velocity_pattern;
        let pat_len = pat.len().max(1);
        let base = pat[self.step_idx % pat_len];

        // ── EEG: tension drives loudness ──
        let tension_vel = 0.5 + self.tension * 0.5; // 0.5 → 1.0

        // Phrase crescendo
        let phrase_curve = 0.7 + 0.3 * (self.phrase_beat() / 16.0);

        // Bar breathing
        let bar_breath = if self.bar() % 2 == 1 { 0.85 } else { 1.0 };

        // Humanization
        let human = 0.8 + self.rng_f32() * 0.4;

        // Blend
        let eeg_vel = base * tension_vel;
        let auto_vel = base * phrase_curve * bar_breath;
        let vel = auto_vel * (1.0 - self.confidence) + eeg_vel * self.confidence;

        (vel * human).clamp(0.1, 1.0)
    }

    fn next_note(&mut self) -> u8 {
        if self.midi_notes.is_empty() {
            return 60;
        }
        let n = self.midi_notes.len();
        let phrase_step = self.step_idx % 16;

        // ── Arpeggio pattern (structural safety net) ──
        let arp_idx = match phrase_step {
            0 => 0,
            1 => 1 % n,
            2 => 2 % n,
            3 => (n - 1) % n,
            4 => (n - 1) % n,
            5 => (n.saturating_sub(2)) % n,
            6 => 0,
            7 => 1 % n,
            8 => 0,
            9 => 0,
            10 => 2 % n,
            11 => 1 % n,
            12 => 0,
            13 => (n - 1) % n,
            14 => 1 % n,
            15 => (n - 1) % n,
            _ => self.note_idx % n,
        };

        let base_note = self.midi_notes[arp_idx];
        self.note_idx += 1;

        // ── EEG: motion_y drives register (theta = deeper → lower octave) ──
        let register_shift = if self.motion_y > 0.3 {
            -12 // high theta → octave down (deeper, more introspective)
        } else if self.motion_y < -0.3 {
            12 // low theta → octave up (alert, bright)
        } else {
            0
        };

        // ── EEG: high tension → more octave jumps (energetic) ──
        let tension_jump = if self.tension > 0.6 && self.rng_f32() > 0.5 {
            if self.rng_f32() > 0.5 { 12 } else { -12 }
        } else {
            0
        };

        // ── EEG: low stability → chance of rest (silence = space) ──
        if self.stability < 0.2 && self.rng_f32() > 0.6 {
            return 0; // MIDI 0 = rest (very low, effectively silent)
        }

        // Blend: confidence determines how much EEG drives register
        let eeg_shift = register_shift + tension_jump;
        let shift = if self.confidence > 0.3 { eeg_shift } else { 0 };

        // Fill octave drops
        let fill_shift = if phrase_step >= 12 && self.rng_f32() > 0.7 { -12 } else { 0 };

        (base_note as i32 + shift + fill_shift).clamp(36, 96) as u8
    }

    fn advance(&mut self) {
        self.step_idx += 1;
        self.last_fire = std::time::Instant::now();
    }

    fn rand_pan(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng as f32 / u32::MAX as f32) * 1.8 - 0.9
    }

    fn rng_f32(&self) -> f32 {
        let h = (self.step_idx as u32).wrapping_mul(2654435761);
        h as f32 / u32::MAX as f32
    }
}

// ── ScHandle ────────────────────────────────────────────────────────────────

pub struct ScHandle {
    pub cmd_tx: mpsc::Sender<ScCommand>,
    pub connected: bool,
}

impl ScHandle {
    pub fn play_chord(&self, midi_notes: Vec<u8>, voice: Voice, params: ChordParams, random_pan: bool) {
        let _ = self.cmd_tx.send(ScCommand::PlayChord { midi_notes, voice, params, random_pan });
    }
    pub fn release_all(&self) {
        let _ = self.cmd_tx.send(ScCommand::ReleaseAll);
    }
    pub fn set_reverb(&self, mix: f32, room: f32, damp: f32) {
        let _ = self.cmd_tx.send(ScCommand::SetReverb { mix, room, damp });
    }
    pub fn start_sequencer(&self, profile: SoundProfile, midi_notes: Vec<u8>) {
        let _ = self.cmd_tx.send(ScCommand::StartSequencer { profile, midi_notes });
    }
    pub fn update_chord(&self, midi_notes: Vec<u8>) {
        let _ = self.cmd_tx.send(ScCommand::UpdateChord { midi_notes });
    }
    pub fn update_eeg(&self, tension: f32, stability: f32, motion_x: f32, motion_y: f32, confidence: f32) {
        let _ = self.cmd_tx.send(ScCommand::UpdateEeg { tension, stability, motion_x, motion_y, confidence });
    }
    pub fn stop_sequencer(&self) {
        let _ = self.cmd_tx.send(ScCommand::StopSequencer);
    }
}

impl Drop for ScHandle {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(ScCommand::Shutdown);
    }
}

/// Spawn the SC worker. Connects to scsynth via sorceress::server::Server.
pub fn spawn_sc_worker() -> ScHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ScCommand>();

    let connected = match Server::connect(SCSYNTH_ADDR) {
        Ok(server) => {
            // Create groups via raw OSC (sorceress lacks GroupNew).
            // Small delay to let scsynth finish booting if it just started.
            std::thread::sleep(std::time::Duration::from_millis(500));
            create_groups();
            std::thread::sleep(std::time::Duration::from_millis(100));
            eprintln!("[sc] connected to scsynth at {SCSYNTH_ADDR} via sorceress");

            std::thread::Builder::new()
                .name("sc-worker".into())
                .spawn(move || sc_worker_loop(server, cmd_rx))
                .expect("failed to spawn SC worker");
            true
        }
        Err(e) => {
            eprintln!("[sc] failed to connect to scsynth: {e}");
            std::thread::Builder::new()
                .name("sc-worker-stub".into())
                .spawn(move || { while cmd_rx.recv().is_ok() {} })
                .expect("failed to spawn SC stub");
            false
        }
    };

    ScHandle { cmd_tx, connected }
}

// ── Worker loop using sorceress ─────────────────────────────────────────────

fn sc_worker_loop(server: Server, cmd_rx: mpsc::Receiver<ScCommand>) {
    let mut next_node_id = NODE_ID_BASE;
    let mut sustained_nodes: Vec<i32> = Vec::new();
    let mut rng_state: u32 = 0xDEAD_BEEF;
    let mut sequencer: Option<SequencerState> = None;

    // Start master reverb on FX group
    let _ = server.send(
        SynthNew::new("mreverb", GROUP_FX)
            .synth_id(99)
            .add_action(AddAction::TailOfGroup)
            .controls(vec![
                Control::new("mix", 0.3_f32),
                Control::new("room", 0.7_f32),
            ]),
    );

    loop {
        // Non-blocking recv so the sequencer can tick independently
        let timeout = std::time::Duration::from_millis(1);
        match cmd_rx.recv_timeout(timeout) {
            Ok(cmd) => {
                handle_command(
                    &server, cmd,
                    &mut next_node_id, &mut sustained_nodes, &mut rng_state,
                    &mut sequencer,
                );
                // Check for shutdown
                if sequencer.is_none() && sustained_nodes.is_empty()
                    && next_node_id == NODE_ID_BASE
                {
                    // Might have been a Shutdown — but we break inside handle_command
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // ── Sequencer tick ──────────────────────────────────────────────
        if let Some(seq) = &mut sequencer {
            // Section change every 30 seconds
            let secs_elapsed = seq.started_at.elapsed().as_secs_f32();
            let new_section = ((secs_elapsed / SECTION_DURATION_SECS) as usize) % NUM_SECTIONS;
            if new_section != seq.section {
                seq.section = new_section;
                eprintln!("[sc] section change → {new_section}");
                // Modulate reverb per section
                let (rmix, rroom, rdamp) = match new_section {
                    0 => (0.13, 0.75, 0.3),  // tight, bright
                    1 => (0.30, 0.85, 0.4),  // wider, more ethereal
                    2 => (0.15, 0.70, 0.2),  // punchy, dry-ish
                    _ => (0.40, 0.90, 0.5),  // washed out, dreamy
                };
                let _ = server.send(NodeSet::new(99, vec![
                    Control::new("mix", rmix),
                    Control::new("room", rroom),
                    Control::new("damp", rdamp),
                ]));
                // Modulate pad layer filter per section
                let pad_ffreq = match new_section {
                    0 => 700.0_f32,
                    1 => 1200.0,
                    2 => 500.0,
                    _ => 1500.0,
                };
                for &nid in &seq.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("ffreq", pad_ffreq),
                    ]));
                }
            }

            let elapsed = seq.last_fire.elapsed();
            let step_dur = seq.step_duration();

            if elapsed >= step_dur && !seq.midi_notes.is_empty() {
                // Track beat position for kick/hat pattern
                let step_beats = seq.profile.rhythm_pattern
                    .get(seq.step_idx % seq.profile.rhythm_pattern.len().max(1))
                    .copied()
                    .unwrap_or(1.0);
                let prev_beat = seq.beat_counter;
                seq.beat_counter += step_beats;

                // ── Kick pattern with groove ──
                // Beat positions within a 4-beat bar: kick on 1, 2.5 (syncopated), 3
                let prev_eighth = (prev_beat * 2.0).floor() as u32;
                let curr_eighth = (seq.beat_counter * 2.0).floor() as u32;
                let section = seq.section;
                let has_kick = section != 1;
                // Kick on 8th-note positions 0 (beat 1), 4 (beat 3), and 5 (the "and" of 3)
                // ── EEG: tension adds more kick hits ──
                let kick_positions: &[u32] = if seq.phrase_beat() >= 12.0 {
                    &[0, 2, 4, 5, 6] // fill: dense
                } else if seq.tension > 0.6 {
                    &[0, 3, 4, 5] // high tension: four-on-the-floor + syncopation
                } else if seq.tension > 0.3 {
                    &[0, 5] // medium: groove
                } else {
                    &[0] // low tension: sparse, just the downbeat
                };
                let bar_eighth = curr_eighth % 8;
                if curr_eighth > prev_eighth
                    && kick_positions.contains(&bar_eighth)
                    && seq.profile.bpm > 0.0
                    && has_kick
                {
                    let kick_id = next_node_id;
                    next_node_id += 1;
                    if next_node_id > 9000 { next_node_id = NODE_ID_BASE; }
                    let kick_amp = 0.2 + seq.tension * 0.15; // louder with tension
                    let _ = server.send(
                        SynthNew::new("mkick", GROUP_SOURCES)
                            .synth_id(kick_id)
                            .add_action(AddAction::HeadOfGroup)
                            .controls(vec![
                                Control::new("amp", kick_amp),
                                Control::new("dur", 0.35_f32),
                            ]),
                    );
                }

                // ── Hi-hat: 16th-note grid with accents and ghost notes ──
                let prev_16th = (prev_beat * 4.0).floor() as u32;
                let curr_16th = (seq.beat_counter * 4.0).floor() as u32;
                if curr_16th > prev_16th && seq.profile.bpm > 0.0 {
                    let pos_in_beat = curr_16th % 4; // 0=downbeat, 1=e, 2=and, 3=a
                    // Accent pattern: loud on downbeat, medium on "and", ghost on "e" and "a"
                    let (hat_amp, hat_dur) = match pos_in_beat {
                        0 => (0.09, 0.05),  // downbeat accent
                        2 => (0.06, 0.04),  // "and" — medium
                        1 => (0.025, 0.025), // "e" — ghost note
                        _ => (0.02, 0.02),   // "a" — ghost note, softest
                    };
                    // Skip some ghost notes for variety (60% chance)
                    let play = pos_in_beat < 2 || seq.rng_f32() > 0.4;
                    // In ethereal section, only play downbeat hats
                    let play = play && (section != 1 || pos_in_beat == 0);

                    if play {
                        let hat_id = next_node_id;
                        next_node_id += 1;
                        if next_node_id > 9000 { next_node_id = NODE_ID_BASE; }
                        let _ = server.send(
                            SynthNew::new("mhat", GROUP_SOURCES)
                                .synth_id(hat_id)
                                .add_action(AddAction::HeadOfGroup)
                                .controls(vec![
                                    Control::new("amp", hat_amp),
                                    Control::new("dur", hat_dur),
                                    Control::new("pan", seq.rand_pan() * 0.5),
                                ]),
                        );
                    }
                }

                // ── Melodic note — voice and params vary by section ──
                let midi = seq.next_note();
                let vel = seq.current_velocity();
                let pan = seq.rand_pan();
                let section = seq.section;

                // Section + EEG determine voice.
                // High stability → ethereal/glass; high tension → base voice;
                // low confidence → stick with time-based sections.
                let (def_name, octave_shift): (&str, i32) = if seq.confidence > 0.3 {
                    // EEG-driven voice selection
                    if seq.stability > 0.6 {
                        ("mglass", 12) // calm, stable → ethereal glass, octave up
                    } else if seq.tension > 0.5 {
                        (seq.profile.voice.sc_name(), 0) // tense → punchy base voice
                    } else if seq.motion_y > 0.2 {
                        (seq.profile.voice.sc_name(), -12) // deep/theta → octave down
                    } else {
                        ("mbell", 0) // neutral → bell chimes
                    }
                } else {
                    // Autopilot: time-based sections
                    match section {
                        0 => (seq.profile.voice.sc_name(), 0),
                        1 => ("mglass", 12),
                        2 => (seq.profile.voice.sc_name(), -12),
                        _ => ("mbell", 0),
                    }
                };

                let shifted_midi = (midi as i32 + octave_shift).clamp(24, 108) as u8;
                let freq = 440.0_f32
                    * (2.0_f32).powf((shifted_midi as f32 - 69.0) / 12.0);
                let node_id = next_node_id;
                next_node_id += 1;
                if next_node_id > 9000 { next_node_id = NODE_ID_BASE; }

                let ffreq = seq.profile.params.filter_freq
                    * (0.5 + seq.stability * 1.5);

                // Section-dependent amplitude (quieter in ethereal sections)
                let section_amp = match section {
                    1 => seq.profile.params.amp * 0.6, // bells quieter
                    3 => seq.profile.params.amp * 0.8, // pluck slightly quieter
                    _ => seq.profile.params.amp,
                };

                let mut controls = vec![
                    Control::new("freq", freq),
                    Control::new("amp", section_amp * vel),
                    Control::new("pan", pan),
                ];

                // Voice-specific params
                match def_name {
                    "mpad" | "mdrone" | "mshimmer" => {
                        controls.extend_from_slice(&[
                            Control::new("ffreq", ffreq),
                            Control::new("atk", seq.profile.params.attack),
                            Control::new("rel", seq.profile.params.release),
                            Control::new("detune", seq.profile.params.detune),
                            Control::new("gate", 1.0_f32),
                        ]);
                    }
                    "mrcpluck" | "mlofi" => {
                        controls.extend_from_slice(&[
                            Control::new("vel", vel),
                            Control::new("fbase", ffreq),
                            Control::new("fdecay", 0.3_f32),
                            Control::new("vdecay", seq.profile.params.duration),
                        ]);
                    }
                    "mpluck" => {
                        controls.extend_from_slice(&[
                            Control::new("dur", seq.profile.params.duration * 1.5),
                            Control::new("coef", 0.15_f32),
                        ]);
                    }
                    "mbell" => {
                        controls.push(Control::new("dur", 2.5_f32));
                    }
                    "mpulse" => {
                        controls.push(Control::new("dur", 0.15_f32));
                    }
                    "mglass" => {
                        controls.extend_from_slice(&[
                            Control::new("ffreq", ffreq * 3.0),
                            Control::new("atk", 2.0_f32),
                            Control::new("rel", 4.0_f32),
                            Control::new("gate", 1.0_f32),
                        ]);
                    }
                    _ => {}
                }

                let _ = server.send(
                    SynthNew::new(def_name, GROUP_SOURCES)
                        .synth_id(node_id)
                        .add_action(AddAction::HeadOfGroup)
                        .controls(controls),
                );

                seq.advance();
            }
        }
    }
}

/// Handle a single command.  Returns true if it was Shutdown.
fn handle_command(
    server: &Server,
    cmd: ScCommand,
    next_node_id: &mut i32,
    sustained_nodes: &mut Vec<i32>,
    rng_state: &mut u32,
    sequencer: &mut Option<SequencerState>,
) {
    match cmd {
        ScCommand::PlayChord { midi_notes, voice, params, random_pan } => {
            // Direct play (no sequencer).  Release previous sustained notes.
            for &nid in sustained_nodes.iter() {
                let _ = server.send(NodeSet::new(nid, vec![
                    Control::new("gate", 0.0_f32),
                ]));
            }
            sustained_nodes.clear();

            let def_name = voice.sc_name();
            let n = midi_notes.len().max(1);

            for (i, &midi) in midi_notes.iter().enumerate() {
                let freq = 440.0_f32 * (2.0_f32).powf((midi as f32 - 69.0) / 12.0);
                let node_id = *next_node_id;
                *next_node_id += 1;
                if *next_node_id > 9000 { *next_node_id = NODE_ID_BASE; }

                let pan = if random_pan {
                    *rng_state ^= *rng_state << 13;
                    *rng_state ^= *rng_state >> 17;
                    *rng_state ^= *rng_state << 5;
                    (*rng_state as f32 / u32::MAX as f32) * 1.8 - 0.9
                } else if n > 1 {
                    (i as f32 / (n - 1) as f32) * 1.6 - 0.8
                } else { 0.0 };

                let mut controls = vec![
                    Control::new("freq", freq),
                    Control::new("amp", params.amp / n as f32),
                    Control::new("pan", pan),
                ];

                match voice {
                    Voice::Pad | Voice::Drone => {
                        controls.extend_from_slice(&[
                            Control::new("ffreq", params.filter_freq),
                            Control::new("atk", params.attack),
                            Control::new("rel", params.release),
                            Control::new("detune", params.detune),
                            Control::new("gate", 1.0_f32),
                        ]);
                        sustained_nodes.push(node_id);
                    }
                    Voice::RcPluck | Voice::LoFi => {
                        let vel = params.release.clamp(0.1, 1.0);
                        controls.extend_from_slice(&[
                            Control::new("vel", vel),
                            Control::new("fbase", params.filter_freq),
                            Control::new("fdecay", 0.3_f32),
                            Control::new("vdecay", params.duration),
                        ]);
                    }
                    Voice::Pluck => {
                        controls.extend_from_slice(&[
                            Control::new("dur", params.duration),
                            Control::new("coef", 0.1_f32),
                        ]);
                    }
                    Voice::Bell => {
                        controls.push(Control::new("dur", params.duration));
                    }
                    Voice::Pulse => {
                        controls.push(Control::new("dur", params.duration.min(0.3)));
                    }
                    Voice::Glass => {
                        controls.extend_from_slice(&[
                            Control::new("ffreq", params.filter_freq),
                            Control::new("atk", params.attack),
                            Control::new("rel", params.release),
                            Control::new("gate", 1.0_f32),
                        ]);
                        sustained_nodes.push(node_id);
                    }
                }

                let _ = server.send(
                    SynthNew::new(def_name, GROUP_SOURCES)
                        .synth_id(node_id)
                        .add_action(AddAction::HeadOfGroup)
                        .controls(controls),
                );
            }
        }

        ScCommand::StartSequencer { profile, midi_notes } => {
            // Stop any existing sequencer pad layer
            if let Some(old) = sequencer {
                for &nid in &old.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("gate", 0.0_f32),
                    ]));
                }
            }

            // Apply reverb settings
            let _ = server.send(NodeSet::new(99, vec![
                Control::new("mix", profile.reverb_mix),
                Control::new("room", profile.reverb_room),
                Control::new("damp", profile.reverb_damp),
            ]));

            // Start an ethereal shimmer pad layer + sub bass underneath
            let mut pad_nodes = Vec::new();
            for &midi in &midi_notes {
                let freq = 440.0_f32 * (2.0_f32).powf((midi as f32 - 69.0) / 12.0);
                let nid = *next_node_id;
                *next_node_id += 1;
                if *next_node_id > 9000 { *next_node_id = NODE_ID_BASE; }

                let _ = server.send(
                    SynthNew::new("mglass", GROUP_SOURCES)
                        .synth_id(nid)
                        .add_action(AddAction::TailOfGroup)
                        .controls(vec![
                            Control::new("freq", freq),
                            Control::new("amp", 0.06_f32),
                            Control::new("ffreq", 1500.0_f32),
                            Control::new("atk", 4.0_f32),
                            Control::new("rel", 6.0_f32),
                            Control::new("gate", 1.0_f32),
                        ]),
                );
                pad_nodes.push(nid);
            }
            // Sub bass on the root note (one octave down)
            if let Some(&root) = midi_notes.first() {
                let sub_freq = 440.0_f32 * (2.0_f32).powf((root as f32 - 69.0 - 12.0) / 12.0);
                let nid = *next_node_id;
                *next_node_id += 1;
                if *next_node_id > 9000 { *next_node_id = NODE_ID_BASE; }
                let _ = server.send(
                    SynthNew::new("mdrone", GROUP_SOURCES)
                        .synth_id(nid)
                        .add_action(AddAction::TailOfGroup)
                        .controls(vec![
                            Control::new("freq", sub_freq),
                            Control::new("amp", 0.06_f32),
                            Control::new("ffreq", 250.0_f32),
                            Control::new("gate", 1.0_f32),
                        ]),
                );
                pad_nodes.push(nid);
            }

            let mut seq = SequencerState::new(profile, midi_notes);
            seq.pad_nodes = pad_nodes;
            *sequencer = Some(seq);
            eprintln!("[sc] sequencer started");
        }

        ScCommand::UpdateChord { midi_notes } => {
            if let Some(seq) = sequencer {
                // Release old pad + sub layer, start new
                for &nid in &seq.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("gate", 0.0_f32),
                    ]));
                }
                let mut pad_nodes = Vec::new();
                for &midi in &midi_notes {
                    let freq = 440.0_f32 * (2.0_f32).powf((midi as f32 - 69.0) / 12.0);
                    let nid = *next_node_id;
                    *next_node_id += 1;
                    if *next_node_id > 9000 { *next_node_id = NODE_ID_BASE; }
                    let _ = server.send(
                        SynthNew::new("mglass", GROUP_SOURCES)
                            .synth_id(nid)
                            .add_action(AddAction::TailOfGroup)
                            .controls(vec![
                                Control::new("freq", freq),
                                Control::new("amp", 0.07_f32),
                                Control::new("ffreq", 900.0_f32),
                                Control::new("detune", 0.015_f32),
                                Control::new("atk", 3.0_f32),
                                Control::new("rel", 5.0_f32),
                                Control::new("gate", 1.0_f32),
                            ]),
                    );
                    pad_nodes.push(nid);
                }
                // Sub bass on root
                if let Some(&root) = midi_notes.first() {
                    let sub_freq = 440.0_f32
                        * (2.0_f32).powf((root as f32 - 69.0 - 12.0) / 12.0);
                    let nid = *next_node_id;
                    *next_node_id += 1;
                    if *next_node_id > 9000 { *next_node_id = NODE_ID_BASE; }
                    let _ = server.send(
                        SynthNew::new("mdrone", GROUP_SOURCES)
                            .synth_id(nid)
                            .add_action(AddAction::TailOfGroup)
                            .controls(vec![
                                Control::new("freq", sub_freq),
                                Control::new("amp", 0.06_f32),
                                Control::new("ffreq", 250.0_f32),
                                Control::new("gate", 1.0_f32),
                            ]),
                    );
                    pad_nodes.push(nid);
                }
                seq.pad_nodes = pad_nodes;
                seq.midi_notes = midi_notes;
                seq.note_idx = 0;
            }
        }

        ScCommand::UpdateEeg { tension, stability, motion_x, motion_y, confidence } => {
            if let Some(seq) = sequencer {
                // Smooth EEG values (EMA) to avoid jitter
                let a = 0.15; // smoothing factor
                seq.tension = seq.tension * (1.0 - a) + tension * a;
                seq.stability = seq.stability * (1.0 - a) + stability * a;
                seq.motion_x = seq.motion_x * (1.0 - a) + motion_x * a;
                seq.motion_y = seq.motion_y * (1.0 - a) + motion_y * a;
                seq.confidence = seq.confidence * (1.0 - a) + confidence * a;

                // Live-update the pad layer filter based on stability
                let pad_ffreq = 400.0 + seq.stability * 2000.0;
                for &nid in &seq.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("ffreq", pad_ffreq),
                    ]));
                }

                // Live-update reverb based on motion_y (depth/theta)
                let rev_mix = 0.15 + seq.motion_y.abs() * 0.35;
                let _ = server.send(NodeSet::new(99, vec![
                    Control::new("mix", rev_mix.clamp(0.1, 0.5)),
                ]));
            }
        }

        ScCommand::StopSequencer => {
            if let Some(seq) = sequencer {
                for &nid in &seq.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("gate", 0.0_f32),
                    ]));
                }
            }
            *sequencer = None;
            eprintln!("[sc] sequencer stopped");
        }

        ScCommand::ReleaseAll => {
            for &nid in sustained_nodes.iter() {
                let _ = server.send(NodeSet::new(nid, vec![
                    Control::new("gate", 0.0_f32),
                ]));
            }
            sustained_nodes.clear();
        }

        ScCommand::SetReverb { mix, room, damp } => {
            let _ = server.send(NodeSet::new(99, vec![
                Control::new("mix", mix),
                Control::new("room", room),
                Control::new("damp", damp),
            ]));
        }

        ScCommand::Shutdown => {
            if let Some(seq) = sequencer {
                for &nid in &seq.pad_nodes {
                    let _ = server.send(NodeSet::new(nid, vec![
                        Control::new("gate", 0.0_f32),
                    ]));
                }
            }
            for &nid in sustained_nodes.iter() {
                let _ = server.send(NodeSet::new(nid, vec![
                    Control::new("gate", 0.0_f32),
                ]));
            }
            let _ = server.send(NodeFree::new(vec![99]));
            raw_osc_send("127.0.0.1", 57110, "/g_freeAll",
                vec![OscType::Int(GROUP_SOURCES)]);
            raw_osc_send("127.0.0.1", 57110, "/g_freeAll",
                vec![OscType::Int(GROUP_FX)]);
            // Signal the loop to exit by clearing next_node_id
            *next_node_id = NODE_ID_BASE;
        }
    }
}

pub fn midi_to_freq(midi: u8) -> f32 {
    440.0 * (2.0_f32).powf((midi as f32 - 69.0) / 12.0)
}
