//! Signal-to-parameter mapping data model for the Neural Instrument.
//!
//! A `NeuralPatch` is a list of `SignalMapping` entries, each routing one
//! EEG-derived signal to one SuperCollider synthesis parameter via OSC.
//! Patches are serialized to `profiles/{name}/patch.json`.

use serde::{Deserialize, Serialize};

// ── Signal sources ────────────────────────────────────────────────────────────

/// EEG/EMG-derived signal that can drive a synthesis parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignalSource {
    // ── Continuous signals (normalized 0–1 or -1–1) ───────────────────────
    /// Global occipital alpha power (8–13 Hz), normalized to baseline.
    AlphaPower,
    /// Frontal alpha asymmetry: ln(R) − ln(L) on AFF4/AFF3.
    /// Range: −1.0 (left-dominant / withdrawal) → +1.0 (right-dominant / approach).
    AlphaAsymmetry,
    /// Global frontal beta power (13–30 Hz), normalized.
    BetaPower,
    /// Frontal beta asymmetry: ln(R) − ln(L) on AFF4/AFF3.
    BetaAsymmetry,
    /// Frontal-midline theta power (4–8 Hz) on FFCz (≈Fz), normalized.
    ThetaPower,
    /// Engagement index: β / (α + θ), normalized.
    Engagement,
    // ── Discrete / event signals ──────────────────────────────────────────
    /// Single jaw clench (< 600 ms) detected on lateral temporal channels.
    JawSingle,
    /// Double jaw clench (two clenches within 800 ms).
    JawDouble,
    /// Jaw clench duration, normalized over 0–2000 ms → 0.0–1.0.
    JawDuration,
    /// Single intentional slow blink (300–700 ms) on AFp3/AFp4.
    BlinkSingle,
    /// Double intentional blink (two blinks within 1000 ms).
    BlinkDouble,
    /// Sustained intentional blink rate (blinks/min), normalized over 0–30.
    BlinkRate,
}

impl SignalSource {
    /// Returns `true` for continuously-valued signals (0–1 or -1–1).
    /// Returns `false` for discrete event signals.
    pub fn is_continuous(&self) -> bool {
        matches!(
            self,
            Self::AlphaPower
                | Self::AlphaAsymmetry
                | Self::BetaPower
                | Self::BetaAsymmetry
                | Self::ThetaPower
                | Self::Engagement
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::AlphaPower     => "α power",
            Self::AlphaAsymmetry => "α asymmetry",
            Self::BetaPower      => "β power",
            Self::BetaAsymmetry  => "β asymmetry",
            Self::ThetaPower     => "θ power (Fz)",
            Self::Engagement     => "engagement",
            Self::JawSingle      => "jaw clench ×1",
            Self::JawDouble      => "jaw clench ×2",
            Self::JawDuration    => "jaw duration",
            Self::BlinkSingle    => "blink ×1",
            Self::BlinkDouble    => "blink ×2",
            Self::BlinkRate      => "blink rate",
        }
    }

    /// Default input normalization range for this signal.
    pub fn default_input_range(&self) -> (f32, f32) {
        match self {
            Self::AlphaAsymmetry | Self::BetaAsymmetry => (-1.0, 1.0),
            _ => (0.0, 1.0),
        }
    }

    pub fn all_continuous() -> Vec<SignalSource> {
        vec![
            Self::AlphaPower,
            Self::AlphaAsymmetry,
            Self::BetaPower,
            Self::BetaAsymmetry,
            Self::ThetaPower,
            Self::Engagement,
        ]
    }

    pub fn all_events() -> Vec<SignalSource> {
        vec![
            Self::JawSingle,
            Self::JawDouble,
            Self::JawDuration,
            Self::BlinkSingle,
            Self::BlinkDouble,
            Self::BlinkRate,
        ]
    }
}

// ── Parameter targets ─────────────────────────────────────────────────────────

/// Synthesis parameter in SuperCollider that a signal can control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterTarget {
    // ── Tonnetz (colleague's engine) ──────────────────────────────────────
    /// Horizontal axis of the Tonnetz orbifold: minor/dark ↔ major/bright.
    TonnetzAxis,
    /// Harmonic tension: distance from tonic in the orbifold.
    TonnetzTension,
    // ── Arpeggio ──────────────────────────────────────────────────────────
    /// Arpeggio playback rate (notes/second).
    ArpeggioSpeed,
    /// Arpeggio direction (up / up-down / random, determined by threshold).
    ArpeggioDirection,
    // ── Waveform synth ────────────────────────────────────────────────────
    /// Synth waveform: 0=sine, 1=triangle, 2=saw, 3=square.
    WaveformShape,
    // ── Bass ──────────────────────────────────────────────────────────────
    /// Bass filter cutoff frequency (Hz).
    BassFilter,
    // ── Event / toggle targets ────────────────────────────────────────────
    /// Toggle drums on/off.
    DrumsToggle,
    /// Cycle drum pattern (4-on-floor → breakbeat → half-time → off).
    DrumsPattern,
    /// Toggle bass on/off.
    BassToggle,
    /// Cycle bass pattern (root-only → walking → syncopated).
    BassPattern,
}

impl ParameterTarget {
    pub fn is_continuous(&self) -> bool {
        matches!(
            self,
            Self::TonnetzAxis
                | Self::TonnetzTension
                | Self::ArpeggioSpeed
                | Self::ArpeggioDirection
                | Self::WaveformShape
                | Self::BassFilter
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::TonnetzAxis       => "Tonnetz L/R axis",
            Self::TonnetzTension    => "harmonic tension",
            Self::ArpeggioSpeed     => "arpeggio speed",
            Self::ArpeggioDirection => "arpeggio direction",
            Self::WaveformShape     => "waveform shape",
            Self::BassFilter        => "bass filter",
            Self::DrumsToggle       => "drums on/off",
            Self::DrumsPattern      => "drum pattern",
            Self::BassToggle        => "bass on/off",
            Self::BassPattern       => "bass pattern",
        }
    }

    pub fn module(&self) -> Module {
        match self {
            Self::TonnetzAxis | Self::TonnetzTension => Module::Tonnetz,
            Self::ArpeggioSpeed | Self::ArpeggioDirection => Module::Arpeggio,
            Self::WaveformShape => Module::Waveform,
            Self::DrumsToggle | Self::DrumsPattern => Module::Drums,
            Self::BassToggle | Self::BassPattern | Self::BassFilter => Module::Bass,
        }
    }

    /// Sensible default output range for this parameter.
    pub fn default_output_range(&self) -> (f32, f32) {
        match self {
            Self::TonnetzAxis       => (-1.0, 1.0),
            Self::TonnetzTension    => (0.0, 1.0),
            Self::ArpeggioSpeed     => (0.5, 8.0),
            Self::ArpeggioDirection => (0.0, 1.0),
            Self::WaveformShape     => (0.0, 3.0),
            Self::BassFilter        => (200.0, 8000.0),
            _                       => (0.0, 1.0),
        }
    }

    /// OSC address this parameter maps to in SuperCollider.
    pub fn osc_address(&self) -> &'static str {
        match self {
            Self::TonnetzAxis       => "/instrument/tonnetz/axis",
            Self::TonnetzTension    => "/instrument/tonnetz/tension",
            Self::ArpeggioSpeed     => "/instrument/arp/speed",
            Self::ArpeggioDirection => "/instrument/arp/direction",
            Self::WaveformShape     => "/instrument/synth/waveform",
            Self::BassFilter        => "/instrument/bass/filter",
            Self::DrumsToggle       => "/instrument/drums/toggle",
            Self::DrumsPattern      => "/instrument/drums/pattern",
            Self::BassToggle        => "/instrument/bass/toggle",
            Self::BassPattern       => "/instrument/bass/pattern",
        }
    }
}

// ── Modules ───────────────────────────────────────────────────────────────────

/// A SuperCollider synthesis module grouping related parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Module {
    Tonnetz,
    Arpeggio,
    Waveform,
    Drums,
    Bass,
}

impl Module {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Tonnetz  => "TONNETZ",
            Self::Arpeggio => "ARPEGGIO",
            Self::Waveform => "WAVEFORM",
            Self::Drums    => "DRUMS",
            Self::Bass     => "BASS",
        }
    }

    pub fn all() -> Vec<Module> {
        vec![
            Self::Tonnetz,
            Self::Arpeggio,
            Self::Waveform,
            Self::Drums,
            Self::Bass,
        ]
    }

    pub fn parameters(&self) -> Vec<ParameterTarget> {
        match self {
            Self::Tonnetz  => vec![ParameterTarget::TonnetzAxis, ParameterTarget::TonnetzTension],
            Self::Arpeggio => vec![ParameterTarget::ArpeggioSpeed, ParameterTarget::ArpeggioDirection],
            Self::Waveform => vec![ParameterTarget::WaveformShape],
            Self::Drums    => vec![ParameterTarget::DrumsToggle, ParameterTarget::DrumsPattern],
            Self::Bass     => vec![
                ParameterTarget::BassToggle,
                ParameterTarget::BassPattern,
                ParameterTarget::BassFilter,
            ],
        }
    }

    /// Emoji icon for display.
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Tonnetz  => "🎵",
            Self::Arpeggio => "🎹",
            Self::Waveform => "🌊",
            Self::Drums    => "🥁",
            Self::Bass     => "🎸",
        }
    }
}

// ── Curve ─────────────────────────────────────────────────────────────────────

/// Response curve applied when mapping input → output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Curve {
    /// Output = input (identity).
    Linear,
    /// Output = input² (slow start, fast end).
    Exponential,
    /// Output = √input (fast start, slow end).
    Logarithmic,
    /// Output = 1 − input (invert).
    Inverted,
    /// Quantize output to N discrete steps.
    Stepped(u32),
}

impl Curve {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Linear      => "linear",
            Self::Exponential => "exp",
            Self::Logarithmic => "log",
            Self::Inverted    => "inverted",
            Self::Stepped(_)  => "stepped",
        }
    }

    /// Apply curve to a normalized value t ∈ [0, 1].
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear      => t,
            Self::Exponential => t * t,
            Self::Logarithmic => t.sqrt(),
            Self::Inverted    => 1.0 - t,
            Self::Stepped(n)  => {
                if *n <= 1 { return t; }
                (t * *n as f32).floor() / (*n - 1) as f32
            }
        }
    }

    pub fn all() -> Vec<Curve> {
        vec![
            Self::Linear,
            Self::Exponential,
            Self::Logarithmic,
            Self::Inverted,
            Self::Stepped(4),
        ]
    }
}

// ── SignalMapping ─────────────────────────────────────────────────────────────

/// One routing: a signal source → a synthesis parameter, with scaling and smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMapping {
    pub source: SignalSource,
    pub target: ParameterTarget,
    /// Clamp/normalize the raw signal within this range before applying curve.
    pub input_range: (f32, f32),
    /// Scale the curved value to this output range.
    pub output_range: (f32, f32),
    pub curve: Curve,
    /// Slew-limiter time constant in milliseconds (0 = no smoothing).
    pub smooth_ms: f32,
    pub enabled: bool,
    // ── Runtime state (not serialized) ────────────────────────────────────
    /// Last raw value received from the signal source (before mapping).
    #[serde(skip)]
    pub current_raw: f32,
    /// Last smoothed output value (after full mapping pipeline).
    #[serde(skip)]
    pub smoothed_value: f32,
}

impl SignalMapping {
    pub fn new(source: SignalSource, target: ParameterTarget) -> Self {
        let input_range  = source.default_input_range();
        let output_range = target.default_output_range();
        Self {
            source,
            target,
            input_range,
            output_range,
            curve: Curve::Linear,
            smooth_ms: 100.0,
            enabled: true,
            current_raw: 0.0,
            smoothed_value: 0.0,
        }
    }

    /// Full mapping pipeline: raw signal → clamped → curved → scaled → smoothed.
    /// `dt_ms` = elapsed time since last call (for the slew limiter).
    /// Returns the smoothed output value.
    pub fn process(&mut self, raw: f32, dt_ms: f32) -> f32 {
        if !self.enabled {
            return self.smoothed_value;
        }
        self.current_raw = raw;

        // 1. Normalize input to 0..1
        let (in_min, in_max) = self.input_range;
        let span = (in_max - in_min).abs().max(1e-10);
        let normalized = ((raw - in_min) / span).clamp(0.0, 1.0);

        // 2. Apply response curve
        let curved = self.curve.apply(normalized);

        // 3. Scale to output range
        let (out_min, out_max) = self.output_range;
        let scaled = out_min + curved * (out_max - out_min);

        // 4. Slew limiter (exponential moving average)
        let alpha = if self.smooth_ms > 0.0 {
            1.0 - (-dt_ms / self.smooth_ms).exp()
        } else {
            1.0
        };
        self.smoothed_value += alpha * (scaled - self.smoothed_value);

        self.smoothed_value
    }
}

// ── NeuralPatch ───────────────────────────────────────────────────────────────

/// A named collection of signal mappings saved per person/session.
/// Serialized to `profiles/{name}/patch.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPatch {
    pub name: String,
    pub mappings: Vec<SignalMapping>,
    /// SuperCollider host (default: loopback).
    pub sc_host: String,
    /// SuperCollider port (sclang default: 57120).
    pub sc_port: u16,
}

impl Default for NeuralPatch {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            mappings: vec![
                // Suggested default mappings based on signal-parameter affinity
                SignalMapping::new(SignalSource::AlphaAsymmetry, ParameterTarget::TonnetzAxis),
                SignalMapping::new(SignalSource::AlphaPower,     ParameterTarget::TonnetzTension),
                SignalMapping::new(SignalSource::BetaPower,      ParameterTarget::ArpeggioSpeed),
                SignalMapping::new(SignalSource::BetaAsymmetry,  ParameterTarget::ArpeggioDirection),
                SignalMapping::new(SignalSource::ThetaPower,     ParameterTarget::WaveformShape),
                SignalMapping::new(SignalSource::JawSingle,      ParameterTarget::DrumsToggle),
                SignalMapping::new(SignalSource::BlinkSingle,    ParameterTarget::BassToggle),
            ],
            sc_host: "127.0.0.1".to_string(),
            sc_port: 57120,
        }
    }
}
