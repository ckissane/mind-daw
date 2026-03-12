//! Latent control state — the bridge between EEG decoding and musical geometry.
//!
//! The decoder populates a `ControlState`, and the music engine consumes only this
//! state, never raw EEG. Confidence scores drive smoothing: low confidence →
//! slower movement, more damping, bias toward current harmonic state.

use crate::calibration::{BandPowers, CalibrationProfile};
use std::time::Instant;

// ── Control state ────────────────────────────────────────────────────────────

/// The compact latent control state that the decoder outputs and the music
/// engine consumes. All fields are continuously updated.
#[derive(Clone, Debug)]
pub struct ControlState {
    /// Continuous 2D motion in orbifold space, each in [-1, 1].
    pub motion_x: f32,
    pub motion_y: f32,

    /// Tension: 0 = maximally relaxed/consonant, 1 = maximally tense/dissonant.
    pub tension: f32,

    /// Stability: 0 = free-flowing/changing, 1 = locked/stable.
    pub stability: f32,

    /// Discrete event flags (consumed once per trigger).
    pub confirm_event: bool,
    pub reset_event: bool,

    /// Freeze: when true, hold current position and chord.
    pub freeze: bool,

    /// Confidence scores for the discrete and continuous channels.
    pub confidence_discrete: f32,
    pub confidence_continuous: f32,

    /// Timestamp of last update.
    pub last_update: Instant,
}

impl Default for ControlState {
    fn default() -> Self {
        Self {
            motion_x: 0.0,
            motion_y: 0.0,
            tension: 0.3,
            stability: 0.5,
            confirm_event: false,
            reset_event: false,
            freeze: false,
            confidence_discrete: 0.0,
            confidence_continuous: 0.0,
            last_update: Instant::now(),
        }
    }
}

impl ControlState {
    /// Overall confidence (geometric mean of discrete and continuous).
    pub fn overall_confidence(&self) -> f32 {
        (self.confidence_discrete * self.confidence_continuous)
            .sqrt()
            .clamp(0.0, 1.0)
    }

    /// Consume the reset event.
    pub fn take_reset(&mut self) -> bool {
        let v = self.reset_event;
        self.reset_event = false;
        v
    }
}

// ── Decoder ──────────────────────────────────────────────────────────────────

/// Transforms EEG band powers + artifact events into a ControlState.
/// Uses calibration profile for per-user normalization.
pub struct ControlDecoder {
    // Exponential moving average state for smoothing
    ema_motion_x: f32,
    ema_motion_y: f32,
    ema_tension: f32,
    ema_stability: f32,
    ema_confidence_c: f32,
    ema_confidence_d: f32,

    // Smoothing factor: higher = more smoothing (slower response)
    alpha_fast: f32,  // for high-confidence updates
    alpha_slow: f32,  // for low-confidence updates

    // Artifact detection state
    blink_cooldown: f32,
    last_blink_time: f32,
    jaw_clench_cooldown: f32,

    // Running time
    session_time: f32,
}

impl ControlDecoder {
    pub fn new() -> Self {
        Self {
            ema_motion_x: 0.0,
            ema_motion_y: 0.0,
            ema_tension: 0.3,
            ema_stability: 0.5,
            ema_confidence_c: 0.0,
            ema_confidence_d: 0.0,
            alpha_fast: 0.15,
            alpha_slow: 0.03,
            blink_cooldown: 0.0,
            last_blink_time: -10.0,
            jaw_clench_cooldown: 0.0,
            session_time: 0.0,
        }
    }

    /// Main decode step. Takes raw band powers, artifact signals, and calibration profile.
    /// Returns an updated ControlState.
    ///
    /// `raw_bands` — current band powers (delta, theta, alpha, beta, gamma)
    /// `blink_detected` — true if a blink artifact was detected this frame
    /// `jaw_clench_detected` — true if a jaw clench was detected this frame
    /// `channel_variance` — current per-channel variance (for confidence estimation)
    /// `profile` — calibration profile for normalization (optional)
    /// `dt` — time delta in seconds
    pub fn decode(
        &mut self,
        raw_bands: &BandPowers,
        blink_detected: bool,
        jaw_clench_detected: bool,
        channel_variance: f32,
        profile: Option<&CalibrationProfile>,
        dt: f32,
        state: &mut ControlState,
    ) {
        self.session_time += dt;
        self.blink_cooldown = (self.blink_cooldown - dt).max(0.0);
        self.jaw_clench_cooldown = (self.jaw_clench_cooldown - dt).max(0.0);

        // ── Normalize band powers ────────────────────────────────────────
        let norm_bands = if let Some(profile) = profile {
            profile.normalize_bands(raw_bands)
        } else {
            // Without calibration, use simple ratio normalization
            let total = raw_bands.delta + raw_bands.theta + raw_bands.alpha
                + raw_bands.beta + raw_bands.gamma;
            let t = total.max(1e-6);
            BandPowers {
                delta: raw_bands.delta / t,
                theta: raw_bands.theta / t,
                alpha: raw_bands.alpha / t,
                beta: raw_bands.beta / t,
                gamma: raw_bands.gamma / t,
            }
        };

        // ── Estimate confidence ──────────────────────────────────────────
        let noise_level = if let Some(profile) = profile {
            // Compare current variance to calibrated noise floor
            let avg_noise = if profile.noise_floor.is_empty() {
                1.0
            } else {
                profile.noise_floor.iter().sum::<f32>() / profile.noise_floor.len() as f32
            };
            (channel_variance / avg_noise.max(1e-6)).clamp(0.1, 10.0)
        } else {
            1.0
        };

        // Confidence decreases when noise is much higher than baseline
        let raw_confidence_c = (1.0 / noise_level).clamp(0.0, 1.0);
        let raw_confidence_d = raw_confidence_c * 0.8; // discrete is harder

        // Smooth confidence to avoid jitter
        self.ema_confidence_c += 0.1 * (raw_confidence_c - self.ema_confidence_c);
        self.ema_confidence_d += 0.1 * (raw_confidence_d - self.ema_confidence_d);

        // ── Adaptive smoothing based on confidence ───────────────────────
        let alpha = lerp(self.alpha_slow, self.alpha_fast, self.ema_confidence_c);

        // ── Map band powers to control dimensions ────────────────────────
        //
        // motion_x ← alpha/beta ratio (relaxation axis)
        //   High alpha, low beta → positive (relaxed, rightward)
        //   Low alpha, high beta → negative (focused, leftward)
        //
        // motion_y ← theta (arousal/depth axis)
        //   High theta → positive (deeper, upward)
        //
        // tension ← beta + gamma (activation → dissonance)
        //
        // stability ← alpha dominance (calm → consonance, stability)

        let alpha_beta_ratio = if profile.is_some() {
            // Z-scored: positive = more alpha than usual
            (norm_bands.alpha - norm_bands.beta).clamp(-3.0, 3.0) / 3.0
        } else {
            // Ratio-based fallback
            (norm_bands.alpha - norm_bands.beta).clamp(-1.0, 1.0)
        };

        let theta_signal = if profile.is_some() {
            norm_bands.theta.clamp(-3.0, 3.0) / 3.0
        } else {
            (norm_bands.theta * 4.0 - 1.0).clamp(-1.0, 1.0)
        };

        let raw_tension = if profile.is_some() {
            let activation = norm_bands.beta + norm_bands.gamma * 0.5;
            (activation / 4.0).clamp(0.0, 1.0)
        } else {
            (norm_bands.beta + norm_bands.gamma).clamp(0.0, 1.0)
        };

        let raw_stability = if profile.is_some() {
            let calm = norm_bands.alpha - (norm_bands.beta + norm_bands.gamma) * 0.3;
            ((calm + 2.0) / 4.0).clamp(0.0, 1.0)
        } else {
            let calm = norm_bands.alpha / (norm_bands.beta + norm_bands.gamma + 0.01);
            calm.clamp(0.0, 1.0)
        };

        // ── Apply EMA smoothing ──────────────────────────────────────────
        self.ema_motion_x += alpha * (alpha_beta_ratio - self.ema_motion_x);
        self.ema_motion_y += alpha * (theta_signal - self.ema_motion_y);
        self.ema_tension += alpha * (raw_tension - self.ema_tension);
        self.ema_stability += alpha * (raw_stability - self.ema_stability);

        // ── Artifact-based events ────────────────────────────────────────
        if blink_detected && self.blink_cooldown <= 0.0 {
            let time_since_last = self.session_time - self.last_blink_time;
            if time_since_last < 0.8 && time_since_last > 0.15 {
                // Double blink → confirm event
                state.confirm_event = true;
                self.blink_cooldown = 1.0;
            }
            self.last_blink_time = self.session_time;
        }

        if jaw_clench_detected && self.jaw_clench_cooldown <= 0.0 {
            // Jaw clench → toggle freeze
            state.freeze = !state.freeze;
            self.jaw_clench_cooldown = 1.5;
        }

        // ── Write to control state ───────────────────────────────────────
        if !state.freeze {
            state.motion_x = self.ema_motion_x;
            state.motion_y = self.ema_motion_y;
            state.tension = self.ema_tension;
            state.stability = self.ema_stability;
        }
        state.confidence_continuous = self.ema_confidence_c;
        state.confidence_discrete = self.ema_confidence_d;
        state.last_update = Instant::now();
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// ── Feature extraction helpers ───────────────────────────────────────────────

/// Extract band powers from a multi-channel EEG feature vector.
/// `features` is the PCA-style feature vector (64 channels × 32 bins).
/// Returns band powers averaged over the first `n_channels` channels.
pub fn extract_band_powers(features: &[f32], bins_per_ch: usize, n_channels: usize) -> BandPowers {
    if features.is_empty() || bins_per_ch < 8 {
        return BandPowers::default();
    }

    let ch_limit = n_channels.min(features.len() / bins_per_ch).min(16);
    let mut delta = 0.0f32;
    let mut theta = 0.0f32;
    let mut alpha = 0.0f32;
    let mut beta = 0.0f32;
    let mut gamma = 0.0f32;

    // Approximate bin-to-band mapping for 300 Hz sample rate, 64-sample FFT:
    // bin_hz ≈ 300/64 ≈ 4.7 Hz per bin
    // bin 0: DC, bin 1: ~4.7 Hz, bin 2: ~9.4 Hz, bin 3: ~14.1 Hz, etc.
    // But for 128-sample FFT: bin_hz ≈ 2.3 Hz
    // We use fractional bin ranges for generality.
    for ch in 0..ch_limit {
        let offset = ch * bins_per_ch;
        // Delta (0.5-4 Hz): bins ~0-1
        for b in 0..=1.min(bins_per_ch - 1) {
            delta += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        // Theta (4-8 Hz): bins ~1-2
        for b in 1..=2.min(bins_per_ch - 1) {
            theta += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        // Alpha (8-13 Hz): bins ~2-3
        for b in 2..=3.min(bins_per_ch - 1) {
            alpha += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        // Beta (13-30 Hz): bins ~3-6
        for b in 3..=6.min(bins_per_ch - 1) {
            beta += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        // Gamma (30-80 Hz): bins ~7-17
        for b in 7..=17.min(bins_per_ch - 1) {
            gamma += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
    }

    let n = ch_limit.max(1) as f32;
    BandPowers {
        delta: delta / n,
        theta: theta / n,
        alpha: alpha / n,
        beta: beta / n,
        gamma: gamma / n,
    }
}

/// Simple blink detector: checks if any frontal channel has a sample exceeding
/// the blink amplitude threshold from the calibration profile.
pub fn detect_blink(channel_data: &[Vec<f32>], profile: Option<&CalibrationProfile>) -> bool {
    let threshold = profile
        .map(|p| p.blink_amplitude)
        .unwrap_or(150.0);

    // Check frontal channels (0-7)
    for ch in 0..8.min(channel_data.len()) {
        let data = &channel_data[ch];
        if data.len() < 3 {
            continue;
        }
        // Look at the last few samples for a large deflection
        let recent = &data[data.len().saturating_sub(5)..];
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        for &x in recent {
            if (x - mean).abs() > threshold {
                return true;
            }
        }
    }
    false
}

/// Simple jaw clench detector: checks for high-frequency energy spike.
pub fn detect_jaw_clench(channel_data: &[Vec<f32>], profile: Option<&CalibrationProfile>) -> bool {
    let threshold = profile
        .map(|p| p.jaw_clench_amplitude)
        .unwrap_or(100.0);

    // Check temporal channels
    for ch in 0..channel_data.len().min(16) {
        let data = &channel_data[ch];
        if data.len() < 5 {
            continue;
        }
        // High-frequency energy = large sample-to-sample differences
        let recent = &data[data.len().saturating_sub(8)..];
        for i in 1..recent.len() {
            if (recent[i] - recent[i - 1]).abs() > threshold {
                return true;
            }
        }
    }
    false
}
