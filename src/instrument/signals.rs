//! Real-time EEG signal state, shared between the EEG processing thread, UI, and OSC sender.
//!
//! `LiveProcessor` ingests raw 64-channel frames and maintains a rolling
//! window from which it computes band powers, asymmetry, engagement index,
//! jaw-clench events, and blink events every ~50 ms.
//!
//! All output is written to `LiveSignals`, which is wrapped in `SharedSignals`
//! (Arc<Mutex<>>) so any thread can read the latest values.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rustfft::{num_complex::Complex, FftPlanner};

use crate::recorder::features::{
    BLINK_CHANNELS, CZ_CHANNEL, FZ_CHANNEL, JAW_CHANNELS,
    LEFT_FRONTAL_CHANNEL, RIGHT_FRONTAL_CHANNEL,
};

use super::mapping::SignalSource;

// ── Constants ─────────────────────────────────────────────────────────────────

const SAMPLE_RATE: f32 = 300.0;
const N_CH: usize = 64;
/// FFT size for live band-power estimation (~1 s of data at 300 Hz).
const FFT_SIZE: usize = 256;
/// Process a new window every this many samples (~50 ms at 300 Hz).
const HOP_SAMPLES: usize = 15;

/// EMG high-frequency band for jaw-clench detection (Hz).
const JAW_EMG_LO: f32 = 80.0;
const JAW_EMG_HI: f32 = 200.0;
/// RMS threshold above which a jaw clench is considered active (normalised units).
const JAW_THRESHOLD: f32 = 0.15;
/// A clench shorter than this (ms) is ignored as noise.
const JAW_MIN_MS: f32 = 80.0;
/// Single clench: shorter than this (ms). Longer → counts towards duration signal.
const JAW_SINGLE_MAX_MS: f32 = 600.0;
/// Maximum gap between two single clenches to count as a double (ms).
const JAW_DOUBLE_WINDOW_MS: f32 = 800.0;

/// Peak-to-peak amplitude threshold for blink detection (µV equivalent units).
const BLINK_AMPLITUDE_THRESHOLD: f32 = 80.0;
/// A blink shorter than this (ms) is a natural blink — ignore.
const BLINK_NATURAL_MAX_MS: f32 = 300.0;
/// A blink longer than this (ms) is not a blink.
const BLINK_MAX_MS: f32 = 700.0;
/// Maximum gap between two blinks to count as double blink (ms).
const BLINK_DOUBLE_WINDOW_MS: f32 = 1000.0;

// ── LiveSignals ───────────────────────────────────────────────────────────────

/// Snapshot of all EEG-derived signals, updated ~20× per second.
/// Read by the UI (live bars) and OSC sender (send to SuperCollider).
#[derive(Debug, Clone, Default)]
pub struct LiveSignals {
    // ── Continuous 0.0–1.0 ───────────────────────────────────────────────
    /// Occipital alpha power (8–13 Hz), baseline-normalised.
    pub alpha_power: f32,
    /// Frontal beta power (13–30 Hz), baseline-normalised.
    pub beta_power: f32,
    /// Frontal-midline theta power on FFCz (4–8 Hz), baseline-normalised.
    pub theta_power: f32,
    /// Engagement index β / (α + θ), normalised.
    pub engagement: f32,

    // ── Asymmetry −1.0–+1.0 ──────────────────────────────────────────────
    /// Frontal alpha asymmetry: ln(R) − ln(L) on AFF4/AFF3.
    /// Positive = right-dominant (approach). Negative = left-dominant (withdrawal).
    pub alpha_asymmetry: f32,
    /// Frontal beta asymmetry: same channels, beta band.
    pub beta_asymmetry: f32,

    // ── Event flags (true for one processing tick, then cleared) ─────────
    pub jaw_single: bool,
    pub jaw_double: bool,
    /// Duration of last jaw clench, normalised 0–1 over 0–2000 ms.
    pub jaw_duration: f32,
    pub blink_single: bool,
    pub blink_double: bool,
    /// Intentional blink rate (blinks/min), normalised over 0–30 bpm.
    pub blink_rate: f32,

    /// Total sample frames processed.
    pub sample_count: u64,
}

impl LiveSignals {
    /// Look up the current value for any `SignalSource`.
    pub fn get(&self, source: &SignalSource) -> f32 {
        match source {
            SignalSource::AlphaPower     => self.alpha_power,
            SignalSource::AlphaAsymmetry => self.alpha_asymmetry,
            SignalSource::BetaPower      => self.beta_power,
            SignalSource::BetaAsymmetry  => self.beta_asymmetry,
            SignalSource::ThetaPower     => self.theta_power,
            SignalSource::Engagement     => self.engagement,
            SignalSource::JawSingle      => if self.jaw_single  { 1.0 } else { 0.0 },
            SignalSource::JawDouble      => if self.jaw_double  { 1.0 } else { 0.0 },
            SignalSource::JawDuration    => self.jaw_duration,
            SignalSource::BlinkSingle    => if self.blink_single { 1.0 } else { 0.0 },
            SignalSource::BlinkDouble    => if self.blink_double { 1.0 } else { 0.0 },
            SignalSource::BlinkRate      => (self.blink_rate / 30.0).clamp(0.0, 1.0),
        }
    }

    /// Clear one-shot event flags after they've been read (call once per UI/OSC tick).
    pub fn clear_events(&mut self) {
        self.jaw_single   = false;
        self.jaw_double   = false;
        self.blink_single = false;
        self.blink_double = false;
    }
}

/// Thread-safe handle to the latest `LiveSignals`.
pub type SharedSignals = Arc<Mutex<LiveSignals>>;

// ── LiveProcessor ─────────────────────────────────────────────────────────────

/// Ingests raw EEG frames and updates `SharedSignals` every ~50 ms.
///
/// Usage:
/// ```
/// let signals = LiveProcessor::new_shared();
/// // On every EEG frame:
/// processor.push_frame(&frame);
/// // In UI / OSC loop:
/// let snap = signals.lock().unwrap().clone();
/// ```
pub struct LiveProcessor {
    /// Shared output written after every `HOP_SAMPLES` frames.
    pub signals: SharedSignals,
    /// Rolling ring buffer of raw frames (capacity = FFT_SIZE).
    ring: VecDeque<[f32; N_CH]>,
    /// Sample counter within the current hop.
    hop_counter: usize,

    // ── Jaw clench state ─────────────────────────────────────────────────
    /// Whether a clench is currently active.
    jaw_active: bool,
    /// Timestamp (in samples) when the current clench started.
    jaw_onset_sample: u64,
    /// Timestamp of the last completed single clench (for double detection).
    jaw_last_single_sample: Option<u64>,
    /// Total samples processed (used as a clock).
    total_samples: u64,

    // ── Blink state ───────────────────────────────────────────────────────
    blink_active: bool,
    blink_onset_sample: u64,
    blink_last_single_sample: Option<u64>,
    /// History of intentional blink timestamps for rate estimation.
    blink_history: VecDeque<u64>,

    // ── Baseline normalization references ─────────────────────────────────
    /// Baseline alpha power per channel — used to normalize live values.
    /// If None, raw (un-normalized) power is used.
    baseline_alpha: Option<Vec<f32>>,
    baseline_beta:  Option<Vec<f32>>,
    baseline_theta: Option<Vec<f32>>,
}

impl LiveProcessor {
    /// Create a new processor and return it paired with the `SharedSignals` it writes.
    pub fn new() -> (Self, SharedSignals) {
        let signals = Arc::new(Mutex::new(LiveSignals::default()));
        let proc = Self {
            signals: Arc::clone(&signals),
            ring: VecDeque::with_capacity(FFT_SIZE + HOP_SAMPLES),
            hop_counter: 0,
            jaw_active: false,
            jaw_onset_sample: 0,
            jaw_last_single_sample: None,
            total_samples: 0,
            blink_active: false,
            blink_onset_sample: 0,
            blink_last_single_sample: None,
            blink_history: VecDeque::with_capacity(30),
            baseline_alpha: None,
            baseline_beta:  None,
            baseline_theta: None,
        };
        (proc, signals)
    }

    /// Optionally load baseline band powers for normalization.
    pub fn set_baseline(&mut self, mean_band_powers: &[[f32; 5]]) {
        self.baseline_alpha = Some(mean_band_powers.iter().map(|p| p[2].max(1e-10)).collect());
        self.baseline_beta  = Some(mean_band_powers.iter().map(|p| p[3].max(1e-10)).collect());
        self.baseline_theta = Some(mean_band_powers.iter().map(|p| p[1].max(1e-10)).collect());
    }

    /// Push one 64-channel frame. Triggers processing every `HOP_SAMPLES` frames.
    pub fn push_frame(&mut self, frame: &[f32; N_CH]) {
        self.ring.push_back(*frame);
        if self.ring.len() > FFT_SIZE {
            self.ring.pop_front();
        }

        // Jaw clench: check on every sample (needs tight timing)
        self.update_jaw(frame);
        // Blink: check on every sample
        self.update_blink(frame);

        self.total_samples += 1;
        self.hop_counter += 1;

        if self.hop_counter >= HOP_SAMPLES && self.ring.len() >= FFT_SIZE / 2 {
            self.hop_counter = 0;
            self.compute_and_publish();
        }
    }

    // ── Jaw clench detection ──────────────────────────────────────────────

    fn update_jaw(&mut self, frame: &[f32; N_CH]) {
        // RMS amplitude on lateral temporal channels as EMG proxy.
        // (True EMG detection would use 80–200 Hz bandpass, but this
        //  is computed per-window below for efficiency; here we use raw amplitude.)
        let rms = JAW_CHANNELS
            .iter()
            .map(|&ch| frame[ch].powi(2))
            .sum::<f32>()
            .sqrt()
            / JAW_CHANNELS.len() as f32;

        // Simple threshold-based state machine.
        let is_high = rms > JAW_THRESHOLD * 100.0; // scale to raw µV range

        let samples_to_ms = |s: u64| (s as f32 / SAMPLE_RATE) * 1000.0;

        if is_high && !self.jaw_active {
            // Clench onset
            self.jaw_active = true;
            self.jaw_onset_sample = self.total_samples;
        } else if !is_high && self.jaw_active {
            // Clench offset — classify
            self.jaw_active = false;
            let duration_ms = samples_to_ms(self.total_samples - self.jaw_onset_sample);

            if duration_ms < JAW_MIN_MS {
                return; // noise
            }

            let norm_duration = (duration_ms / 2000.0).clamp(0.0, 1.0);

            if duration_ms <= JAW_SINGLE_MAX_MS {
                // Check for double clench
                let is_double = self.jaw_last_single_sample.map_or(false, |prev| {
                    samples_to_ms(self.jaw_onset_sample - prev) < JAW_DOUBLE_WINDOW_MS
                });

                if let Ok(mut sig) = self.signals.lock() {
                    if is_double {
                        sig.jaw_double = true;
                        sig.jaw_duration = norm_duration;
                    } else {
                        sig.jaw_single = true;
                        sig.jaw_duration = norm_duration;
                    }
                }
                self.jaw_last_single_sample = Some(self.jaw_onset_sample);
            } else {
                // Long clench — just report duration
                if let Ok(mut sig) = self.signals.lock() {
                    sig.jaw_duration = norm_duration;
                }
            }
        }
    }

    // ── Blink detection ───────────────────────────────────────────────────

    fn update_blink(&mut self, frame: &[f32; N_CH]) {
        // Average absolute amplitude on frontal-polar channels (EOG proxy).
        let amp = BLINK_CHANNELS
            .iter()
            .map(|&ch| frame[ch].abs())
            .sum::<f32>()
            / BLINK_CHANNELS.len() as f32;

        let is_high = amp > BLINK_AMPLITUDE_THRESHOLD;
        let samples_to_ms = |s: u64| (s as f32 / SAMPLE_RATE) * 1000.0;

        if is_high && !self.blink_active {
            self.blink_active = true;
            self.blink_onset_sample = self.total_samples;
        } else if !is_high && self.blink_active {
            self.blink_active = false;
            let duration_ms = samples_to_ms(self.total_samples - self.blink_onset_sample);

            // Natural blink: < BLINK_NATURAL_MAX_MS → ignore
            if duration_ms < BLINK_NATURAL_MAX_MS || duration_ms > BLINK_MAX_MS {
                return;
            }

            // Intentional slow blink detected
            let is_double = self.blink_last_single_sample.map_or(false, |prev| {
                samples_to_ms(self.blink_onset_sample - prev) < BLINK_DOUBLE_WINDOW_MS
            });

            // Record for rate estimation (keep last 60 s worth)
            let cutoff = self.total_samples.saturating_sub((60.0 * SAMPLE_RATE) as u64);
            self.blink_history.retain(|&s| s >= cutoff);
            self.blink_history.push_back(self.blink_onset_sample);

            // Blink rate: blinks in last 60 s → blinks/min
            let blink_rate = self.blink_history.len() as f32;

            if let Ok(mut sig) = self.signals.lock() {
                if is_double {
                    sig.blink_double = true;
                } else {
                    sig.blink_single = true;
                }
                sig.blink_rate = blink_rate;
            }
            self.blink_last_single_sample = Some(self.blink_onset_sample);
        }
    }

    // ── Band power computation ────────────────────────────────────────────

    fn compute_and_publish(&self) {
        if self.ring.len() < FFT_SIZE / 2 {
            return;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        // Collect ring buffer into a contiguous slice per channel
        let frames: Vec<[f32; N_CH]> = self.ring.iter().copied().collect();

        let band_power = |ch: usize, lo_hz: f32, hi_hz: f32| -> f32 {
            let signal: Vec<f32> = frames.iter().map(|f| f[ch]).collect();
            bandpower_range(&signal, &fft, lo_hz, hi_hz)
        };

        // ── Alpha ─────────────────────────────────────────────────────────
        // Use occipital channels (O1=59, POOz=60, O2=61) for alpha power
        let alpha_raw: f32 = [59usize, 60, 61]
            .iter()
            .map(|&ch| band_power(ch, 8.0, 13.0))
            .sum::<f32>()
            / 3.0;

        // ── Beta ──────────────────────────────────────────────────────────
        let beta_raw: f32 = [LEFT_FRONTAL_CHANNEL, RIGHT_FRONTAL_CHANNEL, CZ_CHANNEL]
            .iter()
            .map(|&ch| band_power(ch, 13.0, 30.0))
            .sum::<f32>()
            / 3.0;

        // ── Theta (frontal midline = FFCz = ch16) ─────────────────────────
        let theta_raw = band_power(FZ_CHANNEL, 4.0, 8.0);

        // ── Asymmetries ───────────────────────────────────────────────────
        let left_alpha  = band_power(LEFT_FRONTAL_CHANNEL,  8.0, 13.0).max(1e-10);
        let right_alpha = band_power(RIGHT_FRONTAL_CHANNEL, 8.0, 13.0).max(1e-10);
        let alpha_asym  = (right_alpha.ln() - left_alpha.ln()).clamp(-2.0, 2.0) / 2.0; // → -1..1

        let left_beta   = band_power(LEFT_FRONTAL_CHANNEL,  13.0, 30.0).max(1e-10);
        let right_beta  = band_power(RIGHT_FRONTAL_CHANNEL, 13.0, 30.0).max(1e-10);
        let beta_asym   = (right_beta.ln() - left_beta.ln()).clamp(-2.0, 2.0) / 2.0;

        // ── Normalize against baseline (if available) ─────────────────────
        // Simple: divide by sqrt(baseline_power) per key channel, then clamp 0–1.
        let alpha_norm = if let Some(ref bl) = self.baseline_alpha {
            let bl_mean = [59usize, 60, 61].iter()
                .map(|&ch| bl.get(ch).copied().unwrap_or(1.0))
                .sum::<f32>() / 3.0;
            (alpha_raw / bl_mean.max(1e-10)).clamp(0.0, 3.0) / 3.0
        } else {
            normalize_fallback(alpha_raw)
        };

        let beta_norm = if let Some(ref bl) = self.baseline_beta {
            let bl_mean = [LEFT_FRONTAL_CHANNEL, RIGHT_FRONTAL_CHANNEL, CZ_CHANNEL].iter()
                .map(|&ch| bl.get(ch).copied().unwrap_or(1.0))
                .sum::<f32>() / 3.0;
            (beta_raw / bl_mean.max(1e-10)).clamp(0.0, 3.0) / 3.0
        } else {
            normalize_fallback(beta_raw)
        };

        let theta_norm = if let Some(ref bl) = self.baseline_theta {
            let bl_fz = bl.get(FZ_CHANNEL).copied().unwrap_or(1.0);
            (theta_raw / bl_fz.max(1e-10)).clamp(0.0, 3.0) / 3.0
        } else {
            normalize_fallback(theta_raw)
        };

        // ── Engagement index β / (α + θ) ──────────────────────────────────
        let engagement = (beta_norm / (alpha_norm + theta_norm + 0.01)).clamp(0.0, 1.0);

        // ── Write to shared state ─────────────────────────────────────────
        if let Ok(mut sig) = self.signals.lock() {
            sig.alpha_power      = alpha_norm;
            sig.beta_power       = beta_norm;
            sig.theta_power      = theta_norm;
            sig.engagement       = engagement;
            sig.alpha_asymmetry  = alpha_asym;
            sig.beta_asymmetry   = beta_asym;
            sig.sample_count     = self.total_samples;
            // (event flags left as-is — they are only cleared explicitly)
        }
    }
}

// ── FFT helpers ───────────────────────────────────────────────────────────────

fn bandpower_range(
    signal: &[f32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    lo_hz: f32,
    hi_hz: f32,
) -> f32 {
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); FFT_SIZE];
    let n = signal.len().min(FFT_SIZE);
    let start = signal.len().saturating_sub(FFT_SIZE);
    for i in 0..n {
        let w = (std::f32::consts::PI * i as f32 / FFT_SIZE as f32).sin().powi(2);
        buf[i] = Complex::new(signal[start + i] * w, 0.0);
    }
    fft.process(&mut buf);

    let bin_hz = SAMPLE_RATE / FFT_SIZE as f32;
    let s = (lo_hz / bin_hz).ceil() as usize;
    let e = ((hi_hz / bin_hz).floor() as usize).min(FFT_SIZE / 2);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for b in s..=e {
        sum += buf[b].norm_sqr();
        cnt += 1;
    }
    if cnt > 0 { (sum / cnt as f32).sqrt() } else { 0.0 }
}

/// Rough log-scale normalization when no baseline is available.
fn normalize_fallback(raw: f32) -> f32 {
    // EEG band powers are log-distributed; map to 0–1 roughly.
    let log_val = (raw + 1.0).ln();
    (log_val / 6.0).clamp(0.0, 1.0) // ln(~400) ≈ 6 covers typical range
}
