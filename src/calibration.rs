//! Per-user EEG calibration system.
//!
//! Multi-stage calibration flow:
//!   1. Signal quality — get the headset into a usable state (30–90 s)
//!   2. Resting baseline — eyes open then eyes closed (60–90 s each)
//!   3. Artifact gestures — blink, double-blink, jaw clench
//!   4. Expression calibration — relaxed vs focused vs effortful
//!   5. Musical sandbox — validate that the calibration produced something playable
//!
//! Returning users get a quick refresh flow instead of full recalibration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Sample rate of the Cognionics headset (and default for calibration math).
const SAMPLE_RATE: f32 = crate::cognionics::SAMPLE_RATE as f32;

// ── Calibration profile ──────────────────────────────────────────────────────

/// Per-user calibration profile.  Saved as JSON, reloadable across sessions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationProfile {
    pub user_name: String,
    pub created_at: String,
    /// How many times this profile has been used / refreshed.
    pub session_count: u32,

    // ── channel quality ──────────────────────────────────────────────
    /// Per-channel quality score (0.0–1.0).  Green > 0.6, yellow > 0.3.
    pub channel_quality: Vec<f32>,
    /// Which channels passed quality check.
    pub good_channels: Vec<usize>,
    /// Per-channel RMS noise floor (µV).
    pub noise_floor: Vec<f32>,
    /// Per-channel 50/60 Hz line-noise power (µV²).
    pub line_noise: Vec<f32>,

    // ── resting baseline ─────────────────────────────────────────────
    pub channel_means_open: Vec<f32>,
    pub channel_stds_open: Vec<f32>,
    pub channel_means_closed: Vec<f32>,
    pub channel_stds_closed: Vec<f32>,

    /// Per-band power baselines from eyes-open rest.
    pub band_power_means: BandPowers,
    pub band_power_stds: BandPowers,

    /// Individual alpha peak frequency (Hz).
    pub alpha_peak_hz: f32,
    /// Individual alpha frequency range (low, high) in Hz.
    pub alpha_range: (f32, f32),

    // ── artifact templates ───────────────────────────────────────────
    pub blink_amplitude: f32,
    pub double_blink_interval: f32,
    pub jaw_clench_amplitude: f32,

    // ── expression anchors ───────────────────────────────────────────
    /// Band powers during relaxed soft-gaze condition.
    pub expression_relaxed: BandPowers,
    /// Band powers during focused-attention condition.
    pub expression_focused: BandPowers,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BandPowers {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

impl BandPowers {
    pub fn as_array(&self) -> [f32; 5] {
        [self.delta, self.theta, self.alpha, self.beta, self.gamma]
    }
    pub fn from_array(a: [f32; 5]) -> Self {
        Self { delta: a[0], theta: a[1], alpha: a[2], beta: a[3], gamma: a[4] }
    }
}

impl CalibrationProfile {
    pub fn profiles_dir() -> PathBuf {
        let mut dir = dirs_home().unwrap_or_else(|| PathBuf::from("."));
        dir.push(".mind-daw");
        dir.push("profiles");
        dir
    }

    pub fn save(&self) -> Result<PathBuf, std::io::Error> {
        let dir = Self::profiles_dir();
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", safe_filename(&self.user_name)));
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, json)?;
        Ok(path)
    }

    pub fn load(name: &str) -> Result<Self, std::io::Error> {
        let path = Self::profiles_dir().join(format!("{}.json", safe_filename(name)));
        let json = std::fs::read_to_string(&path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn list_profiles() -> Vec<String> {
        let dir = Self::profiles_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
        entries
            .filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().to_string();
                name.strip_suffix(".json").map(|s| s.to_string())
            })
            .collect()
    }

    /// Normalize raw band powers using baseline statistics → z-scores.
    pub fn normalize_bands(&self, raw: &BandPowers) -> BandPowers {
        let r = raw.as_array();
        let m = self.band_power_means.as_array();
        let s = self.band_power_stds.as_array();
        let mut out = [0.0f32; 5];
        for i in 0..5 {
            out[i] = (r[i] - m[i]) / s[i].max(1e-6);
        }
        BandPowers::from_array(out)
    }

}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn safe_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ── Calibration steps ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationStep {
    Idle,
    /// 1. Signal quality — sit still, relax, steady gaze.  30–90 s.
    SignalQuality,
    /// 2a. Resting baseline, eyes open.  60 s.
    RestingEyesOpen,
    /// 2b. Resting baseline, eyes closed.  60 s.
    RestingEyesClosed,
    /// 3a. Intentional single blinks.
    ArtifactBlink,
    /// 3b. Intentional jaw clenches.
    ArtifactJawClench,
    /// 4a. Expression calibration — relaxed soft gaze.
    ExpressionRelaxed,
    /// 4b. Expression calibration — focused attention.
    ExpressionFocused,
    /// 5. Musical sandbox — user explores the instrument.
    Sandbox,
    /// Done.
    Complete,
}

impl CalibrationStep {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle              => "Ready to calibrate",
            Self::SignalQuality     => "1/8  Signal quality",
            Self::RestingEyesOpen   => "2/8  Resting baseline (eyes open)",
            Self::RestingEyesClosed => "3/8  Resting baseline (eyes closed)",
            Self::ArtifactBlink     => "4/8  Blink recording",
            Self::ArtifactJawClench => "5/8  Jaw clench recording",
            Self::ExpressionRelaxed => "6/8  Relaxed gaze",
            Self::ExpressionFocused => "7/8  Focused attention",
            Self::Sandbox           => "8/8  Musical sandbox",
            Self::Complete          => "Calibration complete",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Self::Idle              => "Press Start to begin calibration.",
            Self::SignalQuality     => "Sit still, relax your shoulders and jaw, keep a steady gaze.",
            Self::RestingEyesOpen   => "Keep your eyes open and relax. Breathe naturally.",
            Self::RestingEyesClosed => "Close your eyes and relax. Breathe naturally.",
            Self::ArtifactBlink     => "Blink deliberately 5 times, pausing ~1 s between each.",
            Self::ArtifactJawClench => "Clench your jaw firmly 3 times, pausing ~2 s between each.",
            Self::ExpressionRelaxed => "Relax completely. Soft gaze, calm breathing.",
            Self::ExpressionFocused => "Focus intently on the center of the screen. Count backwards from 100 by 7.",
            Self::Sandbox           => "Explore the harmonic space! Move around the orbifold with your brain.",
            Self::Complete          => "Calibration complete. Profile saved.",
        }
    }

    pub fn duration_secs(self) -> f32 {
        match self {
            Self::SignalQuality     => 30.0,
            Self::RestingEyesOpen   => 60.0,
            Self::RestingEyesClosed => 60.0,
            Self::ArtifactBlink     => 10.0,
            Self::ArtifactJawClench => 10.0,
            Self::ExpressionRelaxed => 30.0,
            Self::ExpressionFocused => 30.0,
            Self::Sandbox           => 60.0,
            _ => 0.0,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Idle              => Self::SignalQuality,
            Self::SignalQuality     => Self::RestingEyesOpen,
            Self::RestingEyesOpen   => Self::RestingEyesClosed,
            Self::RestingEyesClosed => Self::ArtifactBlink,
            Self::ArtifactBlink     => Self::ArtifactJawClench,
            Self::ArtifactJawClench => Self::ExpressionRelaxed,
            Self::ExpressionRelaxed => Self::ExpressionFocused,
            Self::ExpressionFocused => Self::Sandbox,
            Self::Sandbox           => Self::Complete,
            Self::Complete          => Self::Complete,
        }
    }
}

// ── Quick-refresh steps (for returning users) ────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RefreshStep {
    Idle,
    SignalQuality,  // 15 s
    QuickBaseline,  // 30 s eyes open
    Sandbox,        // 30 s
    Complete,
}

impl RefreshStep {
    pub fn duration_secs(self) -> f32 {
        match self {
            Self::SignalQuality => 15.0,
            Self::QuickBaseline => 30.0,
            Self::Sandbox       => 30.0,
            _ => 0.0,
        }
    }
    pub fn next(self) -> Self {
        match self {
            Self::Idle           => Self::SignalQuality,
            Self::SignalQuality  => Self::QuickBaseline,
            Self::QuickBaseline  => Self::Sandbox,
            Self::Sandbox        => Self::Complete,
            Self::Complete       => Self::Complete,
        }
    }
}

// ── Per-channel quality diagnostics ──────────────────────────────────────────

/// Detailed per-channel quality info for the UI.
#[derive(Clone, Debug, Default)]
pub struct ChannelDiag {
    /// Overall quality 0.0–1.0.
    pub quality: f32,
    /// True if channel appears flat / disconnected.
    pub flat: bool,
}

// ── Calibration state ────────────────────────────────────────────────────────

pub struct CalibrationState {
    pub step: CalibrationStep,
    pub user_name: String,
    pub elapsed_secs: f32,
    pub profile: Option<CalibrationProfile>,
    pub is_refresh: bool,
    pub refresh_step: RefreshStep,

    num_channels: usize,

    /// Raw sample ring-buffer per channel.  Only NEW samples go here.
    /// Max capacity: longest step (60 s) × sample_rate ≈ 18 000.
    sample_buf: Vec<Vec<f32>>,
    /// Counter for deduplicating `feed_buffer` calls.
    last_buf_hash: u64,

    // Intermediate results carried between steps
    means_open: Vec<f32>,
    stds_open: Vec<f32>,
    means_closed: Vec<f32>,
    stds_closed: Vec<f32>,
    band_means: BandPowers,
    band_stds: BandPowers,
    alpha_peak: f32,
    alpha_range: (f32, f32),
    noise_floor: Vec<f32>,
    line_noise: Vec<f32>,
    good_channels: Vec<usize>,
    blink_amplitude: f32,
    jaw_clench_amplitude: f32,
    expr_relaxed: BandPowers,
    expr_focused: BandPowers,

    /// Per-channel diagnostics, updated every signal-quality frame.
    pub channel_diag: Vec<ChannelDiag>,

    /// Region warnings produced during signal quality.
    pub warnings: Vec<String>,

    pub available_profiles: Vec<String>,
}

impl CalibrationState {
    pub fn new(num_channels: usize) -> Self {
        Self {
            step: CalibrationStep::Idle,
            user_name: String::new(),
            elapsed_secs: 0.0,
            profile: None,
            is_refresh: false,
            refresh_step: RefreshStep::Idle,
            num_channels,
            sample_buf: vec![Vec::new(); num_channels],
            last_buf_hash: 0,
            means_open: vec![0.0; num_channels],
            stds_open: vec![0.0; num_channels],
            means_closed: vec![0.0; num_channels],
            stds_closed: vec![0.0; num_channels],
            band_means: BandPowers::default(),
            band_stds: BandPowers::default(),
            alpha_peak: 10.0,
            alpha_range: (8.0, 13.0),
            noise_floor: vec![0.0; num_channels],
            line_noise: vec![0.0; num_channels],
            good_channels: Vec::new(),
            blink_amplitude: 150.0,
            jaw_clench_amplitude: 100.0,
            expr_relaxed: BandPowers::default(),
            expr_focused: BandPowers::default(),
            channel_diag: vec![ChannelDiag::default(); num_channels],
            warnings: Vec::new(),
            available_profiles: CalibrationProfile::list_profiles(),
        }
    }

    // ── public API ───────────────────────────────────────────────────

    pub fn start(&mut self) {
        self.step = CalibrationStep::SignalQuality;
        self.is_refresh = false;
        self.elapsed_secs = 0.0;
        self.warnings.clear();
        self.clear_samples();
    }

    pub fn start_refresh(&mut self) {
        self.is_refresh = true;
        self.refresh_step = RefreshStep::SignalQuality;
        self.step = CalibrationStep::SignalQuality; // reuse UI display
        self.elapsed_secs = 0.0;
        self.warnings.clear();
        self.clear_samples();
    }

    /// Feed the RAW Cognionics ring-buffers (not outlier-clipped).
    /// `raw_bufs[ch]` is the VecDeque<f32> for channel `ch`.
    /// We extract only genuinely new samples since the last call.
    pub fn feed_raw_bufs(&mut self, raw_bufs: &[std::collections::VecDeque<f32>], dt: f32) {
        if !self.is_active() {
            return;
        }
        self.elapsed_secs += dt;

        // Deduplicate: hash the last sample of each of the first 4 channels.
        let hash = quick_hash(raw_bufs);
        if hash == self.last_buf_hash {
            // No new data this frame — still advance time but don't re-add.
            self.check_step_done();
            return;
        }
        self.last_buf_hash = hash;

        // Append the tail of each raw buffer (last ~10 samples = new data at 300 Hz / 30 fps).
        let new_per_frame = (SAMPLE_RATE / 30.0).ceil() as usize; // ~10 samples
        let n_ch = raw_bufs.len().min(self.num_channels);
        for ch in 0..n_ch {
            let buf = &raw_bufs[ch];
            let take = buf.len().min(new_per_frame);
            let start = buf.len().saturating_sub(take);
            for i in start..buf.len() {
                self.sample_buf[ch].push(buf[i]);
            }
        }

        // Live quality update during signal-quality step
        if self.current_step() == CalibrationStep::SignalQuality
            && self.sample_buf[0].len() >= 300
        {
            self.update_channel_diag();
        }

        self.check_step_done();
    }

    /// Feed from LSL waveform data (Vec<Vec<f32>>).
    pub fn feed_lsl_bufs(&mut self, waveform_data: &[Vec<f32>], dt: f32) {
        if !self.is_active() {
            return;
        }
        self.elapsed_secs += dt;

        let hash = waveform_data.iter().take(4)
            .map(|d| d.last().copied().unwrap_or(0.0).to_bits() as u64)
            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b));
        if hash == self.last_buf_hash {
            self.check_step_done();
            return;
        }
        self.last_buf_hash = hash;

        let new_per_frame = 10usize;
        let n_ch = waveform_data.len().min(self.num_channels);
        for ch in 0..n_ch {
            let d = &waveform_data[ch];
            let take = d.len().min(new_per_frame);
            let start = d.len().saturating_sub(take);
            for &v in &d[start..] {
                self.sample_buf[ch].push(v);
            }
        }

        if self.current_step() == CalibrationStep::SignalQuality
            && self.sample_buf[0].len() >= 300
        {
            self.update_channel_diag();
        }

        self.check_step_done();
    }

    pub fn progress(&self) -> f32 {
        let dur = self.current_duration();
        if dur <= 0.0 { return 0.0; }
        (self.elapsed_secs / dur).min(1.0)
    }

    pub fn load_profile(&mut self, name: &str) -> bool {
        match CalibrationProfile::load(name) {
            Ok(p) => {
                self.user_name = p.user_name.clone();
                self.channel_diag = p.channel_quality.iter()
                    .map(|&q| ChannelDiag { quality: q, ..Default::default() })
                    .collect();
                self.good_channels = p.good_channels.clone();
                self.profile = Some(p);
                self.step = CalibrationStep::Complete;
                true
            }
            Err(e) => {
                eprintln!("Failed to load profile '{}': {}", name, e);
                false
            }
        }
    }

    pub fn good_channel_count(&self) -> usize {
        self.channel_diag.iter().filter(|d| d.quality > 0.3).count()
    }

    // ── internals ────────────────────────────────────────────────────

    fn is_active(&self) -> bool {
        self.step != CalibrationStep::Idle && self.step != CalibrationStep::Complete
    }

    fn current_step(&self) -> CalibrationStep {
        self.step
    }

    fn current_duration(&self) -> f32 {
        if self.is_refresh {
            self.refresh_step.duration_secs()
        } else {
            self.step.duration_secs()
        }
    }

    fn check_step_done(&mut self) {
        if self.elapsed_secs >= self.current_duration() && self.current_duration() > 0.0 {
            self.finalize_step();
        }
    }

    fn clear_samples(&mut self) {
        for buf in &mut self.sample_buf {
            buf.clear();
        }
        self.last_buf_hash = 0;
    }

    fn finalize_step(&mut self) {
        if self.is_refresh {
            match self.refresh_step {
                RefreshStep::SignalQuality => {
                    self.compute_signal_quality();
                }
                RefreshStep::QuickBaseline => {
                    self.compute_resting_baseline(true);
                    // Merge into existing profile
                    if let Some(ref mut p) = self.profile {
                        p.channel_quality = self.channel_diag.iter().map(|d| d.quality).collect();
                        p.good_channels = self.good_channels.clone();
                        p.noise_floor = self.noise_floor.clone();
                        p.band_power_means = self.band_means.clone();
                        p.band_power_stds = self.band_stds.clone();
                        p.session_count += 1;
                        let _ = p.save();
                    }
                }
                RefreshStep::Sandbox => {
                    // Sandbox just runs — nothing to compute
                }
                _ => {}
            }
            self.refresh_step = self.refresh_step.next();
            // Map refresh steps to display steps
            self.step = match self.refresh_step {
                RefreshStep::SignalQuality => CalibrationStep::SignalQuality,
                RefreshStep::QuickBaseline => CalibrationStep::RestingEyesOpen,
                RefreshStep::Sandbox       => CalibrationStep::Sandbox,
                RefreshStep::Complete       => CalibrationStep::Complete,
                RefreshStep::Idle           => CalibrationStep::Idle,
            };
        } else {
            match self.step {
                CalibrationStep::SignalQuality     => self.compute_signal_quality(),
                CalibrationStep::RestingEyesOpen   => self.compute_resting_baseline(true),
                CalibrationStep::RestingEyesClosed => self.compute_resting_baseline(false),
                CalibrationStep::ArtifactBlink     => self.compute_blink_template(),
                CalibrationStep::ArtifactJawClench => self.compute_jaw_clench_template(),
                CalibrationStep::ExpressionRelaxed => self.compute_expression(true),
                CalibrationStep::ExpressionFocused => {
                    self.compute_expression(false);
                    self.build_and_save_profile();
                }
                CalibrationStep::Sandbox => { /* just runs */ }
                _ => {}
            }
            self.step = self.step.next();
        }
        self.elapsed_secs = 0.0;
        self.clear_samples();
    }

    // ── signal quality ───────────────────────────────────────────────

    fn update_channel_diag(&mut self) {
        self.compute_signal_quality_inner(false);
    }

    fn compute_signal_quality(&mut self) {
        self.compute_signal_quality_inner(true);
    }

    fn compute_signal_quality_inner(&mut self, finalize: bool) {
        self.good_channels.clear();
        self.warnings.clear();

        let mut any_occipital_bad = false;
        let mut any_central_bad = false;

        for ch in 0..self.num_channels {
            let data = &self.sample_buf[ch];
            if data.len() < 30 {
                self.channel_diag[ch] = ChannelDiag { quality: 0.0, flat: true, ..Default::default() };
                continue;
            }

            let n = data.len() as f32;

            // ── basic stats ──
            let mean = data.iter().sum::<f32>() / n;
            let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let rms = variance.sqrt();

            // ── flat-line detection ──
            // A disconnected channel has near-zero variance or constant DC.
            let flat = rms < 0.5; // < 0.5 µV RMS ≈ flat

            // ── clipping detection ──
            // Check if many samples are at the extremes of the observed range.
            let mut sorted = data.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = sorted[0];
            let hi = sorted[sorted.len() - 1];
            let range = hi - lo;
            let clipped = if range > 1.0 {
                let at_lo = data.iter().filter(|&&x| (x - lo).abs() < range * 0.001).count();
                let at_hi = data.iter().filter(|&&x| (x - hi).abs() < range * 0.001).count();
                (at_lo + at_hi) as f32 / n > 0.05 // > 5% of samples at rails
            } else {
                false
            };

            // ── drift detection ──
            // Large low-frequency drift: compare mean of first quarter vs last quarter.
            let q1_end = data.len() / 4;
            let q4_start = data.len() * 3 / 4;
            let mean_q1 = if q1_end > 0 {
                data[..q1_end].iter().sum::<f32>() / q1_end as f32
            } else { mean };
            let mean_q4 = if q4_start < data.len() {
                data[q4_start..].iter().sum::<f32>() / (data.len() - q4_start) as f32
            } else { mean };
            let drift = (mean_q4 - mean_q1).abs();
            let drifting = drift > rms * 3.0 && drift > 50.0; // drift > 3× RMS and > 50 µV

            // ── line noise detection ──
            // Estimate power at 50 and 60 Hz relative to broadband.
            let line_noise_ratio = estimate_line_noise(data, SAMPLE_RATE);

            // ── composite quality score ──
            let mut quality = 1.0f32;
            if flat {
                quality = 0.0;
            } else {
                // Penalize very high RMS (likely motion artifact or bad contact)
                // Typical clean EEG: 5–50 µV RMS.  Noisy but usable: 50–200. Bad: > 200.
                if rms > 200.0 {
                    quality -= 0.6;
                } else if rms > 100.0 {
                    quality -= 0.3;
                } else if rms > 50.0 {
                    quality -= 0.1;
                }
                // Very low RMS (< 2 µV) might be a nearly-dead channel
                if rms < 2.0 {
                    quality -= 0.4;
                }
                if clipped { quality -= 0.4; }
                if drifting { quality -= 0.2; }
                if line_noise_ratio > 0.5 { quality -= 0.3; }
                else if line_noise_ratio > 0.2 { quality -= 0.1; }
                quality = quality.clamp(0.0, 1.0);
            }

            self.channel_diag[ch] = ChannelDiag {
                quality,
                flat,
            };

            self.noise_floor[ch] = rms;
            self.line_noise[ch] = line_noise_ratio;

            if quality > 0.3 {
                self.good_channels.push(ch);
            }

            // Region warnings (Cognionics HD-72 rough layout)
            if finalize {
                if ch >= 24 && ch < 32 && quality < 0.3 {
                    any_occipital_bad = true;
                }
                if ch >= 8 && ch < 16 && quality < 0.3 {
                    any_central_bad = true;
                }
            }
        }

        if finalize {
            if any_occipital_bad {
                self.warnings.push(
                    "Occipital channels are weak — visual control modes (SSVEP) may be degraded.".into()
                );
            }
            if any_central_bad {
                self.warnings.push(
                    "Central channels are weak — motor-imagery mode may be degraded.".into()
                );
            }
            if self.good_channels.is_empty() {
                self.warnings.push(
                    "No channels passed quality check. Check headset contact and try again.".into()
                );
            }
        }
    }

    // ── resting baseline ─────────────────────────────────────────────

    fn compute_resting_baseline(&mut self, eyes_open: bool) {
        let mut means = vec![0.0f32; self.num_channels];
        let mut stds = vec![0.0f32; self.num_channels];

        for ch in 0..self.num_channels {
            let data = &self.sample_buf[ch];
            if data.is_empty() { continue; }
            let n = data.len() as f32;
            let mean = data.iter().sum::<f32>() / n;
            let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            means[ch] = mean;
            stds[ch] = var.sqrt();
        }

        let band_powers = self.compute_band_powers();

        if eyes_open {
            self.means_open = means;
            self.stds_open = stds;
            self.band_means = band_powers.0;
            self.band_stds = band_powers.1;
            self.alpha_peak = self.find_alpha_peak();
            // Estimate individual alpha range: peak ± 2 Hz, clamped to 6–15 Hz.
            self.alpha_range = (
                (self.alpha_peak - 2.0).max(6.0),
                (self.alpha_peak + 2.0).min(15.0),
            );
        } else {
            self.means_closed = means;
            self.stds_closed = stds;
        }
    }

    // ── band power computation ───────────────────────────────────────

    fn compute_band_powers(&self) -> (BandPowers, BandPowers) {
        let fft_size = 256usize;
        let bin_hz = SAMPLE_RATE / fft_size as f32;

        let band_ranges: [(f32, f32); 5] = [
            (0.5, 4.0),   // delta
            (4.0, 8.0),   // theta
            (8.0, 13.0),  // alpha
            (13.0, 30.0), // beta
            (30.0, 80.0), // gamma
        ];
        let bin_ranges: Vec<(usize, usize)> = band_ranges.iter()
            .map(|&(lo, hi)| (
                (lo / bin_hz).ceil() as usize,
                (hi / bin_hz).floor() as usize,
            ))
            .collect();

        let channels_to_use: Vec<usize> = if self.good_channels.is_empty() {
            (0..self.num_channels.min(8)).collect()
        } else {
            self.good_channels.iter().copied().take(16).collect()
        };

        let mut all_bp: Vec<[f32; 5]> = Vec::new();

        for &ch in &channels_to_use {
            let data = &self.sample_buf[ch];
            if data.len() < fft_size { continue; }

            let hop = fft_size / 2;
            let n_windows = (data.len() - fft_size) / hop + 1;
            let mut band_sum = [0.0f32; 5];

            for w in 0..n_windows {
                let start = w * hop;
                let window = &data[start..start + fft_size];
                let power = windowed_power_spectrum(window, fft_size);

                for (b, &(lo, hi)) in bin_ranges.iter().enumerate() {
                    let hi = hi.min(power.len());
                    if lo < hi {
                        band_sum[b] += power[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
                    }
                }
            }

            if n_windows > 0 {
                for v in &mut band_sum { *v /= n_windows as f32; }
                all_bp.push(band_sum);
            }
        }

        if all_bp.is_empty() {
            return (BandPowers::default(), BandPowers::default());
        }

        let n = all_bp.len() as f32;
        let mut mean = [0.0f32; 5];
        for bp in &all_bp { for i in 0..5 { mean[i] += bp[i]; } }
        for v in &mut mean { *v /= n; }

        let mut std_a = [0.0f32; 5];
        for bp in &all_bp { for i in 0..5 { std_a[i] += (bp[i] - mean[i]).powi(2); } }
        for v in &mut std_a { *v = (*v / n).sqrt().max(1e-6); }

        (BandPowers::from_array(mean), BandPowers::from_array(std_a))
    }

    fn find_alpha_peak(&self) -> f32 {
        let fft_size = 256usize;
        let bin_hz = SAMPLE_RATE / fft_size as f32;
        let alpha_lo = (7.0 / bin_hz).ceil() as usize;
        let alpha_hi = (14.0 / bin_hz).floor() as usize;

        // Average power spectrum across good posterior channels
        let posterior: Vec<usize> = if self.good_channels.is_empty() {
            (0..self.num_channels.min(8)).collect()
        } else {
            // Prefer posterior channels (roughly ch 24-40 on HD-72)
            let post: Vec<usize> = self.good_channels.iter()
                .copied().filter(|&c| c >= 20).take(8).collect();
            if post.is_empty() { self.good_channels.iter().copied().take(8).collect() }
            else { post }
        };

        let mut avg_power = vec![0.0f32; fft_size / 2];
        let mut count = 0;

        for &ch in &posterior {
            let data = &self.sample_buf[ch];
            if data.len() < fft_size { continue; }
            let window = &data[data.len() - fft_size..];
            let power = windowed_power_spectrum(window, fft_size);
            for (i, &p) in power.iter().enumerate() {
                avg_power[i] += p;
            }
            count += 1;
        }

        if count == 0 { return 10.0; }
        for v in &mut avg_power { *v /= count as f32; }

        let mut max_power = 0.0f32;
        let mut max_bin = alpha_lo;
        for k in alpha_lo..=alpha_hi.min(avg_power.len() - 1) {
            if avg_power[k] > max_power {
                max_power = avg_power[k];
                max_bin = k;
            }
        }

        max_bin as f32 * bin_hz
    }

    // ── artifact templates ───────────────────────────────────────────

    fn compute_blink_template(&mut self) {
        let frontal: Vec<usize> = if self.good_channels.is_empty() {
            (0..8.min(self.num_channels)).collect()
        } else {
            self.good_channels.iter().copied().filter(|&c| c < 16).collect()
        };

        // Find the top-5 peak amplitudes (relative to local mean) across frontal channels.
        let mut peaks: Vec<f32> = Vec::new();
        for &ch in &frontal {
            let data = &self.sample_buf[ch];
            if data.len() < 30 { continue; }

            // Compute running mean with a 1-second window
            let win = (SAMPLE_RATE as usize).min(data.len());
            for i in win..data.len() {
                let local_mean: f32 = data[i - win..i].iter().sum::<f32>() / win as f32;
                let amp = (data[i] - local_mean).abs();
                peaks.push(amp);
            }
        }

        peaks.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Blink threshold = median of top-10 peaks × 0.5
        // (top peaks are the actual blinks; threshold at half their amplitude)
        let top_n = 10.min(peaks.len());
        if top_n > 0 {
            let median_peak = peaks[top_n / 2];
            self.blink_amplitude = (median_peak * 0.5).max(20.0);
        }
    }

    fn compute_jaw_clench_template(&mut self) {
        // Jaw clenches produce high-frequency EMG bursts.
        // Detect as peaks in the sample-to-sample difference (derivative).
        let mut diff_peaks: Vec<f32> = Vec::new();
        for ch in 0..self.num_channels.min(16) {
            let data = &self.sample_buf[ch];
            if data.len() < 30 { continue; }
            for i in 1..data.len() {
                diff_peaks.push((data[i] - data[i - 1]).abs());
            }
        }

        diff_peaks.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let top_n = 20.min(diff_peaks.len());
        if top_n > 0 {
            let median_peak = diff_peaks[top_n / 2];
            self.jaw_clench_amplitude = (median_peak * 0.4).max(15.0);
        }
    }

    // ── expression calibration ───────────────────────────────────────

    fn compute_expression(&mut self, is_relaxed: bool) {
        let bp = self.compute_band_powers();
        if is_relaxed {
            self.expr_relaxed = bp.0;
        } else {
            self.expr_focused = bp.0;
        }
    }

    // ── build profile ────────────────────────────────────────────────

    fn build_and_save_profile(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let profile = CalibrationProfile {
            user_name: if self.user_name.is_empty() { "default".into() } else { self.user_name.clone() },
            created_at: format!("{}", now),
            session_count: 1,
            channel_quality: self.channel_diag.iter().map(|d| d.quality).collect(),
            good_channels: self.good_channels.clone(),
            noise_floor: self.noise_floor.clone(),
            line_noise: self.line_noise.clone(),
            channel_means_open: self.means_open.clone(),
            channel_stds_open: self.stds_open.clone(),
            channel_means_closed: self.means_closed.clone(),
            channel_stds_closed: self.stds_closed.clone(),
            band_power_means: self.band_means.clone(),
            band_power_stds: self.band_stds.clone(),
            alpha_peak_hz: self.alpha_peak,
            alpha_range: self.alpha_range,
            blink_amplitude: self.blink_amplitude,
            double_blink_interval: 0.3,
            jaw_clench_amplitude: self.jaw_clench_amplitude,
            expression_relaxed: self.expr_relaxed.clone(),
            expression_focused: self.expr_focused.clone(),
        };

        match profile.save() {
            Ok(path) => eprintln!("Calibration profile saved to {:?}", path),
            Err(e) => eprintln!("Failed to save calibration profile: {}", e),
        }

        self.profile = Some(profile);
        self.available_profiles = CalibrationProfile::list_profiles();
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Quick non-cryptographic hash of the last sample of the first 4 channels.
fn quick_hash(bufs: &[std::collections::VecDeque<f32>]) -> u64 {
    bufs.iter().take(4)
        .map(|b| b.back().copied().unwrap_or(0.0).to_bits() as u64)
        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b))
}

/// Hann-windowed power spectrum (bins 1..N/2).
fn windowed_power_spectrum(samples: &[f32], fft_size: usize) -> Vec<f32> {
    let pi2 = 2.0 * std::f32::consts::PI;
    let n = fft_size as f32;
    let mut power = vec![0.0f32; fft_size / 2];

    for k in 1..fft_size / 2 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, &x) in samples.iter().enumerate().take(fft_size) {
            let hann = 0.5 * (1.0 - (pi2 * i as f32 / n).cos());
            let angle = pi2 * k as f32 * i as f32 / n;
            re += x * hann * angle.cos();
            im -= x * hann * angle.sin();
        }
        power[k] = (re * re + im * im) / (n * n);
    }
    power
}

/// Estimate line noise as ratio of 50/60 Hz power to total broadband power.
fn estimate_line_noise(data: &[f32], sample_rate: f32) -> f32 {
    let fft_size = 256.min(data.len());
    if fft_size < 128 { return 0.0; }

    let window = &data[data.len() - fft_size..];
    let bin_hz = sample_rate / fft_size as f32;
    let pi2 = 2.0 * std::f32::consts::PI;
    let n = fft_size as f32;

    // Compute power at 50 Hz and 60 Hz (± 1 bin)
    let mut line_power = 0.0f32;
    let mut total_power = 0.0f32;

    for k in 1..fft_size / 2 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, &x) in window.iter().enumerate().take(fft_size) {
            let hann = 0.5 * (1.0 - (pi2 * i as f32 / n).cos());
            let angle = pi2 * k as f32 * i as f32 / n;
            re += x * hann * angle.cos();
            im -= x * hann * angle.sin();
        }
        let p = (re * re + im * im) / (n * n);
        total_power += p;

        let freq = k as f32 * bin_hz;
        if (freq - 50.0).abs() < bin_hz * 1.5 || (freq - 60.0).abs() < bin_hz * 1.5 {
            line_power += p;
        }
    }

    if total_power < 1e-12 { return 0.0; }
    (line_power / total_power).clamp(0.0, 1.0)
}
