//! Baseline EEG profile.
//!
//! Record 30–60 s of resting EEG to build a personalised reference.
//! The profile exposes:
//!   • Per-channel band powers  → classifier normalisation
//!   • Channel quality scores   → electrode contact heatmap
//!   • Individual Alpha Frequency (IAF) → your personal alpha peak
//!   • Frontal Alpha Asymmetry (FAA)    → approach / withdrawal index
//!   • Dominant band per brain region   → resting-state fingerprint

use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

// ── Constants ─────────────────────────────────────────────────────────────────

pub const N_CH: usize = 64;
/// FFT size for band-power estimation (300 Hz / 256 = 1.17 Hz/bin).
const FFT_SIZE: usize = 256;
/// Samples per processing window (1 second @ 300 Hz).
pub const WINDOW_SAMPLES: usize = 300;

pub const BAND_NAMES: [&str; 5] = ["δ delta", "θ theta", "α alpha", "β beta", "γ gamma"];
/// Band short symbols for compact display.
pub const BAND_SYMS: [&str; 5] = ["δ", "θ", "α", "β", "γ"];
/// Hues (0-1) for each band used in colour coding.
pub const BAND_HUES: [f32; 5] = [0.72, 0.55, 0.33, 0.1, 0.0];
const BANDS: [(f32, f32); 5] = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 80.0),
];

/// Approximate brain region groupings (channel indices, layout is an estimate).
pub const REGION_NAMES: [&str; 5] = ["Frontal", "Temporal", "Central", "Parietal", "Occipital"];
pub const REGION_CHANNELS: [&[usize]; 5] = [
    &[0, 1, 2, 3],
    &[10, 11, 22, 23],
    &[20, 21, 30, 31],
    &[40, 41, 50, 51],
    &[60, 61, 62, 63],
];

// ── Baseline profile (serialisable, persists to disk) ────────────────────────

/// Computed resting-state reference derived from a baseline recording session.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BaselineProfile {
    /// Unix timestamp (s) when the baseline was finalised.
    pub recorded_at: f64,
    /// Recording duration in seconds (number of 1-s windows processed).
    pub duration_s: f32,
    /// Mean band power per channel per band: `[channel][band]`.
    pub mean_band_powers: Vec<[f32; 5]>,
    /// Per-channel signal std-dev (noise-floor proxy, raw ADC units).
    pub noise_floor: Vec<f32>,
    /// Per-channel contact quality 0 (noisy) – 1 (clean).
    pub channel_quality: Vec<f32>,
    /// Dominant band index (0=δ … 4=γ) per channel.
    pub dominant_band: Vec<usize>,
    /// Individual Alpha Frequency in Hz — your personal alpha peak.
    pub iaf_hz: f32,
    /// Frontal Alpha Asymmetry: ln(right_alpha) − ln(left_alpha).
    /// > 0 → approach-motivated; < 0 → withdrawal-motivated.
    pub faa: f32,
    /// Average band powers per region: `[region][band]`.
    pub region_powers: Vec<[f32; 5]>,
    /// Mean amplitude spectrum per channel: `mean_spectrum[channel][bin]`.
    /// Shape [N_CH][FFT_SIZE/2] = [64][128]. Bin k ≈ k * (sample_rate / FFT_SIZE) Hz.
    #[serde(default)]
    pub mean_spectrum: Vec<Vec<f32>>,
    // ── FOOOF / specparam aperiodic fit (filled by MNE post-processing) ───────
    /// Aperiodic offset (log-power intercept). Default 0 until MNE processes.
    #[serde(default)]
    pub fooof_offset: f32,
    /// Aperiodic exponent (1/f slope). Higher = more 1/f; typical range 1–3.
    #[serde(default)]
    pub fooof_exponent: f32,
    /// R² goodness-of-fit of the aperiodic model (0–1). 0 = not yet computed.
    #[serde(default)]
    pub fooof_r2: f32,
    /// Whether this profile was computed by the full MNE pipeline (vs. Rust preview).
    #[serde(default)]
    pub mne_processed: bool,
}

impl BaselineProfile {
    /// Fraction of total power in each band, averaged across all channels.
    /// Useful for the global "brain-state" bar chart.
    pub fn global_band_ratios(&self) -> [f32; 5] {
        let mut totals = [0.0f32; 5];
        for ch in &self.mean_band_powers {
            for (i, &p) in ch.iter().enumerate() {
                totals[i] += p;
            }
        }
        let sum: f32 = totals.iter().sum();
        if sum > 1e-10 {
            for v in &mut totals {
                *v /= sum;
            }
        }
        totals
    }

    /// Fraction of total power in each band for a single region.
    pub fn region_band_ratios(&self, region: usize) -> [f32; 5] {
        if region >= self.region_powers.len() {
            return [0.0; 5];
        }
        let powers = self.region_powers[region];
        let sum: f32 = powers.iter().sum();
        let mut out = powers;
        if sum > 1e-10 {
            for v in &mut out {
                *v /= sum;
            }
        }
        out
    }

    /// Human-readable FAA interpretation.
    pub fn faa_label(&self) -> &'static str {
        if self.faa > 0.25 {
            "approach-oriented"
        } else if self.faa < -0.25 {
            "withdrawal-oriented"
        } else {
            "balanced"
        }
    }

    /// FAA as 0-1 position on a gauge (0 = max withdrawal, 0.5 = balanced, 1 = max approach).
    pub fn faa_gauge(&self) -> f32 {
        ((self.faa.clamp(-1.5, 1.5) / 1.5) * 0.5 + 0.5).clamp(0.0, 1.0)
    }

    /// IAF as 0-1 position within the 8-13 Hz range.
    pub fn iaf_gauge(&self) -> f32 {
        ((self.iaf_hz - 8.0) / 5.0).clamp(0.0, 1.0)
    }
}

// ── Normalisation helper ──────────────────────────────────────────────────────

/// Normalise a 332-dim feature vector extracted by `features::extract_features`
/// using a resting-state baseline.
///
/// Band-power features (indices 10..330, channels × bands) are scaled by
/// `1 / sqrt(baseline_power)` so that channels/bands deviating from rest
/// contribute more to classification than "background" activity.
/// The vector is then re-L2-normalised.
pub fn normalize_features(features: &[f32], baseline: &BaselineProfile) -> Vec<f32> {
    const PEAK_OFFSET: usize = 10; // peak amplitude features precede band powers
    let mut out = features.to_vec();

    for ch in 0..N_CH {
        for band in 0..5 {
            let idx = PEAK_OFFSET + ch * 5 + band;
            if idx >= out.len() {
                break;
            }
            let bl = if ch < baseline.mean_band_powers.len() {
                baseline.mean_band_powers[ch][band]
            } else {
                1.0
            };
            // Scale: divide by sqrt(baseline) so high-baseline channels are down-weighted.
            let scale = 1.0 / (bl.sqrt() + 1e-8);
            out[idx] *= scale;
        }
    }

    // Re-L2-normalise
    let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut out {
            *v /= norm;
        }
    }
    out
}

// ── Baseline recorder ─────────────────────────────────────────────────────────

/// Accumulates live EEG frames and computes running statistics.
/// Call [`push_sample`] for every incoming frame; call [`finalize`] when done.
pub struct BaselineRecorder {
    band_power_sums: Vec<[f64; 5]>,
    sample_sums: Vec<f64>,
    sample_sq_sums: Vec<f64>,
    sample_counts: Vec<u64>,
    /// Accumulated FFT amplitude spectrum per channel: `spectrum_sums[ch][bin]`.
    /// Used for PSD display and IAF detection across all channels.
    spectrum_sums: Vec<Vec<f64>>,
    spectrum_count: usize,
    /// Partial current window (fills to `WINDOW_SAMPLES` before being processed).
    pub window_buf: Vec<[f32; 64]>,
    /// Number of complete 1-second windows processed so far.
    pub windows_done: usize,
    /// Target number of windows (= recording duration in seconds).
    pub target_windows: usize,
    pub sample_rate: f32,
    /// 1st-order IIR high-pass filter state: (prev_input, prev_output) per channel.
    /// Cutoff ≈ 0.5 Hz — removes DC drift without distorting EEG bands.
    hp_state: Vec<(f32, f32)>,
    /// Number of windows skipped due to artifact rejection (for diagnostics).
    pub windows_rejected: usize,
    /// All raw (pre-filter) sample frames, kept for MNE post-processing.
    /// Shape: [n_samples][64]. Populated only during recording; cleared after export.
    raw_frames: Vec<[f32; 64]>,
}

impl BaselineRecorder {
    pub fn new(duration_s: usize, sample_rate: f32) -> Self {
        let expected_samples = duration_s * sample_rate as usize;
        Self {
            band_power_sums: vec![[0.0f64; 5]; N_CH],
            sample_sums: vec![0.0f64; N_CH],
            sample_sq_sums: vec![0.0f64; N_CH],
            sample_counts: vec![0u64; N_CH],
            spectrum_sums: vec![vec![0.0f64; FFT_SIZE / 2]; N_CH],
            spectrum_count: 0,
            window_buf: Vec::with_capacity(WINDOW_SAMPLES),
            windows_done: 0,
            target_windows: duration_s,
            sample_rate,
            hp_state: vec![(0.0, 0.0); N_CH],
            windows_rejected: 0,
            raw_frames: Vec::with_capacity(expected_samples),
        }
    }

    /// Take ownership of the raw frames for export to disk.
    /// Clears the internal buffer to free memory.
    pub fn take_raw_frames(&mut self) -> Vec<[f32; 64]> {
        std::mem::take(&mut self.raw_frames)
    }

    /// Push one 64-channel sample frame.
    /// Returns `true` if a full 1-second window was just committed.
    pub fn push_sample(&mut self, frame: &[f32; 64]) -> bool {
        self.raw_frames.push(*frame); // keep unfiltered copy for MNE
        self.window_buf.push(*frame);
        if self.window_buf.len() >= WINDOW_SAMPLES {
            let window = std::mem::take(&mut self.window_buf);
            self.window_buf = Vec::with_capacity(WINDOW_SAMPLES);
            self.process_window(&window);
            true
        } else {
            false
        }
    }

    pub fn is_complete(&self) -> bool {
        self.windows_done >= self.target_windows
    }

    /// 0.0 – 1.0 recording progress.
    pub fn progress(&self) -> f32 {
        (self.windows_done as f32 / self.target_windows as f32).min(1.0)
    }

    /// Build the final [`BaselineProfile`] from accumulated statistics.
    pub fn finalize(&self) -> Option<BaselineProfile> {
        if self.windows_done == 0 {
            return None;
        }
        let n = self.windows_done as f64;

        // Mean band powers
        let mean_band_powers: Vec<[f32; 5]> = self
            .band_power_sums
            .iter()
            .map(|sums| {
                let mut m = [0.0f32; 5];
                for (i, &s) in sums.iter().enumerate() {
                    m[i] = (s / n) as f32;
                }
                m
            })
            .collect();

        // Noise floor = std-dev per channel (Welford-style)
        let noise_floor: Vec<f32> = (0..N_CH)
            .map(|ch| {
                let cnt = self.sample_counts[ch] as f64;
                if cnt < 2.0 {
                    return 0.0;
                }
                let mean = self.sample_sums[ch] / cnt;
                let var = (self.sample_sq_sums[ch] / cnt) - mean * mean;
                var.max(0.0).sqrt() as f32
            })
            .collect();

        // Channel quality: low noise relative to the worst channel → high quality
        let max_noise = noise_floor.iter().copied().fold(0.0f32, f32::max).max(1.0);
        let channel_quality: Vec<f32> = noise_floor
            .iter()
            .map(|&n| 1.0 - (n / max_noise).min(1.0) * 0.92)
            .collect();

        // Dominant band per channel
        let dominant_band: Vec<usize> = mean_band_powers
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(2)
            })
            .collect();

        // IAF: peak alpha frequency estimated from occipital channels.
        // Cognionics HD-72 actual channel order (from LSL config):
        //   O1 = ch 59,  POOz ≈ Oz = ch 60,  O2 = ch 61
        let iaf_hz = if self.spectrum_count > 0 {
            let bin_hz = self.sample_rate / FFT_SIZE as f32;
            let lo = (8.0_f32 / bin_hz).ceil() as usize;
            let hi = ((13.0_f32 / bin_hz).floor() as usize).min(FFT_SIZE / 2 - 1);
            let occ = [59usize, 60, 61]; // O1, POOz, O2
            let mut avg = vec![0.0f64; FFT_SIZE / 2];
            for &ch in &occ {
                if ch < self.spectrum_sums.len() {
                    for (b, &v) in self.spectrum_sums[ch].iter().enumerate() {
                        if b < avg.len() { avg[b] += v; }
                    }
                }
            }
            let peak_bin = avg[lo..=hi]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i + lo)
                .unwrap_or(lo + 2);
            peak_bin as f32 * bin_hz
        } else {
            10.0
        };

        // Mean amplitude spectrum per channel (for PSD visualisation)
        let n_spec = self.spectrum_count as f64;
        let mean_spectrum: Vec<Vec<f32>> = self.spectrum_sums.iter()
            .map(|sums| {
                sums.iter()
                    .map(|&s| if n_spec > 0.0 { (s / n_spec) as f32 } else { 0.0 })
                    .collect()
            })
            .collect();

        // FAA = ln(right_alpha) − ln(left_alpha)
        // Cognionics HD-72 actual channel order (from LSL config):
        //   AFF3 = ch 6  (best 10-5 match to standard F3)
        //   AFF4 = ch 10 (best 10-5 match to standard F4)
        let la = mean_band_powers[6][2].max(1e-10);  // AFF3 ≈ F3
        let ra = mean_band_powers[10][2].max(1e-10); // AFF4 ≈ F4
        let faa = ra.ln() - la.ln();

        // Region mean band powers
        let region_powers: Vec<[f32; 5]> = REGION_CHANNELS
            .iter()
            .map(|&chs| {
                let valid: Vec<usize> = chs
                    .iter()
                    .copied()
                    .filter(|&c| c < N_CH)
                    .collect();
                if valid.is_empty() {
                    return [0.0; 5];
                }
                let mut avg = [0.0f32; 5];
                for &ch in &valid {
                    for (b, &p) in mean_band_powers[ch].iter().enumerate() {
                        avg[b] += p;
                    }
                }
                for v in &mut avg {
                    *v /= valid.len() as f32;
                }
                avg
            })
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        Some(BaselineProfile {
            recorded_at: now,
            duration_s: self.windows_done as f32,
            mean_band_powers,
            noise_floor,
            channel_quality,
            dominant_band,
            iaf_hz,
            faa,
            region_powers,
            mean_spectrum,
            fooof_offset: 0.0,
            fooof_exponent: 0.0,
            fooof_r2: 0.0,
            mne_processed: false,
        })
    }

    fn process_window(&mut self, window: &[[f32; 64]]) {
        // ── 1. High-pass filter (1st-order IIR, fc ≈ 0.5 Hz) ─────────────────
        // α = 1 / (1 + 2π·fc/fs)  ≈ 0.9895 @ 300 Hz / 0.5 Hz
        let alpha = 1.0_f32 / (1.0 + 2.0 * std::f32::consts::PI * 0.5 / self.sample_rate);
        let mut filtered = vec![[0.0f32; N_CH]; window.len()];
        for ch in 0..N_CH {
            let (mut prev_in, mut prev_out) = self.hp_state[ch];
            for (t, frame) in window.iter().enumerate() {
                let x = frame[ch];
                let y = alpha * (prev_out + x - prev_in);
                filtered[t][ch] = y;
                prev_in = x;
                prev_out = y;
            }
            self.hp_state[ch] = (prev_in, prev_out);
        }

        // ── 2. Artifact rejection — skip windows with extreme amplitudes ───────
        // Threshold: peak-to-peak > 150 µV on any channel → jaw clench / blink.
        const PTP_THRESHOLD: f32 = 150.0;
        let is_artifact = (0..N_CH).any(|ch| {
            let vals: Vec<f32> = filtered.iter().map(|f| f[ch]).collect();
            let mn = vals.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (mx - mn) > PTP_THRESHOLD
        });
        if is_artifact {
            self.windows_rejected += 1;
            self.windows_done += 1; // count toward progress but discard data
            return;
        }

        // ── 3. Accumulate statistics on the clean, filtered window ─────────────
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        for ch in 0..N_CH {
            let signal: Vec<f32> = filtered.iter().map(|f| f[ch]).collect();

            // Band powers
            let powers = compute_band_powers(&signal, self.sample_rate, &fft);
            for (i, &p) in powers.iter().enumerate() {
                self.band_power_sums[ch][i] += p as f64;
            }

            // Online variance (Welford)
            for &v in &signal {
                let v64 = v as f64;
                self.sample_sums[ch] += v64;
                self.sample_sq_sums[ch] += v64 * v64;
                self.sample_counts[ch] += 1;
            }

            // Spectrum for all channels — used for PSD display and IAF detection
            let spec = compute_spectrum(&signal, &fft);
            for (i, &v) in spec.iter().enumerate() {
                if i < self.spectrum_sums[ch].len() {
                    self.spectrum_sums[ch][i] += v as f64;
                }
            }
        }
        self.spectrum_count += 1;
        self.windows_done += 1;
    }
}

// ── FFT helpers ───────────────────────────────────────────────────────────────

fn compute_band_powers(
    signal: &[f32],
    sample_rate: f32,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) -> [f32; 5] {
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); FFT_SIZE];
    let n = signal.len().min(FFT_SIZE);
    let start = signal.len().saturating_sub(FFT_SIZE);
    for i in 0..n {
        let w = (std::f32::consts::PI * i as f32 / FFT_SIZE as f32).sin().powi(2);
        buf[i] = Complex::new(signal[start + i] * w, 0.0);
    }
    fft.process(&mut buf);
    let bin_hz = sample_rate / FFT_SIZE as f32;
    let mut out = [0.0f32; 5];
    for (idx, &(lo, hi)) in BANDS.iter().enumerate() {
        let s = (lo / bin_hz).ceil() as usize;
        let e = ((hi / bin_hz).floor() as usize).min(FFT_SIZE / 2);
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for b in s..=e {
            sum += buf[b].norm_sqr();
            cnt += 1;
        }
        out[idx] = if cnt > 0 { (sum / cnt as f32).sqrt() } else { 0.0 };
    }
    out
}

fn compute_spectrum(signal: &[f32], fft: &std::sync::Arc<dyn rustfft::Fft<f32>>) -> Vec<f32> {
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); FFT_SIZE];
    let n = signal.len().min(FFT_SIZE);
    let start = signal.len().saturating_sub(FFT_SIZE);
    for i in 0..n {
        let w = (std::f32::consts::PI * i as f32 / FFT_SIZE as f32).sin().powi(2);
        buf[i] = Complex::new(signal[start + i] * w, 0.0);
    }
    fft.process(&mut buf);
    buf[..FFT_SIZE / 2].iter().map(|c| c.norm()).collect()
}
