use rustfft::{num_complex::Complex, FftPlanner};

use crate::recorder::epoch::StimulusEpoch;

// ── Channel groups (0-indexed) ────────────────────────────────────────────────

/// Frontal channels likely to capture blinks (Fp1/Fp2 area).
pub const BLINK_CHANNELS: [usize; 2] = [0, 1];
/// Temporal channels likely to capture jaw-clench EMG.
pub const JAW_CHANNELS: [usize; 4] = [10, 11, 22, 23];
/// Central channels (C3/C4 area) for motor imagery.
pub const MOTOR_CHANNELS: [usize; 4] = [20, 21, 30, 31];

// ── Band definitions (Hz) ─────────────────────────────────────────────────────

const BANDS: [(f32, f32); 5] = [
    (0.5, 4.0),   // delta
    (4.0, 8.0),   // theta
    (8.0, 13.0),  // alpha
    (13.0, 30.0), // beta
    (30.0, 80.0), // gamma
];

const FFT_SIZE: usize = 128;
const NUM_CHANNELS: usize = 64;

// ── Public API ────────────────────────────────────────────────────────────────

/// Extract a fixed-length feature vector from a `StimulusEpoch`.
///
/// Features (in order):
/// 1. Peak amplitude on each of BLINK_CHANNELS (2 values)
/// 2. Peak amplitude on each of JAW_CHANNELS   (4 values)
/// 3. Peak amplitude on each of MOTOR_CHANNELS (4 values)
/// 4. Band power (5 bands × 64 channels)       (320 values)
/// 5. Fp1/Fp2 amplitude asymmetry              (1 value)
/// 6. Temporal high-freq power (30-100 Hz)      (1 value)
///
/// Total: 332 dimensions.
pub fn extract_features(epoch: &StimulusEpoch) -> Vec<f32> {
    let sample_rate = epoch.sample_rate;
    let mut feat = Vec::with_capacity(332);

    // 1–3. Peak amplitudes on key channel groups
    for &ch in BLINK_CHANNELS.iter().chain(JAW_CHANNELS.iter()).chain(MOTOR_CHANNELS.iter()) {
        feat.push(peak_amplitude(&epoch.channel(ch)));
    }

    // 4. Band power per channel
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    for ch in 0..NUM_CHANNELS {
        let signal = epoch.channel(ch);
        let powers = band_powers(&signal, sample_rate, &fft);
        feat.extend_from_slice(&powers);
    }

    // 5. Fp1/Fp2 asymmetry: (|Fp1| - |Fp2|) / (|Fp1| + |Fp2| + ε)
    let fp1 = peak_amplitude(&epoch.channel(BLINK_CHANNELS[0]));
    let fp2 = peak_amplitude(&epoch.channel(BLINK_CHANNELS[1]));
    let asym = (fp1 - fp2) / (fp1 + fp2 + 1e-6);
    feat.push(asym);

    // 6. Temporal high-freq power (30–100 Hz) averaged across jaw channels
    let hf_power: f32 = JAW_CHANNELS
        .iter()
        .map(|&ch| {
            let sig = epoch.channel(ch);
            bandpower_range(&sig, sample_rate, 30.0, 100.0, &fft)
        })
        .sum::<f32>()
        / JAW_CHANNELS.len() as f32;
    feat.push(hf_power);

    // L2-normalise so cosine similarity works properly
    let norm: f32 = feat.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut feat {
            *v /= norm;
        }
    }

    feat
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Peak-to-peak amplitude of a signal slice.
fn peak_amplitude(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let min = signal.iter().copied().fold(f32::INFINITY, f32::min);
    let max = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (max - min).abs()
}

/// Compute power in each of the 5 standard EEG bands.
fn band_powers(
    signal: &[f32],
    sample_rate: f32,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) -> [f32; 5] {
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); FFT_SIZE];
    let n = signal.len().min(FFT_SIZE);
    let start = signal.len().saturating_sub(FFT_SIZE);
    for i in 0..n {
        // Hann window
        let w = (std::f32::consts::PI * i as f32 / FFT_SIZE as f32).sin().powi(2);
        buf[i] = Complex::new(signal[start + i] * w, 0.0);
    }
    fft.process(&mut buf);

    let bin_hz = sample_rate / FFT_SIZE as f32;
    let mut powers = [0.0f32; 5];
    for (band_idx, &(lo, hi)) in BANDS.iter().enumerate() {
        let start_bin = (lo / bin_hz).ceil() as usize;
        let end_bin = (hi / bin_hz).floor() as usize;
        let end_bin = end_bin.min(FFT_SIZE / 2);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for b in start_bin..=end_bin {
            sum += buf[b].norm_sqr();
            count += 1;
        }
        powers[band_idx] = if count > 0 { (sum / count as f32).sqrt() } else { 0.0 };
    }
    powers
}

/// Power in an arbitrary Hz range.
fn bandpower_range(
    signal: &[f32],
    sample_rate: f32,
    lo_hz: f32,
    hi_hz: f32,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) -> f32 {
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); FFT_SIZE];
    let n = signal.len().min(FFT_SIZE);
    let start = signal.len().saturating_sub(FFT_SIZE);
    for i in 0..n {
        let w = (std::f32::consts::PI * i as f32 / FFT_SIZE as f32).sin().powi(2);
        buf[i] = Complex::new(signal[start + i] * w, 0.0);
    }
    fft.process(&mut buf);

    let bin_hz = sample_rate / FFT_SIZE as f32;
    let start_bin = (lo_hz / bin_hz).ceil() as usize;
    let end_bin = ((hi_hz / bin_hz).floor() as usize).min(FFT_SIZE / 2);
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for b in start_bin..=end_bin {
        sum += buf[b].norm_sqr();
        count += 1;
    }
    if count > 0 { (sum / count as f32).sqrt() } else { 0.0 }
}
