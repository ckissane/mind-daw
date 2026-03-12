use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Channels used for blink detection (frontal).
const FP1: usize = 0;
const FP2: usize = 1;
/// Channels used for jaw-clench detection (temporal).
const JAW_CHS: [usize; 4] = [10, 11, 22, 23];

/// Sample rate assumed for all calculations.
const SAMPLE_RATE: f32 = 300.0;

/// User-adjustable detection thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoDetectThresholds {
    /// Peak-to-peak amplitude on Fp1 or Fp2 to trigger blink detection (µV).
    pub blink_uv: f32,
    /// Mean high-freq (30–100 Hz) power on temporal channels to trigger jaw-clench.
    pub jaw_power: f32,
    /// Global z-score threshold for generic large events.
    pub zscore: f32,
}

impl Default for AutoDetectThresholds {
    fn default() -> Self {
        Self { blink_uv: 150.0, jaw_power: 50.0, zscore: 3.5 }
    }
}

/// The type of event that was auto-detected.
#[derive(Clone, Debug)]
pub enum EventTrigger {
    PossibleBlink,
    PossibleJawClench,
    GenericEvent,
}

/// Inspect the ring buffer for significant events.
///
/// Looks at the last 100 ms (30 samples) of data.
/// Returns `Some(EventTrigger)` if a threshold is crossed, else `None`.
pub fn detect_event(
    ring: &VecDeque<[f32; 64]>,
    thresholds: &AutoDetectThresholds,
) -> Option<EventTrigger> {
    let window_samples = (SAMPLE_RATE * 0.1) as usize; // 100 ms
    let n = ring.len();
    if n < window_samples {
        return None;
    }

    let recent: Vec<&[f32; 64]> = ring.iter().rev().take(window_samples).collect();

    // ── Blink: large spike on Fp1 or Fp2 ─────────────────────────────────────
    for &ch in &[FP1, FP2] {
        let p2p = peak_to_peak(&recent, ch);
        if p2p > thresholds.blink_uv {
            return Some(EventTrigger::PossibleBlink);
        }
    }

    // ── Jaw clench: high-frequency burst on temporal channels ─────────────────
    let jaw_hf: f32 = JAW_CHS
        .iter()
        .map(|&ch| high_freq_power(&recent, ch))
        .sum::<f32>()
        / JAW_CHS.len() as f32;
    if jaw_hf > thresholds.jaw_power {
        return Some(EventTrigger::PossibleJawClench);
    }

    // ── Generic: any channel exceeds z-score threshold ────────────────────────
    // Use a longer baseline window: last 500 ms
    let baseline_samples = (SAMPLE_RATE * 0.5) as usize;
    let baseline: Vec<&[f32; 64]> = ring.iter().rev().take(baseline_samples).collect();
    if global_zscore(&baseline, &recent) > thresholds.zscore {
        return Some(EventTrigger::GenericEvent);
    }

    None
}

// ── Signal helpers ────────────────────────────────────────────────────────────

fn peak_to_peak(window: &[&[f32; 64]], ch: usize) -> f32 {
    let vals: Vec<f32> = window.iter().map(|s| s[ch]).collect();
    let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max - min
}

/// Approximate high-freq (30–100 Hz) power as variance of the differenced signal.
/// Cheap proxy — no FFT needed here since we only need a threshold trigger.
fn high_freq_power(window: &[&[f32; 64]], ch: usize) -> f32 {
    let vals: Vec<f32> = window.iter().map(|s| s[ch]).collect();
    if vals.len() < 2 {
        return 0.0;
    }
    let diffs: Vec<f32> = vals.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
    var.sqrt()
}

/// Global z-score: max amplitude deviation across all channels vs baseline.
fn global_zscore(baseline: &[&[f32; 64]], recent: &[&[f32; 64]]) -> f32 {
    let all_vals: Vec<f32> = baseline.iter().flat_map(|s| s.iter().copied()).collect();
    if all_vals.is_empty() {
        return 0.0;
    }
    let mean = all_vals.iter().sum::<f32>() / all_vals.len() as f32;
    let std = {
        let var = all_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / all_vals.len() as f32;
        var.sqrt().max(1e-6)
    };

    // Maximum z-score over recent window
    recent
        .iter()
        .flat_map(|s| s.iter())
        .map(|&v| ((v - mean) / std).abs())
        .fold(0.0f32, f32::max)
}
