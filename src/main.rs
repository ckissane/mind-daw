mod audio;
mod calibration;
mod cognionics;
mod control;
#[allow(dead_code)]
mod sc;
mod session_log;
mod recorder;
mod soundboard;
mod streams;
mod tonnetz;
mod word_read;

use audio::{AudioCommand, AudioHandle, EegFrame};
use calibration::CalibrationState;
use cognionics::{CogCommand, CogHandle, CogState};
use control::{ControlDecoder, ControlState};
use recorder::auto_detect::detect_event;
use recorder::baseline::{BaselineProfile, BaselineRecorder, BAND_HUES, BAND_NAMES, BAND_SYMS, REGION_NAMES};
use recorder::baseline::normalize_features as baseline_normalize;
use recorder::classifier::{predict_features, TrainedClassifier, MIN_EPOCHS_PER_CLASS, RETRAIN_EVERY};
use recorder::fbcsp::{FbcspModel, FbcspTrial};
use recorder::features::extract_features;
use recorder::{AutoDetectThresholds, ClassifierPrediction, RecordingSession, StimulusEpoch};
use word_read::WordReadState;
use gpui::*;
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::{Input, InputState};
use gpui_component::{ActiveTheme, Disableable, Root, Sizable};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use streams::{PairedStream, StreamMeta};

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Waves,
    Spectrum,
    Pca,
    Words,
    Soundboard,
    Tonnetz,
    Calibration,
}

// ── Recorder UI state ─────────────────────────────────────────────────────────

const REC_RING_CAPACITY: usize = 600; // 2 seconds at 300 Hz

const BUILT_IN_STIMULI: &[&str] = &[
    "blink_left",
    "blink_right",
    "blink_both",
    "jaw_clench",
    "breath_hold",
    "motor_left_hand",
    "motor_right_hand",
    "eyes_open",
    "eyes_closed",
    "relax",
    "sine_wave",
    "saw_wave",
    "triangle_wave",
    "square_wave",
];

/// Colours per stimulus type (hue 0–1) for visual distinction.
fn stimulus_hue(label: &str) -> f32 {
    match label {
        "blink_left" | "blink_right" | "blink_both" => 0.58,
        "jaw_clench" => 0.0,
        "breath_hold" => 0.75,
        "motor_left_hand" | "motor_right_hand" => 0.33,
        "eyes_open" | "eyes_closed" | "relax" => 0.15,
        l if l.ends_with("_wave") => 0.08,
        _ => 0.5,
    }
}

#[derive(Clone, PartialEq, Debug)]
enum RecorderMode {
    Idle,
    Armed,
    Predicting,
}

struct RecorderUiState {
    session: RecordingSession,
    active_stimulus: String,
    custom_stimuli: Vec<String>,
    mode: RecorderMode,
    pending_epoch: Option<StimulusEpoch>,
    /// Epoch loaded from the library for review (cleared when Record/ARM pressed or "Live" clicked).
    review_epoch: Option<StimulusEpoch>,
    classifier: Option<TrainedClassifier>,
    fbcsp_model: Option<FbcspModel>,
    last_prediction: Option<ClassifierPrediction>,
    prediction_history: VecDeque<ClassifierPrediction>,
    thresholds: AutoDetectThresholds,
    epochs_since_retrain: usize,
    // ── Baseline ─────────────────────────────────────────────────────────────
    /// Finalised resting-state profile (persists across recordings in a session).
    baseline: Option<BaselineProfile>,
    /// Active baseline accumulator (Some while recording, None otherwise).
    baseline_rec: Option<BaselineRecorder>,
    /// Whether the baseline dashboard is expanded below the status strip.
    baseline_dashboard_open: bool,
    /// When true, features are normalised by baseline before classification.
    normalize_with_baseline: bool,
    /// Which band (0=δ … 4=γ) the topographic map is currently showing.
    baseline_selected_band: usize,
    /// Status message from the MNE post-processing subprocess.
    /// None = idle,  Some("Processing…") = running,  Some("✓ …") = done / error.
    baseline_mne_status: Option<String>,
}

impl Default for RecorderUiState {
    fn default() -> Self {
        Self {
            session: RecordingSession::new("Cognionics HD-72".to_string()),
            active_stimulus: BUILT_IN_STIMULI[0].to_string(),
            custom_stimuli: Vec::new(),
            mode: RecorderMode::Idle,
            pending_epoch: None,
            review_epoch: None,
            classifier: None,
            fbcsp_model: None,
            last_prediction: None,
            prediction_history: VecDeque::with_capacity(10),
            thresholds: AutoDetectThresholds::default(),
            epochs_since_retrain: 0,
            baseline: None,
            baseline_rec: None,
            baseline_dashboard_open: false,
            normalize_with_baseline: true,
            baseline_selected_band: 2, // alpha — most commonly viewed band
            baseline_mne_status: None,
        }
    }
}

struct SoundboardUiState {
    waveform: soundboard::SbWaveform,
    instrument: soundboard::SbInstrument,
    root_midi: u8,
    chord: soundboard::SbChord,
    bpm: u32,
    n_triggers: u32,
    volume: f32,
    is_playing: bool,
    current_step: u32,
    trigger_count: u64,
}

impl Default for SoundboardUiState {
    fn default() -> Self {
        Self {
            waveform: soundboard::SbWaveform::Sine,
            instrument: soundboard::SbInstrument::Piano,
            root_midi: 69, // A4
            chord: soundboard::SbChord::Single,
            bpm: 120,
            n_triggers: 4,
            volume: 0.7,
            is_playing: false,
            current_step: 0,
            trigger_count: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum BrainWaveBand {
    All,
    Delta, // 0.5–4 Hz
    Theta, // 4–8 Hz
    Alpha, // 8–13 Hz
    Beta,  // 13–30 Hz
    Gamma, // 30–80 Hz
}

const SPECTRUM_FFT_SIZE: usize = 128;
const SPECTRUM_SAMPLE_RATE: f32 = cognionics::SAMPLE_RATE as f32;

impl BrainWaveBand {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Delta => "Delta",
            Self::Theta => "Theta",
            Self::Alpha => "Alpha",
            Self::Beta => "Beta",
            Self::Gamma => "Gamma",
        }
    }

    fn freq_range(self) -> (f32, f32) {
        match self {
            Self::All => (0.5, 150.0),
            Self::Delta => (0.5, 4.0),
            Self::Theta => (4.0, 8.0),
            Self::Alpha => (8.0, 13.0),
            Self::Beta => (13.0, 30.0),
            Self::Gamma => (30.0, 80.0),
        }
    }

    /// Return the (start, end) output-index range for `compute_spectrum`
    /// which returns bins 1..fft_size/2 (index 0 = bin 1).
    fn bin_range(self) -> (usize, usize) {
        let (lo_hz, hi_hz) = self.freq_range();
        let bin_hz = SPECTRUM_SAMPLE_RATE / SPECTRUM_FFT_SIZE as f32;
        // compute_spectrum output[i] corresponds to bin (i+1), freq = (i+1)*bin_hz
        let start = ((lo_hz / bin_hz).ceil() as usize).saturating_sub(1);
        let end = (hi_hz / bin_hz).floor() as usize;
        let max_len = SPECTRUM_FFT_SIZE / 2 - 1; // compute_spectrum returns buf[1..N/2], length = N/2 - 1
        (start.min(max_len), end.min(max_len))
    }

    fn hue(self) -> f32 {
        match self {
            Self::All => 0.0,
            Self::Delta => 0.75, // purple
            Self::Theta => 0.58, // cyan
            Self::Alpha => 0.33, // green
            Self::Beta => 0.15,  // orange
            Self::Gamma => 0.0,  // red
        }
    }
}

/// Application state.
struct MindDaw {
    discovered: Vec<StreamMeta>,
    paired: Option<PairedStream>,
    paired_meta: Option<StreamMeta>,
    scanning: bool,
    waveform_data: Vec<Vec<f32>>,

    // Cognionics BT state
    cog_handle: Option<CogHandle>,
    cog_state: CogState,
    cog_buffer: Vec<VecDeque<f32>>,
    cog_waveform_data: Vec<Vec<f32>>,

    // Audio sonification
    audio_handle: Option<AudioHandle>,
    audio_enabled: bool,
    selected_channel: Option<usize>,

    // PCA
    pca_state: PcaState,
    pca_yaw: f32,
    pca_pitch: f32,
    pca_dragging: bool,
    pca_last_drag_pos: Option<Point<Pixels>>,

    // Word reading
    word_read_state: WordReadState,

    // UI
    active_tab: Tab,
    spectrum_band: BrainWaveBand,

    // Soundboard
    soundboard_handle: Option<soundboard::SoundboardHandle>,
    sc_handle: Option<sc::ScHandle>,
    sc_voice: sc::Voice,
    sc_params: sc::ChordParams,
    sc_profiles: Vec<sc::SoundProfile>,
    sc_active_profile: Option<usize>,
    sb: SoundboardUiState,

    // Tonnetz / Orbifold
    tonnetz_state: tonnetz::TonnetzState,
    prev_tonnetz_chord_idx: usize,
    tonnetz_muted: bool,
    tonnetz_manual_nav: bool,

    // Calibration & Control
    calibration_state: CalibrationState,
    control_state: ControlState,
    control_decoder: ControlDecoder,
    session_log: Option<session_log::SessionLog>,
    /// Frame counter for throttled logging (log every N frames).
    log_frame_counter: u32,

    // Live detection display
    /// Recent detected events with timestamps for display.
    detected_events: std::collections::VecDeque<(std::time::Instant, String)>,
    /// Current detected state flags (updated every frame).
    detecting_blink: bool,
    detecting_jaw_clench: bool,
    /// Band power snapshot for display.
    live_band_powers: calibration::BandPowers,
    /// Last time an action-driven navigation fired (for debouncing).
    last_action_nav: std::time::Instant,

    // Recorder
    rec_ring: VecDeque<[f32; 64]>,
    rec: RecorderUiState,
    /// Backing state for the "new stimulus" text input widget.
    stimulus_input: Entity<InputState>,
    /// Backing state for the baseline profile name input widget.
    profile_name_input: Entity<InputState>,
    /// Names of saved profiles (refreshed on load/save).
    saved_profiles: Vec<String>,
}

const COG_BUFFER_CAPACITY: usize = 150;

// ── Palette ──────────────────────────────────────────────────────────────────

fn c_bg() -> Hsla      { hsla(0.62, 0.20, 0.06, 1.0) }
fn c_surface() -> Hsla  { hsla(0.62, 0.15, 0.09, 1.0) }
fn c_border() -> Hsla   { hsla(0.62, 0.15, 0.16, 1.0) }
fn c_accent() -> Hsla   { hsla(0.50, 0.85, 0.62, 1.0) }
fn c_accent_t() -> Hsla { hsla(0.50, 0.85, 0.72, 1.0) }
fn c_text() -> Hsla     { hsla(0.0, 0.0, 0.93, 1.0) }
fn c_muted() -> Hsla    { hsla(0.62, 0.06, 0.40, 1.0) }
fn c_green() -> Hsla    { hsla(0.38, 0.80, 0.55, 1.0) }
fn c_canvas() -> Hsla   { hsla(0.62, 0.18, 0.04, 1.0) }

/// Convert a StimulusEpoch (samples[time][channel]) to an FbcspTrial (channels[ch][time]).
fn epoch_to_fbcsp_trial(ep: &StimulusEpoch) -> FbcspTrial {
    let n_ch = ep.samples.first().map(|s| s.len()).unwrap_or(0);
    let channels: Vec<Vec<f32>> = (0..n_ch).map(|ch| ep.channel(ch)).collect();
    FbcspTrial {
        channels,
        label: ep.label.clone(),
    }
}

/// Convert FBCSP prediction output to a ClassifierPrediction for the UI.
fn fbcsp_to_classifier_prediction(
    label: String,
    scores: Vec<(String, f32)>,
) -> ClassifierPrediction {
    // Convert raw LDA discriminant scores to 0-1 similarities via softmax
    let max_score = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let total: f32 = exps.iter().sum();
    let similarities: std::collections::HashMap<String, f32> = scores
        .iter()
        .zip(exps.iter())
        .map(|((l, _), &e)| (l.clone(), e / total.max(1e-10)))
        .collect();
    let confidence = similarities.get(&label).copied().unwrap_or(0.0);
    let is_novel = confidence < 0.3;

    ClassifierPrediction {
        predicted_label: label,
        confidence,
        similarities,
        is_novel,
    }
}

/// Map a detected action label to a voice-leading type.
///
/// In Tymoczko's chord-space geometry, graph edges ARE voice leadings.
/// Each action consistently maps to the same kind of voice motion:
///
///   blink        → transpose up (all voices rise)
///   jaw clench   → transpose down (all voices fall)
///   right hand   → raise top voice (expand chord upward)
///   left hand    → lower top voice (contract chord from above)
///   eyes closed  → raise bottom voice (contract chord from below)
///   eyes open    → lower bottom voice (expand chord downward)
fn action_voice_leading(label: &str) -> Option<tonnetz::VoiceLeadingKind> {
    use tonnetz::VoiceLeadingKind::*;
    let l = label.to_ascii_lowercase();
    let l = l.trim();
    match l.as_ref() {
        "blink" | "blink_right" | "blink right"       => Some(TransposeUp),
        "blink_left" | "blink left"                    => Some(TransposeDown),
        "blink_both" | "blink both"                    => None, // confirm, not motion

        "jaw_clench" | "jaw clench" | "jaw"            => Some(TransposeDown),

        "motor_right_hand" | "motor right hand"
        | "right_hand" | "right hand"                   => Some(RaiseTop),
        "motor_left_hand" | "motor left hand"
        | "left_hand" | "left hand"                     => Some(LowerTop),

        "eyes_closed" | "eyes closed" | "relax"         => Some(RaiseBottom),
        "eyes_open" | "eyes open" | "focus" | "focused" => Some(LowerBottom),

        "breath_hold" | "breath hold"                    => Some(RaiseTop),

        _ => None,
    }
}

// PCA constants
const PCA_FFT_SIZE: usize = 64;
const PCA_BINS: usize = PCA_FFT_SIZE / 2;
const PCA_DIM: usize = 64 * PCA_BINS; // 2048
const PCA_K: usize = 3;
const PCA_TRAIL_LEN: usize = 128;

struct PcaState {
    weights: Vec<Vec<f32>>,
    mean: Vec<f32>,
    sample_count: u64,
    trail: VecDeque<[f32; 3]>,
    current_point: [f32; 3],
    y_ema: [f32; 3],
    y_var: [f32; 3],
}

impl PcaState {
    fn new() -> Self {
        let mut weights = vec![vec![0.0f32; PCA_DIM]; PCA_K];
        let spread = [0, PCA_DIM / 3, 2 * PCA_DIM / 3];
        for j in 0..PCA_K {
            weights[j][spread[j]] = 1.0;
        }
        Self {
            weights,
            mean: vec![0.0f32; PCA_DIM],
            sample_count: 0,
            trail: VecDeque::with_capacity(PCA_TRAIL_LEN),
            current_point: [0.0; 3],
            y_ema: [0.0; 3],
            y_var: [0.0; 3],
        }
    }

    fn update(&mut self, x_raw: &[f32]) {
        if x_raw.len() != PCA_DIM {
            return;
        }

        self.sample_count += 1;
        let count = self.sample_count;

        // 1. Update running mean via EMA
        let alpha = if count <= 100 {
            1.0 / count as f32
        } else {
            0.01
        };
        for i in 0..PCA_DIM {
            self.mean[i] += alpha * (x_raw[i] - self.mean[i]);
        }

        // 2. Center input
        let x: Vec<f32> = (0..PCA_DIM).map(|i| x_raw[i] - self.mean[i]).collect();

        // 3. Compute projections
        let mut y = [0.0f32; PCA_K];
        for j in 0..PCA_K {
            y[j] = self.weights[j]
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum();
        }

        // 4. Sanger's rule with progressive deflation
        let eta = 0.01 / (1.0 + count as f32 * 0.0001);
        let old_weights: Vec<Vec<f32>> = self.weights.clone();
        let mut x_res = x;
        for j in 0..PCA_K {
            for i in 0..PCA_DIM {
                self.weights[j][i] += eta * y[j] * x_res[i];
            }
            for i in 0..PCA_DIM {
                x_res[i] -= y[j] * self.weights[j][i];
            }
        }

        // 5. Normalize each weight vector
        for j in 0..PCA_K {
            let norm: f32 = self.weights[j].iter().map(|w| w * w).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for w in &mut self.weights[j] {
                    *w /= norm;
                }
            }
        }

        // 6. Sign correction
        for j in 0..PCA_K {
            let dot: f32 = self.weights[j]
                .iter()
                .zip(old_weights[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot < 0.0 {
                for w in &mut self.weights[j] {
                    *w = -*w;
                }
                y[j] = -y[j];
            }
        }

        // 7. Adaptive projection scaling: per-component EMA + tanh compression
        let alpha_y = 0.02f32;
        let mut pt = [0.0f32; 3];
        for j in 0..3 {
            self.y_ema[j] += alpha_y * (y[j] - self.y_ema[j]);
            let diff = y[j] - self.y_ema[j];
            self.y_var[j] += alpha_y * (diff * diff - self.y_var[j]);
            pt[j] = ((y[j] - self.y_ema[j]) / self.y_var[j].sqrt().max(1e-6)).tanh();
        }
        self.current_point = pt;
        if self.trail.len() >= PCA_TRAIL_LEN {
            self.trail.pop_front();
        }
        self.trail.push_back(pt);
    }
}

/// Clip outliers to the 5th–95th percentile range (90% central range).
fn clip_outliers(data: &mut [f32]) {
    if data.len() < 2 {
        return;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = sorted[sorted.len() * 5 / 100];
    let hi = sorted[sorted.len() * 95 / 100];
    if lo < hi {
        for v in data.iter_mut() {
            *v = v.clamp(lo, hi);
        }
    }
}

fn compute_pca_feature_vector(channel_data: &[Vec<f32>]) -> Vec<f32> {
    let mut feature = Vec::with_capacity(PCA_DIM);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(PCA_FFT_SIZE);

    for ch in 0..64 {
        let data = channel_data.get(ch).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut buf: Vec<Complex<f32>> = vec![Complex::default(); PCA_FFT_SIZE];
        let n = data.len().min(PCA_FFT_SIZE);
        let start = data.len().saturating_sub(PCA_FFT_SIZE);
        for i in 0..n {
            let w = (std::f32::consts::PI * i as f32 / PCA_FFT_SIZE as f32)
                .sin()
                .powi(2);
            buf[i] = Complex::new(data[start + i] * w, 0.0);
        }
        fft.process(&mut buf);

        for i in 0..PCA_BINS {
            feature.push(buf[i].norm());
        }
    }

    // Debias: log-compress, per-channel mean subtraction, then L2 normalize
    for ch_block in feature.chunks_mut(PCA_BINS) {
        // Log-compress to shrink dynamic range
        for v in ch_block.iter_mut() {
            *v = (1.0 + *v).ln();
        }
        // Subtract channel mean so PCA sees shape, not absolute power
        let mean = ch_block.iter().sum::<f32>() / PCA_BINS as f32;
        for v in ch_block.iter_mut() {
            *v -= mean;
        }
    }

    // L2-normalize the full vector: removes the correlated global amplitude
    // factor that causes all 3 PCA components to track the same thing.
    // In 2048 dims, direction still carries rich spectral-shape information.
    let norm = feature.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut feature {
            *v /= norm;
        }
    }

    feature
}

impl MindDaw {
    fn new(stimulus_input: Entity<InputState>, profile_name_input: Entity<InputState>) -> Self {
        let saved_profiles = recorder::storage::list_baseline_profiles();
        Self {
            discovered: Vec::new(),
            paired: None,
            paired_meta: None,
            scanning: false,
            waveform_data: Vec::new(),

            cog_handle: None,
            cog_state: CogState::Disconnected,
            cog_buffer: vec![VecDeque::with_capacity(COG_BUFFER_CAPACITY); cognionics::NUM_CHANNELS],
            cog_waveform_data: Vec::new(),

            audio_handle: None,
            audio_enabled: false,
            selected_channel: None,

            pca_state: PcaState::new(),
            pca_yaw: 0.0,
            pca_pitch: 0.0,
            pca_dragging: false,
            pca_last_drag_pos: None,

            word_read_state: WordReadState::new(),

            active_tab: Tab::Spectrum,
            spectrum_band: BrainWaveBand::All,

            soundboard_handle: None,
            sc_handle: None,
            sc_voice: sc::Voice::Pad,
            sc_params: sc::ChordParams::default(),
            sc_profiles: sc::builtin_profiles(),
            sc_active_profile: None,
            sb: SoundboardUiState::default(),

            tonnetz_state: tonnetz::TonnetzState::new(tonnetz::OrbifoldType::Dyads),
            prev_tonnetz_chord_idx: 0,
            tonnetz_muted: false,
            tonnetz_manual_nav: true,

            calibration_state: CalibrationState::new(cognionics::NUM_CHANNELS),
            control_state: ControlState::default(),
            control_decoder: ControlDecoder::new(),
            session_log: None,
            log_frame_counter: 0,

            detected_events: VecDeque::with_capacity(20),
            detecting_blink: false,
            detecting_jaw_clench: false,
            live_band_powers: calibration::BandPowers::default(),
            last_action_nav: std::time::Instant::now(),

            rec_ring: VecDeque::with_capacity(REC_RING_CAPACITY),
            rec: RecorderUiState::default(),
            stimulus_input,
            profile_name_input,
            saved_profiles,
        }
    }

    fn scan(&mut self, cx: &mut Context<Self>) {
        self.scanning = true;
        cx.notify();

        cx.spawn(async |this, cx| {
            let results = smol::unblock(|| streams::discover_streams(2.0)).await;

            this.update(cx, |this, cx| {
                this.discovered = results;
                this.scanning = false;
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn pair(&mut self, meta: StreamMeta, cx: &mut Context<Self>) {
        // StreamInlet is !Send, so we connect on the main thread.
        // This blocks briefly (~5s max) during resolve + open_stream.
        match PairedStream::connect(&meta, 512) {
            Ok(paired) => {
                self.paired_meta = Some(paired.meta.clone());
                self.paired = Some(paired);
                cx.notify();

                // Start polling loop for pulling samples (~30fps)
                cx.spawn(async |this, cx| {
                    loop {
                        smol::Timer::after(std::time::Duration::from_millis(16)).await;

                        let ok = this
                            .update(cx, |this, cx| {
                                if let Some(ref mut paired) = this.paired {
                                    paired.pull_samples();
                                    let ch = paired.meta.channel_count as usize;
                                    this.waveform_data =
                                        (0..ch).map(|c| paired.channel_data(c)).collect();

                                    // Feed recorder ring buffer with latest frame
                                    let ch_count = ch.min(64);
                                    let mut frame = [0.0f32; 64];
                                    for c in 0..ch_count {
                                        frame[c] = paired.buffer[c].back().copied().unwrap_or(0.0);
                                    }
                                    if this.rec_ring.len() >= REC_RING_CAPACITY {
                                        this.rec_ring.pop_front();
                                    }
                                    this.rec_ring.push_back(frame);

                                    // Send audio frame (build inline to avoid borrow conflict)
                                    // Disable EEG sonification on tabs where chord audio plays.
                                    if this.audio_enabled
                                        && this.active_tab != Tab::Tonnetz
                                        && this.active_tab != Tab::Calibration
                                    {
                                        if let Some(ref handle) = this.audio_handle {
                                            let frame = EegFrame {
                                                channels: (0..ch)
                                                    .map(|c| {
                                                        let buf = &paired.buffer[c];
                                                        let n = buf.len().min(64);
                                                        buf.iter().rev().take(n).rev().copied().collect()
                                                    })
                                                    .collect(),
                                            };
                                            let _ = handle.cmd_tx.try_send(AudioCommand::Frame(frame));
                                        }
                                    }

                                    // Feed calibration from LSL data
                                    this.calibration_state.feed_lsl_bufs(
                                        &this.waveform_data, 0.033,
                                    );

                                    cx.notify();
                                    true
                                } else {
                                    false
                                }
                            })
                            .unwrap_or(false);

                        if !ok {
                            break;
                        }
                    }
                })
                .detach();
            }
            Err(e) => {
                eprintln!("Failed to pair with stream: {e}");
            }
        }
    }

    // ── Audio methods ──────────────────────────────────────────────────

    fn start_audio(&mut self, num_channels: usize, cx: &mut Context<Self>) {
        if self.audio_handle.is_some() {
            return;
        }
        match audio::spawn_audio_engine(num_channels, 64) {
            Ok(handle) => {
                self.audio_handle = Some(handle);
                self.audio_enabled = true;
                cx.notify();
            }
            Err(e) => {
                eprintln!("Failed to start audio: {e}");
            }
        }
    }

    fn stop_audio(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = self.audio_handle.take() {
            let _ = handle.cmd_tx.send(AudioCommand::Stop);
        }
        self.audio_enabled = false;
        cx.notify();
    }

    fn send_audio_frame_from_cog(&self) {
        if !self.audio_enabled {
            return;
        }
        if let Some(ref handle) = self.audio_handle {
            let channels: Vec<Vec<f32>> = if let Some(ch) = self.selected_channel {
                // Single selected channel
                if let Some(buf) = self.cog_buffer.get(ch) {
                    let n = buf.len().min(64);
                    vec![buf.iter().rev().take(n).rev().copied().collect()]
                } else {
                    return;
                }
            } else {
                // All channels
                self.cog_buffer
                    .iter()
                    .map(|buf| {
                        let n = buf.len().min(64);
                        buf.iter().rev().take(n).rev().copied().collect()
                    })
                    .collect()
            };
            let _ = handle.cmd_tx.try_send(AudioCommand::Frame(EegFrame { channels }));
        }
    }

    fn select_channel(&mut self, ch: usize, cx: &mut Context<Self>) {
        if self.selected_channel == Some(ch) {
            // Deselect — stop audio
            self.selected_channel = None;
            self.stop_audio(cx);
        } else {
            // Select new channel — (re)start audio with 1 channel
            self.selected_channel = Some(ch);
            if self.audio_handle.is_some() {
                self.stop_audio(cx);
            }
            self.start_audio(1, cx);
        }
    }

    // ── Recorder methods ─────────────────────────────────────────────────

    /// Build a StimulusEpoch from the current rec_ring contents.
    fn rec_ring_to_epoch(&self, label: &str) -> Option<StimulusEpoch> {
        let n = self.rec_ring.len();
        let pre = 60usize;   // 200 ms
        let post = 240usize; // 800 ms
        let total = pre + post;
        if n < pre {
            return None;
        }
        let take = total.min(n);
        let start = n - take;
        let samples: Vec<Vec<f32>> = self.rec_ring
            .iter()
            .skip(start)
            .map(|frame| frame.to_vec())
            .collect();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Some(StimulusEpoch {
            id: uuid::Uuid::new_v4().to_string(),
            label: label.to_string(),
            timestamp: now,
            samples,
            sample_rate: 300.0,
            pre_samples: pre.min(take),
            notes: None,
        })
    }

    /// Capture an epoch from the ring buffer and put it in pending state.
    fn rec_capture_epoch(&mut self, cx: &mut Context<Self>) {
        let label = self.rec.active_stimulus.clone();
        if let Some(ep) = self.rec_ring_to_epoch(&label) {
            self.rec.pending_epoch = Some(ep);
            self.rec.mode = RecorderMode::Idle;
            cx.notify();
        }
    }

    /// Accept the pending epoch into the session.
    fn rec_accept_epoch(&mut self, cx: &mut Context<Self>) {
        if let Some(ep) = self.rec.pending_epoch.take() {
            self.rec.session.epochs.push(ep);
            self.rec.epochs_since_retrain += 1;
            if self.rec.epochs_since_retrain >= RETRAIN_EVERY {
                self.rec.epochs_since_retrain = 0;
                self.rec.classifier = TrainedClassifier::train(&self.rec.session.epochs);
                self.rec_train_fbcsp();
            }
            cx.notify();
        }
    }

    /// Train FBCSP model from the current session epochs.
    fn rec_train_fbcsp(&mut self) {
        let trials: Vec<FbcspTrial> = self
            .rec
            .session
            .epochs
            .iter()
            .map(|ep| epoch_to_fbcsp_trial(ep))
            .collect();
        let sr = self
            .rec
            .session
            .epochs
            .first()
            .map(|e| e.sample_rate)
            .unwrap_or(300.0);
        match FbcspModel::train(&trials, sr) {
            Some(model) => {
                eprintln!(
                    "[fbcsp] trained: {} classes, {} features",
                    model.labels.len(),
                    model.n_features()
                );
                self.rec.fbcsp_model = Some(model);
            }
            None => {
                eprintln!("[fbcsp] not enough data to train");
            }
        }
    }

    fn rec_reject_epoch(&mut self, cx: &mut Context<Self>) {
        self.rec.pending_epoch = None;
        cx.notify();
    }

    fn rec_save_session(&self) {
        match recorder::storage::save_session(&self.rec.session) {
            Ok(path) => eprintln!("[recorder] saved to {}", path.display()),
            Err(e) => eprintln!("[recorder] save error: {e}"),
        }
    }

    fn rec_export_csv(&self) {
        match recorder::storage::export_csv(&self.rec.session) {
            Ok(path) => eprintln!("[recorder] CSV exported to {}", path.display()),
            Err(e) => eprintln!("[recorder] CSV export error: {e}"),
        }
    }

    // ── Cognionics methods ───────────────────────────────────────────────

    fn cog_scan(&mut self, cx: &mut Context<Self>) {
        // Spawn worker if not yet running
        if self.cog_handle.is_none() {
            self.cog_handle = Some(cognionics::spawn_cog_worker());
            self.start_cog_poll(cx);
        }

        if let Some(ref handle) = self.cog_handle {
            let _ = handle.cmd_tx.send(CogCommand::StartScan);
        }

        self.cog_state = CogState::Scanning;
        cx.notify();
    }

    fn cog_demo(&mut self, cx: &mut Context<Self>) {
        let had_handle = self.cog_handle.is_some();
        if let Some(ref handle) = self.cog_handle {
            let _ = handle.cmd_tx.send(CogCommand::Shutdown);
        }
        self.cog_handle = Some(cognionics::spawn_demo_worker());
        if !had_handle {
            self.start_cog_poll(cx);
        }
        self.cog_state = CogState::Streaming;
        cx.notify();
    }

    fn cog_connect(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(ref handle) = self.cog_handle {
            let _ = handle.cmd_tx.send(CogCommand::Connect(id));
        }
        self.cog_state = CogState::Connecting;
        cx.notify();
    }

    fn cog_disconnect(&mut self, cx: &mut Context<Self>) {
        if let Some(ref handle) = self.cog_handle {
            let _ = handle.cmd_tx.send(CogCommand::Disconnect);
        }
        self.stop_audio(cx);
        self.cog_state = CogState::Disconnected;
        // Clear buffers
        for buf in &mut self.cog_buffer {
            buf.clear();
        }
        self.cog_waveform_data.clear();
        self.pca_state = PcaState::new();
        self.pca_yaw = 0.0;
        self.pca_pitch = 0.0;
        self.pca_dragging = false;
        self.pca_last_drag_pos = None;
        self.word_read_state = WordReadState::new();
        cx.notify();
    }

    /// Start a ~30fps async poll loop that drains samples and state from the BT worker.
    fn start_cog_poll(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async |this, cx| {
            loop {
                smol::Timer::after(std::time::Duration::from_millis(16)).await;

                let ok = this
                    .update(cx, |this, cx| {
                        let Some(ref handle) = this.cog_handle else {
                            return false;
                        };

                        let mut changed = false;

                        // Drain state updates
                        while let Ok(state) = handle.state_rx.try_recv() {
                            this.cog_state = state;
                            changed = true;
                        }

                        // Drain samples into ring buffers
                        while let Ok(sample) = handle.sample_rx.try_recv() {
                            for (ch, &val) in sample.channels.iter().enumerate() {
                                if ch < this.cog_buffer.len() {
                                    let buf = &mut this.cog_buffer[ch];
                                    if buf.len() >= COG_BUFFER_CAPACITY {
                                        buf.pop_front();
                                    }
                                    buf.push_back(val);
                                }
                            }
                            // Feed recorder ring buffer
                            if this.rec_ring.len() >= REC_RING_CAPACITY {
                                this.rec_ring.pop_front();
                            }
                            this.rec_ring.push_back(sample.channels);

                            // Feed baseline recorder if active
                            if let Some(ref mut brec) = this.rec.baseline_rec {
                                brec.push_sample(&sample.channels);
                                if brec.is_complete() {
                                    this.rec.baseline = this.rec.baseline_rec.take().and_then(|r| r.finalize());
                                }
                            }
                            changed = true;
                        }

                        // Update waveform snapshot
                        if changed {
                            this.cog_waveform_data = this
                                .cog_buffer
                                .iter()
                                .map(|buf| {
                                    let mut ch: Vec<f32> = buf.iter().copied().collect();
                                    clip_outliers(&mut ch);
                                    ch
                                })
                                .collect();

                            this.send_audio_frame_from_cog();

                            // PCA update
                            let features =
                                compute_pca_feature_vector(&this.cog_waveform_data);
                            this.pca_state.update(&features);
                            if !this.pca_dragging {
                                this.pca_yaw += 0.005;
                            }

                            // Word reading update
                            this.word_read_state.tick(&features);

                            // ── Calibrated EEG → ControlState → Tonnetz ──
                            // Extract band powers from the feature vector
                            let bands = control::extract_band_powers(
                                &features, PCA_BINS, 64,
                            );

                            // Detect artifacts
                            let blink = control::detect_blink(
                                &this.cog_waveform_data,
                                this.calibration_state.profile.as_ref(),
                            );
                            let jaw = control::detect_jaw_clench(
                                &this.cog_waveform_data,
                                this.calibration_state.profile.as_ref(),
                            );

                            // Store detection state for UI
                            this.detecting_blink = blink;
                            this.detecting_jaw_clench = jaw;
                            this.live_band_powers = bands.clone();
                            let now = std::time::Instant::now();
                            // Debounce: only log if the same event wasn't logged in the last 0.5s
                            if blink {
                                let recent = this.detected_events.iter().rev()
                                    .find(|(_, n)| n == "Blink")
                                    .is_some_and(|(t, _)| t.elapsed().as_secs_f32() < 0.5);
                                if !recent {
                                    this.detected_events.push_back((now, "Blink".into()));
                                }
                            }
                            if jaw {
                                let recent = this.detected_events.iter().rev()
                                    .find(|(_, n)| n == "Jaw clench")
                                    .is_some_and(|(t, _)| t.elapsed().as_secs_f32() < 0.5);
                                if !recent {
                                    this.detected_events.push_back((now, "Jaw clench".into()));
                                }
                            }
                            // Keep only last 20 events and prune old ones (> 5s)
                            while this.detected_events.len() > 20 {
                                this.detected_events.pop_front();
                            }
                            while this.detected_events.front()
                                .is_some_and(|(t, _)| t.elapsed().as_secs_f32() > 5.0)
                            {
                                this.detected_events.pop_front();
                            }

                            // Channel variance for confidence estimation
                            let ch_var = this.cog_waveform_data.iter()
                                .take(8)
                                .filter_map(|d| {
                                    if d.is_empty() { return None; }
                                    let mean = d.iter().sum::<f32>() / d.len() as f32;
                                    Some(d.iter().map(|x| (x - mean).powi(2))
                                        .sum::<f32>() / d.len() as f32)
                                })
                                .sum::<f32>() / 8.0;

                            // Decode into ControlState
                            this.control_decoder.decode(
                                &bands,
                                blink,
                                jaw,
                                ch_var,
                                this.calibration_state.profile.as_ref(),
                                0.033, // ~30 Hz frame rate
                                &mut this.control_state,
                            );

                            // Feed EEG state to sequencer for real-time modulation
                            if let Some(ref h) = this.sc_handle {
                                h.update_eeg(
                                    this.control_state.tension,
                                    this.control_state.stability,
                                    this.control_state.motion_x,
                                    this.control_state.motion_y,
                                    this.control_state.confidence_continuous,
                                );
                            }

                            // Navigate orbifold using ControlState (skip in manual nav mode)
                            if !this.tonnetz_manual_nav {
                                this.tonnetz_state.update_from_control(&this.control_state);
                            }

                            // Action-driven navigation: detected events → voice-leading jumps.
                            // Each edge in the orbifold graph IS a voice leading (Tymoczko).
                            // Actions map to voice-leading types, not spatial directions.
                            // Debounce: at most one jump per 400 ms.
                            if this.tonnetz_manual_nav
                                && this.last_action_nav.elapsed().as_secs_f32() > 0.4
                            {
                                let mut nav_kind: Option<tonnetz::VoiceLeadingKind> = None;

                                // Raw blink / jaw clench detections
                                if blink {
                                    nav_kind = action_voice_leading("blink");
                                }
                                if jaw && nav_kind.is_none() {
                                    nav_kind = action_voice_leading("jaw_clench");
                                }
                                // FBCSP classifier predictions
                                if nav_kind.is_none() {
                                    if let Some(ref pred) = this.rec.last_prediction {
                                        if pred.confidence > 0.5 && !pred.is_novel {
                                            nav_kind =
                                                action_voice_leading(&pred.predicted_label);
                                        }
                                    }
                                }

                                if let Some(kind) = nav_kind {
                                    if this.tonnetz_state.navigate_by_voice_leading(kind) {
                                        this.last_action_nav = now;
                                    }
                                }
                            }

                            // Play chord when it changes
                            if this.tonnetz_state.current_chord_idx
                                != this.prev_tonnetz_chord_idx
                            {
                                this.prev_tonnetz_chord_idx =
                                    this.tonnetz_state.current_chord_idx;
                                this.play_tonnetz_chord();

                                // Log chord change
                                if let Some(ref mut log) = this.session_log {
                                    if let Some(chord) = this.tonnetz_state.current_chord() {
                                        let midi = tonnetz::chord_to_midi_notes(chord);
                                        log.log_chord(&chord.short_label(), &midi);
                                    }
                                }
                            }

                            // Recorder: ARM auto-detect
                            if this.rec.mode == RecorderMode::Armed {
                                if detect_event(&this.rec_ring, &this.rec.thresholds).is_some() {
                                    this.rec_capture_epoch(cx);
                                }
                            }

                            // Recorder: live prediction (prefer FBCSP, fallback to cosine)
                            if this.rec.mode == RecorderMode::Predicting {
                                if let Some(ep) = this.rec_ring_to_epoch("live") {
                                    let pred = if let Some(ref fbcsp) = this.rec.fbcsp_model {
                                        // FBCSP + shrinkage LDA
                                        let trial = epoch_to_fbcsp_trial(&ep);
                                        let (label, scores) = fbcsp.predict(&trial.channels);
                                        Some(fbcsp_to_classifier_prediction(label, scores))
                                    } else if let Some(ref clf) = this.rec.classifier {
                                        // Fallback: cosine similarity
                                        let feat = extract_features(&ep);
                                        let feat = if this.rec.normalize_with_baseline {
                                            if let Some(ref bl) = this.rec.baseline {
                                                baseline_normalize(&feat, bl)
                                            } else { feat }
                                        } else { feat };
                                        Some(predict_features(&feat, clf))
                                    } else {
                                        None
                                    };

                                    if let Some(pred) = pred {
                                        if this.rec.prediction_history.len() >= 10 {
                                            this.rec.prediction_history.pop_front();
                                        }
                                        if pred.confidence > 0.4 {
                                            let label = pred.predicted_label.replace('_', " ");
                                            let last_same = this.detected_events.back()
                                                .is_some_and(|(_, n)| *n == label);
                                            if !last_same {
                                                this.detected_events.push_back((now, label));
                                            }
                                        }
                                        this.rec.prediction_history.push_back(pred.clone());
                                        this.rec.last_prediction = Some(pred);
                                    }
                                }
                            }

                            // Throttled session logging (every 10 frames ≈ 3 Hz)
                            this.log_frame_counter += 1;
                            if this.log_frame_counter % 10 == 0 {
                                if let Some(ref mut log) = this.session_log {
                                    log.log_features(&bands);
                                    log.log_control(&this.control_state);
                                    log.log_position(this.tonnetz_state.position);
                                }
                            }

                            cx.notify();
                        }

                        // Feed calibration from raw Cognionics ring buffers
                        // (not the outlier-clipped waveform data).
                        this.calibration_state.feed_raw_bufs(
                            &this.cog_buffer, 0.033,
                        );
                        // Feed action training protocol (cued trials for FBCSP)
                        this.calibration_state.feed_action_training(
                            &this.cog_buffer, 0.033,
                        );

                        // Handle control events
                        if this.control_state.take_reset() {
                            this.tonnetz_state.reset_to_home();
                        }

                        true
                    })
                    .unwrap_or(false);

                if !ok {
                    break;
                }
            }
        })
        .detach();
    }
}

impl Render for MindDaw {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let scanning = self.scanning;
        let discovered = self.discovered.clone();
        let waveform_data = self.waveform_data.clone();
        let cog_state = self.cog_state.clone();
        let cog_waveform_data = self.cog_waveform_data.clone();

        div()
            .flex()
            .flex_col()
            .size_full()
            .overflow_hidden()
            .bg(c_bg())
            // Accent strip
            .child(div().h(px(2.0)).w_full().bg(c_accent()))
            // Header
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px(px(20.0))
                    .py(px(12.0))
                    .bg(c_surface())
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.0))
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(c_accent_t())
                                    .child("mind-daw"),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(c_muted())
                                    .child("neural audio workstation"),
                            ),
                    )
                    .child(if scanning {
                        Button::new("scan")
                            .label("Scanning...")
                            .disabled(true)
                    } else {
                        Button::new("scan")
                            .label("Scan LSL")
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.scan(cx);
                            }))
                    }),
            )
            // Separator
            .child(div().h(px(1.0)).w_full().bg(c_border()))
            // ── Cognionics panel ─────────────────────────────────────────
            .child(self.render_cog_panel(&cog_state, &cog_waveform_data, cx))
            // Stream list
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .children(if discovered.is_empty() {
                        vec![div()
                            .px(px(20.0))
                            .text_color(c_muted())
                            .child(if scanning {
                                "Searching for LSL streams..."
                            } else {
                                "No streams discovered. Click Scan LSL to search."
                            })
                            .into_any_element()]
                    } else {
                        discovered
                            .iter()
                            .enumerate()
                            .map(|(i, stream)| {
                                let meta = stream.clone();
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .p_3()
                                    .rounded_md()
                                    .border_1()
                                    .border_color(cx.theme().border)
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(
                                                div()
                                                    .font_weight(FontWeight::SEMIBOLD)
                                                    .text_color(cx.theme().foreground)
                                                    .child(stream.name.clone()),
                                            )
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .text_color(cx.theme().muted_foreground)
                                                    .child(format!(
                                                        "Type: {} | Channels: {} | Rate: {:.0} Hz | ID: {}",
                                                        stream.stream_type,
                                                        stream.channel_count,
                                                        stream.sample_rate,
                                                        stream.source_id,
                                                    )),
                                            ),
                                    )
                                    .child(
                                        Button::new(SharedString::from(format!("pair-{i}")))
                                            .label("Pair")
                                            .on_click(cx.listener(move |this, _, _window, cx| {
                                                this.pair(meta.clone(), cx);
                                            })),
                                    )
                                    .into_any_element()
                            })
                            .collect()
                    }),
            )
            // Paired stream panel
            .children(self.render_lsl_panel(&waveform_data, cx))
    }
}

impl MindDaw {
    fn render_lsl_panel(
        &mut self,
        waveform_data: &[Vec<f32>],
        _cx: &mut Context<Self>,
    ) -> Option<Div> {
        let meta = self.paired_meta.as_ref()?.clone();

        Some(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .p_3()
                .rounded_md()
                .border_1()
                .border_color(gpui_component::green_500())
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .size(px(8.0))
                                .rounded_full()
                                .bg(gpui_component::green_500()),
                        )
                        .child(
                            div()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(format!("Paired: {}", meta.name)),
                        ),
                )
                .child(div().text_sm().child(format!(
                    "{} channels @ {:.0} Hz",
                    meta.channel_count, meta.sample_rate,
                )))
                .child(
                    div().flex().flex_col().gap_1().children(
                        waveform_data
                            .iter()
                            .enumerate()
                            .map(|(ch, data)| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_xs()
                                            .w(px(32.0))
                                            .child(format!("Ch{ch}")),
                                    )
                                    .child(waveform_canvas(data, meta.sample_rate as f32))
                                    .into_any_element()
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
        )
    }

    fn render_tab(
        &mut self,
        id: &'static str,
        label: &'static str,
        tab: Tab,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let is_active = self.active_tab == tab;
        div()
            .id(id)
            .flex()
            .flex_col()
            .items_center()
            .cursor(CursorStyle::PointingHand)
            .px(px(14.0))
            .py(px(8.0))
            .text_sm()
            .text_color(if is_active { c_accent_t() } else { c_muted() })
            .font_weight(if is_active { FontWeight::SEMIBOLD } else { FontWeight::NORMAL })
            .on_click(cx.listener(move |this, _, _window, cx| {
                this.active_tab = tab;
                cx.notify();
            }))
            .child(label)
            .child(
                div()
                    .mt(px(4.0))
                    .h(px(2.0))
                    .w_full()
                    .rounded_full()
                    .bg(if is_active { c_accent() } else { hsla(0.0, 0.0, 0.0, 0.0) }),
            )
    }

    fn render_cog_panel(
        &mut self,
        cog_state: &CogState,
        cog_waveform_data: &[Vec<f32>],
        cx: &mut Context<Self>,
    ) -> Div {
        let panel = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h_0()
            .gap_2()
            .mx(px(8.0))
            .my(px(6.0))
            .p_3()
            .rounded(px(8.0))
            .border_1()
            .border_color(c_border())
            .bg(c_surface());

        match cog_state {
            CogState::Disconnected => panel
                .items_center()
                .justify_center()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_3()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.0))
                                .child(
                                    div()
                                        .size(px(10.0))
                                        .rounded_full()
                                        .bg(c_muted()),
                                )
                                .child(
                                    div()
                                        .text_lg()
                                        .font_weight(FontWeight::BOLD)
                                        .text_color(c_text())
                                        .child("Cognionics HD-72"),
                                ),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(c_muted())
                                .child("64-channel EEG · 300 Hz · Bluetooth"),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_3()
                                .mt_2()
                                .child(
                                    Button::new("cog-scan")
                                        .primary()
                                        .label("Connect")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.cog_scan(cx);
                                        })),
                                )
                                .child(
                                    Button::new("cog-demo")
                                        .label("Demo Mode")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.cog_demo(cx);
                                        })),
                                ),
                        ),
                ),

            CogState::Scanning => panel
                .items_center()
                .justify_center()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_3()
                        .child(
                            div()
                                .text_lg()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(c_text())
                                .child("Scanning for devices..."),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(c_muted())
                                .child("Searching for Cognionics HD-72 via Bluetooth"),
                        ),
                ),

            CogState::Found { id, name } => {
                let device_id = id.clone();
                let display_name = name.clone();
                panel
                    .items_center()
                    .justify_center()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.0))
                                    .child(
                                        div()
                                            .size(px(10.0))
                                            .rounded_full()
                                            .bg(hsla(0.12, 0.85, 0.55, 1.0)),
                                    )
                                    .child(
                                        div()
                                            .text_lg()
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(c_text())
                                            .child(format!("Found: {display_name}")),
                                    ),
                            )
                            .child(
                                Button::new("cog-connect")
                                    .primary()
                                    .label("Connect")
                                    .on_click(cx.listener(move |this, _, _window, cx| {
                                        this.cog_connect(device_id.clone(), cx);
                                    })),
                            ),
                    )
            }

            CogState::Connecting => panel
                .items_center()
                .justify_center()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_3()
                        .child(
                            div()
                                .text_lg()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(c_accent_t())
                                .child("Connecting..."),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(c_muted())
                                .child("Establishing Bluetooth RFCOMM link"),
                        ),
                ),

            CogState::Streaming => {
                let active_tab = self.active_tab;

                let waves_tab = self.render_tab("tab-waves", "Waves", Tab::Waves, cx);
                let spectrum_tab = self.render_tab("tab-spectrum", "Spectrum", Tab::Spectrum, cx);
                let pca_tab = self.render_tab("tab-pca", "PCA", Tab::Pca, cx);
                let words_tab = self.render_tab("tab-words", "Words", Tab::Words, cx);
                let soundboard_tab = self.render_tab("tab-soundboard", "Soundboard", Tab::Soundboard, cx);
                let tonnetz_tab = self.render_tab("tab-tonnetz", "Tonnetz", Tab::Tonnetz, cx);
                let calib_tab = self.render_tab("tab-calib", "Calibrate", Tab::Calibration, cx);

                let content: Div = if active_tab == Tab::Calibration {
                    self.render_calibration_view(cx)
                } else if active_tab == Tab::Tonnetz {
                    self.render_tonnetz_view(cx)
                } else if active_tab == Tab::Soundboard {
                    self.render_soundboard_view(cx)
                } else if active_tab == Tab::Words {
                    self.render_word_read_view(cx)
                } else if active_tab == Tab::Pca {
                    self.render_pca_view(cx)
                } else if active_tab == Tab::Spectrum {
                    self.render_spectrum_grid(cog_waveform_data, cx)
                } else {
                    let half = (cog_waveform_data.len() + 1) / 2;
                    let make_col = |items: &[Vec<f32>], start: usize| {
                        div().flex().flex_col().flex_1().gap_1().children(
                            items
                                .iter()
                                .enumerate()
                                .map(|(i, data)| {
                                    let ch = start + i;
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            div()
                                                .text_xs()
                                                .w(px(32.0))
                                                .text_color(cx.theme().muted_foreground)
                                                .child(format!("Ch{ch}")),
                                        )
                                        .child(waveform_canvas(data, cognionics::SAMPLE_RATE as f32))
                                        .into_any_element()
                                })
                                .collect::<Vec<_>>(),
                        )
                    };
                    div().flex().gap_4()
                        .child(make_col(&cog_waveform_data[..half], 0))
                        .child(make_col(&cog_waveform_data[half..], half))
                };

                panel
                .border_color(hsla(0.38, 0.40, 0.25, 0.5))
                // Header row
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.0))
                                .child(
                                    div()
                                        .size(px(8.0))
                                        .rounded_full()
                                        .bg(c_green()),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(c_text())
                                        .child("Cognionics HD-72"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(c_muted())
                                        .child("64ch · 300 Hz"),
                                ),
                        )
                        .child(
                            Button::new("cog-disconnect")
                                .danger()
                                .label("Disconnect")
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.cog_disconnect(cx);
                                })),
                        ),
                )
                // Tab bar
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(2.0))
                        .child(waves_tab)
                        .child(spectrum_tab)
                        .child(pca_tab)
                        .child(words_tab)
                        .child(soundboard_tab)
                        .child(tonnetz_tab)
                        .child(calib_tab),
                )
                // Separator
                .child(div().h(px(1.0)).w_full().bg(c_border()))
                // Content
                .child(
                    div()
                        .id("tab-content-scroll")
                        .overflow_y_scroll()
                        .flex_1()
                        .min_h_0()
                        .child(content),
                )
            }

            CogState::Error(msg) => panel
                .items_center()
                .justify_center()
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_3()
                        .child(
                            div()
                                .text_sm()
                                .text_color(hsla(0.0, 0.75, 0.60, 1.0))
                                .child(msg.clone()),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(
                                    Button::new("cog-retry")
                                        .primary()
                                        .label("Retry")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.cog_scan(cx);
                                        })),
                                )
                                .child(
                                    Button::new("cog-demo-err")
                                        .label("Demo Mode")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.cog_demo(cx);
                                        })),
                                ),
                        ),
                ),
        }
    }
}

/// Auto-correlation analysis: returns (display_offset, period_in_samples).
///
/// `display_offset` is the best offset for stable oscilloscope triggering.
/// `period` is the dominant repeating period found via autocorrelation peak
/// detection (first peak after the zero-lag). Returns 0 if no period found.
fn autocorrelate_analysis(data: &[f32], display_len: usize) -> (usize, usize) {
    if data.len() <= display_len {
        return (0, 0);
    }

    let search_len = (data.len() - display_len).min(display_len);
    if search_len < 4 {
        return (0, 0);
    }

    let reference = &data[..display_len.min(data.len())];

    // Compute normalized autocorrelation for each lag
    let mut corrs = Vec::with_capacity(search_len);
    for lag in 0..search_len {
        let mut corr = 0.0f32;
        let compare_len = display_len.min(data.len() - lag);
        for i in 0..compare_len {
            corr += reference[i] * data[lag + i];
        }
        corrs.push(corr);
    }

    // Find best offset (max correlation for display triggering)
    let mut best_offset = 0;
    let mut best_corr = f32::NEG_INFINITY;
    for (lag, &corr) in corrs.iter().enumerate().skip(1) {
        if corr > best_corr {
            best_corr = corr;
            best_offset = lag;
        }
    }

    // Find dominant period: first peak in autocorrelation after zero-lag.
    // Skip very short lags (< 3 samples) to avoid noise.
    let zero_corr = corrs[0].max(1e-10);
    let min_lag = 3;
    let mut period = 0;
    for lag in (min_lag + 1)..search_len.saturating_sub(1) {
        // A peak: higher than both neighbors and above 20% of zero-lag energy
        if corrs[lag] > corrs[lag - 1]
            && corrs[lag] > corrs[lag + 1]
            && corrs[lag] > zero_corr * 0.2
        {
            period = lag;
            break;
        }
    }

    (best_offset, period)
}

/// Decompose a signal into brain wave frequency bands via FFT bandpass + IFFT.
/// Returns (reconstructed_signal, hue) for each of the 5 bands.
fn decompose_into_bands(data: &[f32], sample_rate: f32) -> Vec<(Vec<f32>, f32)> {
    use rustfft::num_complex::Complex;

    let n = data.len();
    if n < 4 {
        return Vec::new();
    }

    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(n);

    let mut buf: Vec<Complex<f32>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_fwd.process(&mut buf);

    let bin_hz = sample_rate / n as f32;
    let scale = 1.0 / n as f32;

    let bands: [(f32, f32, f32); 5] = [
        (0.5, 4.0, BrainWaveBand::Delta.hue()),
        (4.0, 8.0, BrainWaveBand::Theta.hue()),
        (8.0, 13.0, BrainWaveBand::Alpha.hue()),
        (13.0, 30.0, BrainWaveBand::Beta.hue()),
        (30.0, 80.0, BrainWaveBand::Gamma.hue()),
    ];

    bands
        .iter()
        .map(|&(lo, hi, hue)| {
            let mut filtered = vec![Complex::new(0.0, 0.0); n];
            for k in 0..n {
                let freq = if k <= n / 2 {
                    k as f32 * bin_hz
                } else {
                    (n - k) as f32 * bin_hz
                };
                if freq >= lo && freq < hi {
                    filtered[k] = buf[k];
                }
            }
            let fft_inv = planner.plan_fft_inverse(n);
            fft_inv.process(&mut filtered);
            let signal: Vec<f32> = filtered.iter().map(|c| c.re * scale).collect();
            (signal, hue)
        })
        .collect()
}

/// Prepaint state for waveform canvas.
struct WaveformPrepaint {
    bounds: Bounds<Pixels>,
    points: Vec<(f32, f32)>,
    /// Per-band reconstructed traces: (points, hue).
    band_traces: Vec<(Vec<(f32, f32)>, f32)>,
    /// Pixel X positions of period markers (vertical bars).
    period_xs: Vec<f32>,
    /// Pixel X positions of 0.5s time markers.
    time_marker_xs: Vec<f32>,
    /// Segments where adjacent samples are equal (disconnected signal).
    flat_segments: Vec<(f32, f32, f32, f32)>,
}

// ── Radar / spider chart for classifier deviation map ─────────────────────────

struct RadarPrepaint {
    bounds: Bounds<Pixels>,
    /// Outer polygon axes (x, y) per class at full radius.
    axes: Vec<(f32, f32)>,
    /// Inner polygon (x, y) per class at similarity radius.
    poly: Vec<(f32, f32)>,
    /// Centre point.
    cx: f32,
    cy: f32,
}

fn radar_canvas(classes: &[(String, f32)]) -> impl IntoElement {
    let classes = classes.to_vec();
    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();
            let cx = ox + w / 2.0;
            let cy = oy + h / 2.0;
            let radius = (w.min(h) / 2.0 - 16.0).max(1.0);
            let n = classes.len();
            if n == 0 {
                return RadarPrepaint {
                    bounds,
                    axes: vec![],
                    poly: vec![],
                    cx,
                    cy,
                };
            }

            let mut axes = Vec::with_capacity(n);
            let mut poly = Vec::with_capacity(n);

            for (i, (_label, sim)) in classes.iter().enumerate() {
                let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32
                    - std::f32::consts::FRAC_PI_2; // start from top
                let ax = cx + radius * angle.cos();
                let ay = cy + radius * angle.sin();
                axes.push((ax, ay));

                let pr = sim.clamp(0.0, 1.0) * radius;
                poly.push((cx + pr * angle.cos(), cy + pr * angle.sin()));
            }

            RadarPrepaint { bounds, axes, poly, cx, cy }
        },
        move |_bounds, state: RadarPrepaint, window: &mut Window, _cx: &mut App| {
            if state.axes.is_empty() {
                return;
            }

            // Background
            window.paint_quad(gpui::fill(state.bounds, c_canvas()));

            let cx = state.cx;
            let cy = state.cy;

            // Draw axes from centre to each vertex
            for &(ax, ay) in &state.axes {
                let mut b = PathBuilder::stroke(px(0.5));
                b.move_to(point(px(cx), px(cy)));
                b.line_to(point(px(ax), px(ay)));
                if let Ok(p) = b.build() {
                    window.paint_path(p, gpui::hsla(0.0, 0.0, 0.3, 1.0));
                }
            }

            // Outer reference polygon
            if state.axes.len() >= 2 {
                let mut b = PathBuilder::stroke(px(0.5));
                b.move_to(point(px(state.axes[0].0), px(state.axes[0].1)));
                for &(ax, ay) in &state.axes[1..] {
                    b.line_to(point(px(ax), px(ay)));
                }
                b.line_to(point(px(state.axes[0].0), px(state.axes[0].1)));
                if let Ok(p) = b.build() {
                    window.paint_path(p, gpui::hsla(0.0, 0.0, 0.25, 1.0));
                }

                // Filled similarity polygon
                let mut b = PathBuilder::stroke(px(2.0));
                b.move_to(point(px(state.poly[0].0), px(state.poly[0].1)));
                for &(px_val, py) in &state.poly[1..] {
                    b.line_to(point(px(px_val), px(py)));
                }
                b.line_to(point(px(state.poly[0].0), px(state.poly[0].1)));
                if let Ok(p) = b.build() {
                    window.paint_path(p, gpui::hsla(0.33, 0.85, 0.55, 0.9));
                }

                // Vertex dots
                for &(px_val, py) in &state.poly {
                    let sz = 5.0;
                    let dot = Bounds {
                        origin: point(px(px_val - sz / 2.0), px(py - sz / 2.0)),
                        size: size(px(sz), px(sz)),
                    };
                    window.paint_quad(gpui::fill(dot, gpui::hsla(0.33, 0.9, 0.7, 1.0)));
                }
            }
        },
    )
    .w_full()
    .h(px(160.0))
}

// ── Baseline dashboard (expanded) ─────────────────────────────────────────────

/// Full baseline dashboard rendered below the status strip when expanded.
// ── Standard 10-20 electrode positions ───────────────────────────────────────
// (x, y) in normalised [-1, 1] head coords.  x: left(−) to right(+),
// y: posterior(−) to anterior(+).  Matches the 64-channel layout defined in
// the Python export script (Fp1 first, PO8 last).
/// Top-down azimuthal (x, y) positions for the Cognionics HD-72 64-channel
/// electrode layout, sourced from the official LSL app channel config
/// (github.com/labstreaminglayer/App-Cognionics).
/// x: left(-) → right(+), y: posterior(-) → anterior(+), radius ≈ 1.
const CH_POS: [(f32, f32); 64] = [
    (-0.47,  0.75), // 0  AF7h
    (-0.25,  0.82), // 1  AFp3
    ( 0.00,  0.88), // 2  AFPz
    ( 0.25,  0.82), // 3  AFp4
    ( 0.47,  0.75), // 4  AF8h
    (-0.53,  0.60), // 5  F5h
    (-0.30,  0.67), // 6  AFF3
    (-0.10,  0.70), // 7  AFF1
    ( 0.00,  0.72), // 8  AFFz
    ( 0.10,  0.70), // 9  AFF2
    ( 0.30,  0.67), // 10 AFF4
    ( 0.53,  0.60), // 11 F6h
    (-0.63,  0.28), // 12 FC5
    (-0.38,  0.42), // 13 FFC3
    (-0.19,  0.46), // 14 FFC3h
    (-0.09,  0.48), // 15 FFC1
    ( 0.00,  0.50), // 16 FFCz
    ( 0.09,  0.48), // 17 FFC2
    ( 0.19,  0.46), // 18 FFC4h
    ( 0.38,  0.42), // 19 FFC4
    ( 0.63,  0.28), // 20 FC6
    (-0.72,  0.14), // 21 FCC5h
    (-0.40,  0.22), // 22 FCC3
    (-0.20,  0.25), // 23 FCC3h
    (-0.10,  0.26), // 24 FCC1h
    ( 0.00,  0.27), // 25 FCCz
    ( 0.10,  0.26), // 26 FCC2h
    ( 0.20,  0.25), // 27 FCC4h
    ( 0.40,  0.22), // 28 FCC4
    ( 0.72,  0.14), // 29 FCC6h
    (-0.72, -0.14), // 30 CCP5h
    (-0.40, -0.22), // 31 CCP3
    (-0.20, -0.25), // 32 CCP3h
    (-0.10, -0.26), // 33 CCP1
    ( 0.00, -0.27), // 34 CCPz
    ( 0.10, -0.26), // 35 CCP2
    ( 0.20, -0.25), // 36 CCP4h
    ( 0.40, -0.22), // 37 CCP4
    ( 0.72, -0.14), // 38 CCP6h
    (-0.63, -0.28), // 39 CP5
    (-0.38, -0.42), // 40 CPP3
    (-0.19, -0.46), // 41 CPP3h
    (-0.09, -0.48), // 42 CPP1
    ( 0.00, -0.50), // 43 CPPz
    ( 0.09, -0.48), // 44 CPP2
    ( 0.19, -0.46), // 45 CPP4h
    ( 0.38, -0.42), // 46 CPP4
    ( 0.63, -0.28), // 47 CP6
    (-0.53, -0.60), // 48 P5h
    (-0.38, -0.67), // 49 PPO5
    (-0.22, -0.70), // 50 PPO3
    (-0.10, -0.72), // 51 PO1
    ( 0.00, -0.74), // 52 PPOz
    ( 0.10, -0.72), // 53 PO2
    ( 0.22, -0.70), // 54 PPO4
    ( 0.38, -0.67), // 55 PPO6
    ( 0.53, -0.60), // 56 P6h
    (-0.60, -0.75), // 57 PPO9h
    (-0.40, -0.80), // 58 POO7
    (-0.20, -0.86), // 59 O1
    ( 0.00, -0.90), // 60 POOz
    ( 0.20, -0.86), // 61 O2
    ( 0.40, -0.80), // 62 POO8
    ( 0.60, -0.75), // 63 PPO10h
];

// ── PSD chart ─────────────────────────────────────────────────────────────────

struct PsdPrepaint {
    bounds: Bounds<Pixels>,
    avg_pts: Vec<(f32, f32)>,
    band_rects: Vec<(f32, f32, f32)>, // (x_px, width_px, hue)
    boundary_xs: Vec<f32>,
}

/// Render a Power Spectral Density line chart (0–60 Hz) averaged across all channels.
/// Band regions are colour-coded in the background.
fn psd_chart(mean_spectrum: &[Vec<f32>], sample_rate: f32) -> impl IntoElement {
    const BANDS: [(f32, f32); 5] = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 80.0)];
    const HUES: [f32; 5] = [0.72, 0.55, 0.33, 0.1, 0.0];

    let n_bins = mean_spectrum.first().map(|s| s.len()).unwrap_or(128);
    let n_ch = mean_spectrum.len().max(1);
    let bin_hz = sample_rate / (n_bins as f32 * 2.0);

    // Average across all channels
    let mut avg = vec![0.0f32; n_bins];
    for ch_spec in mean_spectrum {
        for (i, &v) in ch_spec.iter().enumerate().take(n_bins) {
            avg[i] += v;
        }
    }
    for v in &mut avg { *v /= n_ch as f32; }

    // Trim to 60 Hz
    let show_bins = ((60.0_f32 / bin_hz).ceil() as usize).min(n_bins);
    let avg = avg[..show_bins].to_vec();
    let freq_max_hz = show_bins as f32 * bin_hz;

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();

            if avg.len() < 2 || w < 4.0 || h < 4.0 {
                return PsdPrepaint { bounds, avg_pts: vec![], band_rects: vec![], boundary_xs: vec![] };
            }

            let pad = 4.0f32;
            let pw = w - pad * 2.0;
            let ph = h - pad * 2.0;
            let max_val = avg.iter().copied().fold(0.0f32, f32::max).max(1e-10);

            let freq_to_x = |f: f32| ox + pad + (f / freq_max_hz) * pw;
            let amp_to_y  = |a: f32| oy + pad + ph * (1.0 - (a / max_val).clamp(0.0, 1.0));

            // Band background rectangles
            let band_rects: Vec<(f32, f32, f32)> = BANDS.iter().zip(HUES.iter())
                .map(|(&(lo, hi), &hue)| {
                    let x0 = freq_to_x(lo);
                    let x1 = freq_to_x(hi.min(freq_max_hz));
                    (x0, (x1 - x0).max(0.0), hue)
                })
                .collect();

            // Band boundary vertical lines at 4, 8, 13, 30 Hz
            let boundary_xs: Vec<f32> = [4.0f32, 8.0, 13.0, 30.0]
                .iter().map(|&f| freq_to_x(f)).collect();

            // Average spectrum polyline
            let avg_pts: Vec<(f32, f32)> = avg.iter().enumerate()
                .map(|(i, &v)| (freq_to_x(i as f32 * bin_hz), amp_to_y(v)))
                .collect();

            PsdPrepaint { bounds, avg_pts, band_rects, boundary_xs }
        },
        move |_bounds: Bounds<Pixels>, state: PsdPrepaint, window: &mut Window, _cx: &mut App| {
            let bounds = state.bounds;
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.07, 1.0)));
            window.paint_quad(gpui::outline(bounds, gpui::hsla(0.0, 0.0, 0.22, 1.0), gpui::BorderStyle::Solid));

            // Band-coloured backgrounds
            for &(x, bw, hue) in &state.band_rects {
                if bw > 0.0 {
                    window.paint_quad(gpui::fill(
                        Bounds {
                            origin: point(px(x), bounds.origin.y),
                            size: gpui::Size { width: px(bw), height: bounds.size.height },
                        },
                        gpui::hsla(hue, 0.6, 0.11, 0.7),
                    ));
                }
            }

            // Band boundary lines
            let h: f32 = bounds.size.height.into();
            let oy: f32 = bounds.origin.y.into();
            for &x in &state.boundary_xs {
                let mut ln = PathBuilder::stroke(px(0.5));
                ln.move_to(point(px(x), px(oy)));
                ln.line_to(point(px(x), px(oy + h)));
                if let Ok(p) = ln.build() {
                    window.paint_path(p, gpui::hsla(0.0, 0.0, 0.32, 0.7));
                }
            }

            // Average spectrum line
            if state.avg_pts.len() >= 2 {
                let mut builder = PathBuilder::stroke(px(1.5));
                builder.move_to(point(px(state.avg_pts[0].0), px(state.avg_pts[0].1)));
                for &(x, y) in &state.avg_pts[1..] {
                    builder.line_to(point(px(x), px(y)));
                }
                if let Ok(p) = builder.build() {
                    window.paint_path(p, gpui::hsla(0.0, 0.0, 0.88, 1.0));
                }
            }
        },
    )
}

// ── Topographic scalp map ─────────────────────────────────────────────────────

struct TopoPrepaint {
    bounds: Bounds<Pixels>,
    head_pts: Vec<(f32, f32)>,
    nose_pts: [(f32, f32); 3],
    // (dot_x, dot_y, hue, sat, lit)
    electrode_dots: Vec<(f32, f32, f32, f32, f32)>,
}

/// Render a 2-D scalp topographic map coloured by band power for `band` (0–4).
/// Blue = low power, red = high power.
fn topo_map(band_powers: &[[f32; 5]], band: usize) -> impl IntoElement {
    // Extract per-channel power for the selected band
    let powers: Vec<f32> = (0..64)
        .map(|ch| band_powers.get(ch).map(|p| p[band]).unwrap_or(0.0))
        .collect();

    let min_p = powers.iter().copied().fold(f32::INFINITY, f32::min);
    let max_p = powers.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_p - min_p).max(1e-10);

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();

            let cx = ox + w / 2.0;
            let cy = oy + h / 2.0;
            // Head radii — slightly taller than wide, with a margin
            let rx = (w * 0.40).min(h * 0.38);
            let ry = rx * 1.07;

            // Head oval (64-segment polyline)
            let head_pts: Vec<(f32, f32)> = (0..=64)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 64.0;
                    (cx + rx * a.sin(), cy - ry * a.cos())
                })
                .collect();

            // Nose triangle at the top
            let nose_ty = oy + h * 0.03;
            let nose_by = cy - ry * 0.90;
            let nose_hw = rx * 0.07;
            let nose_pts = [
                (cx,            nose_ty),
                (cx - nose_hw,  nose_by),
                (cx + nose_hw,  nose_by),
            ];

            // Electrode dots — (dot_x, dot_y, hue, sat, lit)
            let dot_r = 4.5f32;
            let electrode_dots: Vec<(f32, f32, f32, f32, f32)> = powers.iter()
                .enumerate()
                .map(|(ch, &p)| {
                    let (nx, ny) = CH_POS.get(ch).copied().unwrap_or((0.0, 0.0));
                    let ex = cx + nx * rx;
                    let ey = cy - ny * ry;
                    let t = ((p - min_p) / range).clamp(0.0, 1.0);
                    // Colormap: blue (0.67) → cyan → green → yellow → red (0.0)
                    let hue = 0.67 - 0.67 * t;
                    let sat = 0.85f32;
                    let lit = 0.35 + 0.25 * t;
                    (ex - dot_r, ey - dot_r, hue, sat, lit)
                })
                .collect();

            TopoPrepaint { bounds, head_pts, nose_pts, electrode_dots }
        },
        move |_bounds: Bounds<Pixels>, state: TopoPrepaint, window: &mut Window, _cx: &mut App| {
            let bounds = state.bounds;
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.07, 1.0)));

            // Head outline
            if state.head_pts.len() >= 2 {
                let mut outline = PathBuilder::stroke(px(1.5));
                outline.move_to(point(px(state.head_pts[0].0), px(state.head_pts[0].1)));
                for &(x, y) in &state.head_pts[1..] {
                    outline.line_to(point(px(x), px(y)));
                }
                if let Ok(p) = outline.build() {
                    window.paint_path(p, gpui::hsla(0.0, 0.0, 0.40, 1.0));
                }
            }

            // Nose
            let [a, b, c] = state.nose_pts;
            let mut nose = PathBuilder::stroke(px(1.5));
            nose.move_to(point(px(b.0), px(b.1)));
            nose.line_to(point(px(a.0), px(a.1)));
            nose.line_to(point(px(c.0), px(c.1)));
            if let Ok(p) = nose.build() {
                window.paint_path(p, gpui::hsla(0.0, 0.0, 0.40, 1.0));
            }

            // Electrode dots (9×9 px squares)
            let dot_sz = px(9.0);
            for &(x, y, hue, sat, lit) in &state.electrode_dots {
                window.paint_quad(gpui::fill(
                    Bounds {
                        origin: point(px(x), px(y)),
                        size: gpui::Size { width: dot_sz, height: dot_sz },
                    },
                    gpui::hsla(hue, sat, lit, 1.0),
                ));
            }
        },
    )
}

/// Laid out as two side-by-side panels:
///   Left: channel quality grid + IAF / FAA gauges
///   Right: global band-power profile + per-region dominant-band chips
fn baseline_dashboard_expanded(bl: &BaselineProfile, selected_band: usize, cx: &mut App) -> impl IntoElement {
    // ── LEFT: quality heatmap + gauges ────────────────────────────────────────
    let quality = bl.channel_quality.clone();
    let dominant = bl.dominant_band.clone();

    // 8×8 channel quality grid
    let mut grid = div()
        .flex()
        .flex_col()
        .gap(px(1.5));
    for row in 0..8usize {
        let mut row_div = div().flex().gap(px(1.5));
        for col in 0..8usize {
            let ch = row * 8 + col;
            let q = quality.get(ch).copied().unwrap_or(0.5);
            let dom = dominant.get(ch).copied().unwrap_or(2);
            // Quality determines lightness; dominant band provides hue hint
            let hue = BAND_HUES[dom];
            let lit = 0.15 + q * 0.40;
            let sat = 0.6 + q * 0.3;
            let cell = div()
                .w(px(13.0))
                .h(px(13.0))
                .rounded_sm()
                .bg(gpui::hsla(hue, sat, lit, 1.0))
                .flex()
                .items_center()
                .justify_center()
                .child(
                    div()
                        .text_color(gpui::hsla(0.0, 0.0, 0.0, 0.5))
                        .child(""), // no text — too small; tooltip would need hover state
                );
            row_div = row_div.child(cell);
        }
        grid = grid.child(row_div);
    }

    // Legend row for grid
    let grid_legend = div()
        .flex()
        .gap_3()
        .mt(px(4.0))
        .child(div().w(px(10.0)).h(px(10.0)).rounded_sm().bg(gpui::hsla(0.33, 0.8, 0.45, 1.0)))
        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0)).child("clean"))
        .child(div().w(px(10.0)).h(px(10.0)).rounded_sm().bg(gpui::hsla(0.1, 0.8, 0.35, 1.0)))
        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0)).child("noisy"))
        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0)).child("(hue = dominant band)"));

    // IAF gauge
    let iaf_pos = bl.iaf_gauge(); // 0-1 within 8–13 Hz
    let iaf_gauge = div()
        .flex()
        .flex_col()
        .gap(px(3.0))
        .mt_2()
        .child(
            div().flex().items_center().gap_2()
                .child(div().text_xs().font_weight(FontWeight::SEMIBOLD)
                    .text_color(gpui::hsla(0.0, 0.0, 0.7, 1.0))
                    .child("Individual Alpha Frequency"))
                .child(div().text_sm().font_weight(FontWeight::BOLD)
                    .text_color(gpui::hsla(0.33, 0.8, 0.65, 1.0))
                    .child(format!("{:.1} Hz", bl.iaf_hz))),
        )
        .child(
            div().flex().flex_col().gap(px(2.0))
                .child(
                    // Track
                    div().relative().w(px(180.0)).h(px(6.0)).rounded_full()
                        .bg(gpui::hsla(0.33, 0.3, 0.2, 1.0))
                        .child(
                            // Indicator dot
                            div()
                                .absolute()
                                .top(px(-1.0))
                                .left(px(iaf_pos * 172.0))
                                .w(px(8.0))
                                .h(px(8.0))
                                .rounded_full()
                                .bg(gpui::hsla(0.33, 0.9, 0.65, 1.0)),
                        ),
                )
                .child(
                    div().flex().justify_between().w(px(180.0))
                        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0)).child("8 Hz"))
                        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0)).child("10.5"))
                        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0)).child("13 Hz")),
                ),
        )
        .child(
            div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.45, 1.0))
                .child("Your alpha peak — bands are most accurate when centred here"),
        );

    // FAA gauge
    let faa_pos = bl.faa_gauge(); // 0-1
    let faa_gauge = div()
        .flex()
        .flex_col()
        .gap(px(3.0))
        .mt_2()
        .child(
            div().flex().items_center().gap_2()
                .child(div().text_xs().font_weight(FontWeight::SEMIBOLD)
                    .text_color(gpui::hsla(0.0, 0.0, 0.7, 1.0))
                    .child("Frontal Alpha Asymmetry"))
                .child(div().text_sm().font_weight(FontWeight::BOLD)
                    .text_color(if bl.faa > 0.1 {
                        gpui::hsla(0.33, 0.8, 0.65, 1.0)
                    } else if bl.faa < -0.1 {
                        gpui::hsla(0.0, 0.8, 0.65, 1.0)
                    } else {
                        gpui::hsla(0.0, 0.0, 0.65, 1.0)
                    })
                    .child(format!("{:+.2} — {}", bl.faa, bl.faa_label()))),
        )
        .child(
            div().flex().flex_col().gap(px(2.0))
                .child(
                    div().relative().w(px(180.0)).h(px(6.0)).rounded_full()
                        // Gradient-ish: red left, grey centre, green right
                        .bg(gpui::hsla(0.0, 0.0, 0.2, 1.0))
                        .child(
                            div()
                                .absolute()
                                .top(px(-1.0))
                                .left(px(faa_pos * 172.0))
                                .w(px(8.0))
                                .h(px(8.0))
                                .rounded_full()
                                .bg(if bl.faa > 0.1 {
                                    gpui::hsla(0.33, 0.9, 0.65, 1.0)
                                } else if bl.faa < -0.1 {
                                    gpui::hsla(0.0, 0.9, 0.65, 1.0)
                                } else {
                                    gpui::hsla(0.0, 0.0, 0.65, 1.0)
                                }),
                        ),
                )
                .child(
                    div().flex().justify_between().w(px(180.0))
                        .child(div().text_xs().text_color(gpui::hsla(0.0, 0.8, 0.55, 1.0)).child("← withdrawal"))
                        .child(div().text_xs().text_color(gpui::hsla(0.33, 0.8, 0.55, 1.0)).child("approach →")),
                ),
        )
        .child(
            div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.45, 1.0))
                .child("ln(right frontal α) − ln(left frontal α) · positive = right-dominant"),
        );

    let left_panel = div()
        .flex()
        .flex_col()
        .gap_3()
        .w(px(220.0))
        .flex_shrink_0()
        .child(
            div().text_xs().font_weight(FontWeight::SEMIBOLD)
                .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                .child("CHANNEL QUALITY — 64 electrodes"),
        )
        .child(grid)
        .child(grid_legend)
        .child(iaf_gauge)
        .child(faa_gauge);

    // ── RIGHT: band powers + region breakdown ──────────────────────────────
    let global_ratios = bl.global_band_ratios();

    let mut band_bars = div().flex().flex_col().gap(px(5.0));
    for (i, &name) in BAND_NAMES.iter().enumerate() {
        let ratio = global_ratios[i];
        let hue = BAND_HUES[i];
        let bar_w = (ratio * 220.0) as u32;
        let bar_w = bar_w.max(2);
        band_bars = band_bars.child(
            div().flex().items_center().gap_2()
                .child(
                    div().w(px(58.0)).text_xs()
                        .text_color(gpui::hsla(hue, 0.8, 0.7, 1.0))
                        .child(name),
                )
                .child(
                    div().flex_1().h(px(10.0)).rounded_sm()
                        .bg(gpui::hsla(0.0, 0.0, 0.12, 1.0))
                        .child(
                            div().h(px(10.0)).rounded_sm()
                                .bg(gpui::hsla(hue, 0.75, 0.45, 1.0))
                                .w(px(bar_w as f32)),
                        ),
                )
                .child(
                    div().w(px(30.0)).text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("{:.0}%", ratio * 100.0)),
                ),
        );
    }

    // Region breakdown — dominant band per region
    let region_chips = div().flex().flex_wrap().gap_2().mt_2();
    let region_chips = REGION_NAMES.iter().enumerate().fold(region_chips, |chips, (ri, &rname)| {
        let ratios = bl.region_band_ratios(ri);
        let dom_band = ratios
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2);
        let hue = BAND_HUES[dom_band];
        let sym = BAND_SYMS[dom_band];
        chips.child(
            div()
                .flex()
                .items_center()
                .gap_1()
                .px(px(7.0))
                .py(px(3.0))
                .rounded_md()
                .bg(gpui::hsla(hue, 0.4, 0.14, 1.0))
                .border_1()
                .border_color(gpui::hsla(hue, 0.6, 0.35, 0.7))
                .child(
                    div().text_xs().text_color(cx.theme().muted_foreground)
                        .child(rname),
                )
                .child(
                    div().text_xs().font_weight(FontWeight::BOLD)
                        .text_color(gpui::hsla(hue, 0.9, 0.72, 1.0))
                        .child(sym),
                ),
        )
    });

    // Signal quality summary
    let good_chs = quality.iter().filter(|&&q| q > 0.7).count();
    let bad_chs = quality.iter().filter(|&&q| q < 0.4).count();
    let quality_summary = div()
        .flex()
        .items_center()
        .gap_3()
        .mt_2()
        .child(
            div().text_xs().text_color(gpui::hsla(0.33, 0.8, 0.6, 1.0))
                .child(format!("✓ {} clean", good_chs)),
        )
        .child(
            div().text_xs().text_color(gpui::hsla(0.1, 0.8, 0.6, 1.0))
                .child(format!("~ {} marginal", 64 - good_chs - bad_chs)),
        )
        .child(
            div().text_xs().text_color(gpui::hsla(0.0, 0.8, 0.6, 1.0))
                .child(format!("✗ {} noisy", bad_chs)),
        )
        .child(
            div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.35, 1.0))
                .child("— adjust headset on red channels"),
        );

    // MNE / FOOOF summary row
    let source_badge = if bl.mne_processed {
        div().flex().items_center().gap_2()
            .child(
                div().text_xs().px(px(5.0)).py(px(2.0)).rounded_sm()
                    .bg(gpui::hsla(0.55, 0.6, 0.18, 1.0))
                    .border_1()
                    .border_color(gpui::hsla(0.55, 0.8, 0.4, 0.6))
                    .text_color(gpui::hsla(0.55, 0.8, 0.70, 1.0))
                    .child("MNE pipeline"),
            )
            .child({
                let fooof_text = if bl.fooof_r2 > 0.01 {
                    format!("1/f exponent {:.2}  offset {:.1}  R²={:.3}",
                        bl.fooof_exponent, bl.fooof_offset, bl.fooof_r2)
                } else {
                    "FOOOF not computed (install fooof/specparam)".to_string()
                };
                div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.55, 1.0)).child(fooof_text)
            })
    } else {
        div().flex().items_center().gap_2()
            .child(
                div().text_xs().px(px(5.0)).py(px(2.0)).rounded_sm()
                    .bg(gpui::hsla(0.08, 0.5, 0.18, 1.0))
                    .border_1()
                    .border_color(gpui::hsla(0.08, 0.7, 0.4, 0.5))
                    .text_color(gpui::hsla(0.08, 0.8, 0.65, 1.0))
                    .child("Rust preview"),
            )
            .child(
                div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.45, 1.0))
                    .child("Click \"Save + MNE\" to run the full MNE pipeline → ASR, ICA-ready, FOOOF"),
            )
    };

    let right_panel = div()
        .flex()
        .flex_col()
        .flex_1()
        .gap_3()
        .child(source_badge)
        .child(
            div().text_xs().font_weight(FontWeight::SEMIBOLD)
                .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                .child("RESTING-STATE BAND POWER  (global average)"),
        )
        .child(band_bars)
        .child(
            div().text_xs().font_weight(FontWeight::SEMIBOLD)
                .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                .mt_1()
                .child("DOMINANT BAND BY REGION"),
        )
        .child(region_chips)
        .child(quality_summary)
        .child(
            div().text_xs().text_color(gpui::hsla(0.0, 0.0, 0.35, 1.0)).mt_1()
                .child("Classifier normalisation divides live band powers by these baselines, \
                        surfacing deviations from your rest state rather than absolute signal strength."),
        );

    // ── BOTTOM ROW: PSD chart + topographic map ───────────────────────────────
    let muted = gpui::hsla(0.0, 0.0, 0.40, 1.0);
    let has_spectrum = !bl.mean_spectrum.is_empty();

    // PSD section
    let mut psd_inner = div().flex().flex_col().flex_1().gap_1()
        .child(
            div().text_xs().font_weight(FontWeight::SEMIBOLD)
                .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                .child("POWER SPECTRAL DENSITY  (64-ch mean)"),
        )
        .child({
            // Band legend
            let legend_items = [
                ("δ 0.5–4",  BAND_HUES[0]),
                ("θ 4–8",    BAND_HUES[1]),
                ("α 8–13",   BAND_HUES[2]),
                ("β 13–30",  BAND_HUES[3]),
                ("γ 30+",    BAND_HUES[4]),
            ];
            let mut row = div().flex().items_center().gap_3();
            for (label, hue) in legend_items {
                row = row.child(
                    div().flex().items_center().gap_1()
                        .child(div().w(px(8.0)).h(px(8.0)).rounded_sm()
                            .bg(gpui::hsla(hue, 0.65, 0.45, 1.0)))
                        .child(div().text_xs().text_color(gpui::hsla(hue, 0.8, 0.62, 1.0))
                            .child(label)),
                );
            }
            row
        });

    psd_inner = if has_spectrum {
        psd_inner.child(
            div().w_full().h(px(120.0)).child(psd_chart(&bl.mean_spectrum, 300.0)),
        )
    } else {
        psd_inner.child(
            div().h(px(120.0)).flex().items_center().justify_center()
                .child(div().text_xs().text_color(muted)
                    .child("Re-record baseline to see PSD")),
        )
    };

    // Topo map section
    let topo_section = div().flex().flex_col().w(px(210.0)).flex_shrink_0().gap_1()
        .child(
            div().text_xs().font_weight(FontWeight::SEMIBOLD)
                .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                .child(format!("SCALP MAP  {}", BAND_NAMES[selected_band])),
        )
        .child(
            div().w(px(210.0)).h(px(180.0)).child(topo_map(&bl.mean_band_powers, selected_band)),
        )
        .child(
            div().flex().items_center().gap_2()
                .child(div().w(px(8.0)).h(px(8.0)).rounded_sm()
                    .bg(gpui::hsla(0.67, 0.85, 0.40, 1.0)))
                .child(div().text_xs().text_color(muted).child("low"))
                .child(div().text_xs().text_color(muted).child("→"))
                .child(div().w(px(8.0)).h(px(8.0)).rounded_sm()
                    .bg(gpui::hsla(0.0, 0.85, 0.55, 1.0)))
                .child(div().text_xs().text_color(muted).child("high")),
        );

    let bottom_row = div().flex().gap_4()
        .pt_2()
        .border_t_1()
        .border_color(gpui::hsla(0.0, 0.0, 0.18, 1.0))
        .child(psd_inner)
        .child(topo_section);

    div()
        .flex()
        .flex_col()
        .gap_4()
        .pt_2()
        .border_t_1()
        .border_color(gpui::hsla(0.0, 0.0, 0.18, 1.0))
        .child(
            div().flex().gap_4()
                .child(left_panel)
                .child(right_panel),
        )
        .child(bottom_row)
}

/// Render an oscilloscope-style waveform trace using gpui canvas with stroked paths.
/// Draws vertical bars at the detected autocorrelation period interval.
fn waveform_canvas(data: &[f32], sample_rate: f32) -> impl IntoElement {
    let data = data.to_vec();
    let bands = decompose_into_bands(&data, sample_rate);

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();
            if data.is_empty() || w < 2.0 || h < 2.0 {
                return WaveformPrepaint {
                    bounds,
                    points: Vec::new(),
                    band_traces: Vec::new(),
                    period_xs: Vec::new(),
                    time_marker_xs: Vec::new(),
                    flat_segments: Vec::new(),
                };
            }

            let display_samples = (w as usize).min(data.len());
            let (offset, period) = autocorrelate_analysis(&data, display_samples);

            // Find range for normalization
            let slice = &data[offset..(offset + display_samples).min(data.len())];
            let min_val = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = (max_val - min_val).max(1e-10);
            let mid_val = (min_val + max_val) / 2.0;

            let padding = 2.0f32;
            let draw_h = h - padding * 2.0;
            let samples_to_px = w / (display_samples - 1).max(1) as f32;

            let points: Vec<(f32, f32)> = (0..display_samples)
                .map(|i| {
                    let idx = offset + i;
                    let val = data.get(idx).copied().unwrap_or(0.0);
                    let x = ox + i as f32 * samples_to_px;
                    let norm = (val - min_val) / range;
                    let y = oy + padding + draw_h * (1.0 - norm);
                    (x, y)
                })
                .collect();

            // Compute band trace points (centered around mid, same scale as raw)
            let band_traces: Vec<(Vec<(f32, f32)>, f32)> = bands
                .iter()
                .map(|(signal, hue)| {
                    let pts: Vec<(f32, f32)> = (0..display_samples)
                        .map(|i| {
                            let idx = offset + i;
                            let val = signal.get(idx).copied().unwrap_or(0.0);
                            let x = ox + i as f32 * samples_to_px;
                            // Band signal is zero-centered; map relative to midpoint of raw range
                            let norm = (mid_val + val - min_val) / range;
                            let y = oy + padding + draw_h * (1.0 - norm);
                            (x, y)
                        })
                        .collect();
                    (pts, *hue)
                })
                .collect();

            // Compute period marker X positions
            let period_xs = if period > 0 {
                let mut xs = Vec::new();
                let mut sample_pos = period;
                while sample_pos < display_samples {
                    xs.push(ox + sample_pos as f32 * samples_to_px);
                    sample_pos += period;
                }
                xs
            } else {
                Vec::new()
            };

            // Detect flat segments (disconnected signal):
            // 5 consecutive samples spanning <= 3% of range
            let flat_tol = range * 0.03;
            let mut flat = vec![false; display_samples.saturating_sub(1)];
            for i in 4..display_samples {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for j in 0..5 {
                    let v = data.get(offset + i - 4 + j).copied().unwrap_or(0.0);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if (hi - lo) <= flat_tol {
                    for j in 0..4 {
                        flat[i - 4 + j] = true;
                    }
                }
            }
            let flat_segments: Vec<(f32, f32, f32, f32)> = flat
                .iter()
                .enumerate()
                .filter(|(_, f)| **f)
                .map(|(i, _)| (points[i].0, points[i].1, points[i + 1].0, points[i + 1].1))
                .collect();

            // Compute 0.5s time marker X positions
            let samples_per_half_sec = (sample_rate * 0.5) as usize;
            let time_marker_xs = if samples_per_half_sec > 0 {
                let mut xs = Vec::new();
                let mut sample_pos = samples_per_half_sec;
                while sample_pos < display_samples {
                    xs.push(ox + sample_pos as f32 * samples_to_px);
                    sample_pos += samples_per_half_sec;
                }
                xs
            } else {
                Vec::new()
            };

            WaveformPrepaint {
                bounds,
                points,
                band_traces,
                period_xs,
                time_marker_xs,
                flat_segments,
            }
        },
        move |_bounds: Bounds<Pixels>, state: WaveformPrepaint, window: &mut Window, _cx: &mut App| {
            let bounds = state.bounds;

            // Paint background box
            window.paint_quad(gpui::fill(bounds, c_canvas()));
            window.paint_quad(gpui::outline(
                bounds,
                c_border(),
                gpui::BorderStyle::Solid,
            ));

            if state.points.len() < 2 {
                return;
            }

            let h: f32 = bounds.size.height.into();
            let oy: f32 = bounds.origin.y.into();

            // Draw center line
            let mid_y = oy + h / 2.0;
            let mut center_line = PathBuilder::stroke(px(0.5));
            center_line.move_to(point(bounds.origin.x, px(mid_y)));
            center_line.line_to(point(bounds.origin.x + bounds.size.width, px(mid_y)));
            if let Ok(path) = center_line.build() {
                window.paint_path(path, gpui::hsla(0.0, 0.0, 0.2, 1.0));
            }

            // Draw period marker vertical bars
            for &x in &state.period_xs {
                let mut marker = PathBuilder::stroke(px(0.75));
                marker.move_to(point(px(x), px(oy)));
                marker.line_to(point(px(x), px(oy + h)));
                if let Ok(path) = marker.build() {
                    window.paint_path(path, gpui::hsla(0.6, 0.5, 0.45, 0.5));
                }
            }

            // Draw 0.5s time markers
            for &x in &state.time_marker_xs {
                let mut marker = PathBuilder::stroke(px(1.0));
                marker.move_to(point(px(x), px(oy)));
                marker.line_to(point(px(x), px(oy + h)));
                if let Ok(path) = marker.build() {
                    window.paint_path(path, gpui::hsla(0.0, 0.0, 0.35, 0.6));
                }
            }

            // Draw the raw waveform trace (dimmed)
            let mut builder = PathBuilder::stroke(px(1.0));
            builder.move_to(point(px(state.points[0].0), px(state.points[0].1)));
            for &(x, y) in &state.points[1..] {
                builder.line_to(point(px(x), px(y)));
            }
            if let Ok(path) = builder.build() {
                window.paint_path(path, gpui::hsla(0.0, 0.0, 0.4, 0.5));
            }

            // Draw band-reconstructed traces
            for (pts, hue) in &state.band_traces {
                if pts.len() < 2 {
                    continue;
                }
                let mut builder = PathBuilder::stroke(px(1.5));
                builder.move_to(point(px(pts[0].0), px(pts[0].1)));
                for &(x, y) in &pts[1..] {
                    builder.line_to(point(px(x), px(y)));
                }
                if let Ok(path) = builder.build() {
                    window.paint_path(path, gpui::hsla(*hue, 0.85, 0.55, 0.85));
                }
            }

            // Draw flat (disconnected) segments in red over the traces
            for &(x1, y1, x2, y2) in &state.flat_segments {
                let mut builder = PathBuilder::stroke(px(2.0));
                builder.move_to(point(px(x1), px(y1)));
                builder.line_to(point(px(x2), px(y2)));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, gpui::hsla(0.0, 0.9, 0.5, 1.0));
                }
            }
        },
    )
    .flex_1()
    .h(px(28.0))
    .min_w(px(200.0))
}

/// Compute FFT magnitude spectrum for a channel's data.
/// Returns magnitudes for the positive frequency bins (DC to Nyquist),
/// whitened by multiplying each bin by its frequency index to compensate
/// for the natural 1/f power law of EEG, making the noise floor flat.
fn compute_spectrum(data: &[f32], fft_size: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![0.0; fft_size / 2];
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); fft_size];
    let n = data.len().min(fft_size);
    let start = data.len().saturating_sub(fft_size);
    for i in 0..n {
        // Hann window
        let w = (std::f32::consts::PI * i as f32 / fft_size as f32).sin().powi(2);
        buf[i] = Complex::new(data[start + i] * w, 0.0);
    }

    fft.process(&mut buf);

    // Return magnitude of positive frequencies (skip DC, up to Nyquist),
    // whitened: multiply by bin index to flatten the 1/f EEG power spectrum,
    // then subtract the minimum so the quietest bin sits at zero.
    let spec: Vec<f32> = buf[1..fft_size / 2]
        .iter()
        .map(|c| {
            c.norm() / fft_size as f32
        })
        .collect();
    spec
}

/// Render an FFT spectrum plot for one channel using gpui canvas,
/// showing only the bins within the given band.
fn spectrum_canvas(data: &[f32], ch: usize, band: BrainWaveBand) -> impl IntoElement {
    let full_spectrum = compute_spectrum(data, SPECTRUM_FFT_SIZE);
    let (bin_start, bin_end) = band.bin_range();
    let spectrum: Vec<f32> = full_spectrum
        .get(bin_start..bin_end)
        .unwrap_or(&[])
        .to_vec();
    let band_hue = band.hue();
    let use_channel_hue = band == BrainWaveBand::All;

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();

            if spectrum.is_empty() || w < 2.0 || h < 2.0 {
                return (bounds, Vec::new(), ch, band_hue, use_channel_hue);
            }

            // Log-scale the magnitudes for better visibility
            let log_spec: Vec<f32> = spectrum
                .iter()
                .map(|&m| (1.0 + m * 1000.0).ln())
                .collect();

            let mut sorted_spec = log_spec.clone();
            sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50 = sorted_spec[sorted_spec.len() / 2];
            let max_val = (log_spec.iter().copied().fold(0.0f32, f32::max) - p50).max(0.01);
            let bar_w = w / log_spec.len() as f32;
            let padding = 1.0f32;
            let draw_h = h - padding * 2.0;

            let bars: Vec<(f32, f32, f32, f32)> = log_spec
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let norm = ((val - p50) / max_val).clamp(0.0, 1.0);
                    let bar_h = draw_h * norm;
                    let x = ox + i as f32 * bar_w;
                    let y = oy + padding + draw_h - bar_h;
                    (x, y, bar_w.max(1.0), bar_h)
                })
                .collect();

            (bounds, bars, ch, band_hue, use_channel_hue)
        },
        move |_bounds: Bounds<Pixels>,
              (bounds, bars, ch, band_hue, use_channel_hue): (
                  Bounds<Pixels>,
                  Vec<(f32, f32, f32, f32)>,
                  usize,
                  f32,
                  bool,
              ),
              window: &mut Window,
              _cx: &mut App| {
            // Background
            window.paint_quad(gpui::fill(bounds, c_canvas()));
            window.paint_quad(gpui::outline(
                bounds,
                c_border(),
                gpui::BorderStyle::Solid,
            ));

            let hue = if use_channel_hue {
                (ch as f32 / 64.0) * 0.8
            } else {
                band_hue
            };

            for &(x, y, w, h) in &bars {
                if h < 0.5 {
                    continue;
                }
                let bar_bounds = Bounds {
                    origin: point(px(x), px(y)),
                    size: size(px(w - 0.5), px(h)),
                };
                window.paint_quad(gpui::fill(bar_bounds, gpui::hsla(hue, 0.7, 0.5, 0.85)));
            }
        },
    )
    .flex_1()
    .h(px(48.0))
}

impl MindDaw {
    fn render_pca_view(&mut self, cx: &mut Context<Self>) -> Div {
        let sphere = pca_sphere_canvas(
            self.pca_state.current_point,
            &self.pca_state.trail,
            self.pca_yaw,
            self.pca_pitch,
        );

        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!(
                        "Running PCA: {} samples | {} dims -> 3D",
                        self.pca_state.sample_count, PCA_DIM,
                    )),
            )
            .child(
                div()
                    .cursor(CursorStyle::PointingHand)
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseDownEvent, _window, _cx| {
                            this.pca_dragging = true;
                            this.pca_last_drag_pos = Some(event.position);
                        }),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                        if this.pca_dragging {
                            if let Some(last) = this.pca_last_drag_pos {
                                let dx: f32 = (event.position.x - last.x).into();
                                let dy: f32 = (event.position.y - last.y).into();
                                this.pca_yaw += dx * 0.01;
                                this.pca_pitch = (this.pca_pitch + dy * 0.01)
                                    .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
                                this.pca_last_drag_pos = Some(event.position);
                                cx.notify();
                            }
                        }
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _, _window, _cx| {
                            this.pca_dragging = false;
                            this.pca_last_drag_pos = None;
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _, _window, _cx| {
                            this.pca_dragging = false;
                            this.pca_last_drag_pos = None;
                        }),
                    )
                    .child(sphere),
            )
    }

    /// Render the 8x8 spectrum grid for all 64 channels.
    fn render_spectrum_grid(
        &mut self,
        waveform_data: &[Vec<f32>],
        cx: &mut Context<Self>,
    ) -> Div {
        let cols = 8;
        let rows = 8;
        let selected = self.selected_channel;
        let band = self.spectrum_band;

        // Band toggle buttons
        let bands = [
            BrainWaveBand::All,
            BrainWaveBand::Delta,
            BrainWaveBand::Theta,
            BrainWaveBand::Alpha,
            BrainWaveBand::Beta,
            BrainWaveBand::Gamma,
        ];
        let mut band_bar = div().flex().gap_1().mb_2();
        for b in bands {
            let (lo, hi) = b.freq_range();
            let sublabel = if b == BrainWaveBand::All {
                String::new()
            } else {
                format!(" ({lo:.0}–{hi:.0} Hz)")
            };
            let label = format!("{}{sublabel}", b.label());
            let btn = if band == b {
                Button::new(SharedString::from(format!("band-{}", b.label())))
                    .label(label)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("band-{}", b.label())))
                    .label(label)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.spectrum_band = b;
                        cx.notify();
                    }))
            };
            band_bar = band_bar.child(btn);
        }

        let mut grid = div().flex().flex_col().gap(px(2.0));

        for row in 0..rows {
            let mut row_div = div().flex().gap(px(2.0));
            for col in 0..cols {
                let ch = row * cols + col;
                let data = waveform_data.get(ch).cloned().unwrap_or_default();
                let is_selected = selected == Some(ch);

                let mut cell = div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .cursor_pointer()
                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _window, cx| {
                        this.select_channel(ch, cx);
                    }))
                    .child(
                        div()
                            .text_xs()
                            .text_color(if is_selected {
                                c_accent_t()
                            } else {
                                c_muted()
                            })
                            .child(format!("Ch{ch}")),
                    )
                    .child(spectrum_canvas(&data, ch, band));

                if is_selected {
                    cell = cell
                        .rounded(px(3.0))
                        .border_1()
                        .border_color(c_accent());
                }

                row_div = row_div.child(cell);
            }
            grid = grid.child(row_div);
        }

        div().flex().flex_col().child(band_bar).child(grid)
    }

    fn render_word_read_view(&mut self, cx: &mut Context<Self>) -> Div {
        use word_read::TrainingPhase;

        let phase = self.word_read_state.phase;
        let is_streaming = matches!(self.cog_state, CogState::Streaming);

        // Training area
        let training_box = div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .p_4()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().border)
            .min_h(px(160.0));

        let training_area = match phase {
            TrainingPhase::Idle => {
                let btn = if is_streaming {
                    Button::new("start-training")
                        .primary()
                        .label("Start Training")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.word_read_state.start_training();
                            cx.notify();
                        }))
                } else {
                    Button::new("start-training")
                        .label("Start Training")
                        .disabled(true)
                };

                training_box
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::BOLD)
                            .text_color(cx.theme().foreground)
                            .child("Word Training"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(cx.theme().muted_foreground)
                            .mt_2()
                            .child("Focus on each word as it appears"),
                    )
                    .child(div().mt_4().child(btn))
            }

            TrainingPhase::ShowingWord => {
                let word = self
                    .word_read_state
                    .current_word()
                    .unwrap_or("")
                    .to_string();
                let progress = self.word_read_state.progress();
                let trained = self.word_read_state.words_trained;
                let idx = self.word_read_state.current_word_idx;
                let loop_num = trained / 20;

                let stop_btn = Button::new("stop-training")
                    .danger()
                    .label("Stop")
                    .on_click(cx.listener(|this, _, _window, cx| {
                        this.word_read_state.phase = word_read::TrainingPhase::Idle;
                        this.word_read_state.word_shown_at = None;
                        cx.notify();
                    }));

                training_box
                    .child(
                        div()
                            .text_3xl()
                            .font_weight(FontWeight::EXTRA_BOLD)
                            .text_color(gpui::hsla(0.58, 0.8, 0.7, 1.0))
                            .child(word),
                    )
                    .child(
                        div()
                            .mt_4()
                            .w_full()
                            .max_w(px(400.0))
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child(format!("word {}/{} — loop {}", idx + 1, 20, loop_num + 1)),
                            )
                            .child(
                                div()
                                    .h(px(6.0))
                                    .w_full()
                                    .rounded(px(3.0))
                                    .bg(gpui::hsla(0.0, 0.0, 0.15, 1.0))
                                    .child(
                                        div()
                                            .h_full()
                                            .rounded(px(3.0))
                                            .bg(gpui_component::green_500())
                                            .w(px(400.0 * progress)),
                                    ),
                            ),
                    )
                    .child(div().mt_3().child(stop_btn))
            }

        };

        // Prediction bar (always visible)
        let predictions = &self.word_read_state.top_predictions;
        let mut pred_row = div().flex().gap_4().items_end();

        for (i, (word, score)) in predictions.iter().enumerate() {
            let brightness = 0.9 - i as f32 * 0.12;
            let font_size = if i == 0 { px(20.0) } else { px(14.0) };
            pred_row = pred_row.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .child(
                        div()
                            .text_size(font_size)
                            .font_weight(if i == 0 {
                                FontWeight::BOLD
                            } else {
                                FontWeight::NORMAL
                            })
                            .text_color(gpui::hsla(0.58, 0.7, brightness, 1.0))
                            .child(word.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!("{score:.2}")),
                    ),
            );
        }

        let prediction_bar = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_4()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().muted_foreground)
                    .child("Mind Reading — Top 5 Predictions"),
            )
            .child(pred_row);

        div()
            .flex()
            .flex_col()
            .gap_4()
            .child(training_area)
            .child(prediction_bar)
    }

    // ── Tonnetz / Orbifold ──────────────────────────────────────────────────

    fn play_tonnetz_chord(&mut self) {
        // Enforce chord type restriction from the active profile.
        // Only filter triads (3-note chords) — dyad type labels are different.
        if let Some(pi) = self.sc_active_profile {
            let allowed = &self.sc_profiles[pi].allowed_chord_types;
            if !allowed.is_empty() {
                if let Some(chord) = self.tonnetz_state.current_chord() {
                    if chord.n() >= 3 && !allowed.contains(&chord.type_label()) {
                        let idx = self.tonnetz_state.current_chord_idx;
                        let mut best_d = f32::INFINITY;
                        let mut best_i = idx;
                        for e in &self.tonnetz_state.edges {
                            let other = if e.from == idx {
                                e.to
                            } else if e.to == idx {
                                e.from
                            } else {
                                continue;
                            };
                            let oc = &self.tonnetz_state.nodes[other].chord;
                            if allowed.contains(&oc.type_label()) && e.distance < best_d {
                                best_d = e.distance;
                                best_i = other;
                            }
                        }
                        if best_i != idx {
                            self.tonnetz_state.current_chord_idx = best_i;
                            self.tonnetz_state.position = [
                                self.tonnetz_state.nodes[best_i].ox,
                                self.tonnetz_state.nodes[best_i].oy,
                                self.tonnetz_state.nodes[best_i].oz,
                            ];
                        }
                    }
                }
            }
        }

        if let Some(chord) = self.tonnetz_state.current_chord() {
            let midi_notes = tonnetz::chord_to_midi_notes(chord);

            // SuperCollider (preferred if connected)
            if let Some(ref h) = self.sc_handle {
                let has_sequencer = self
                    .sc_active_profile
                    .and_then(|i| self.sc_profiles.get(i))
                    .is_some_and(|p| p.bpm > 0.0 && !p.rhythm_pattern.is_empty());

                if has_sequencer {
                    // Sequencer is running — always update its chord (even if "muted")
                    h.update_chord(midi_notes.clone());
                    return;
                }
            }
        }

        // Non-sequencer path: respect mute
        if self.tonnetz_muted {
            return;
        }
        if let Some(chord) = self.tonnetz_state.current_chord() {
            let midi_notes = tonnetz::chord_to_midi_notes(chord);

            if let Some(ref h) = self.sc_handle {
                let random_pan = self
                    .sc_active_profile
                    .and_then(|i| self.sc_profiles.get(i))
                    .map(|p| p.random_pan)
                    .unwrap_or(false);
                h.play_chord(
                    midi_notes.clone(),
                    self.sc_voice,
                    self.sc_params.clone(),
                    random_pan,
                );
                return;
            }

            // Fallback: built-in soundboard
            self.sb_ensure_engine();
            if let Some(ref h) = self.soundboard_handle {
                for &midi in &midi_notes {
                    let _ = h.cmd_tx.try_send(soundboard::SbCommand::PlayNote {
                        midi,
                        waveform: soundboard::SbWaveform::Sine,
                        instrument: soundboard::SbInstrument::Piano,
                        chord: soundboard::SbChord::Single,
                        volume: 0.3 / midi_notes.len() as f32,
                    });
                }
            }
        }
    }

    fn ensure_sc(&mut self) {
        if self.sc_handle.is_none() {
            self.sc_handle = Some(sc::spawn_sc_worker());
        }
    }

    fn render_tonnetz_view(&mut self, cx: &mut Context<Self>) -> Div {
        let state = &self.tonnetz_state;
        let orbifold = state.orbifold;
        let current_idx = state.current_chord_idx;
        let current_chord_label = state
            .current_chord()
            .map(|c| format!("{} ({})", c.label(), c.type_label()))
            .unwrap_or_default();
        let trail_len = state.chord_trail.len();
        let nav_vel = state.nav_velocity;

        // ── Orbifold selector ────────────────────────────────────────────────
        let orbifold_types = [
            tonnetz::OrbifoldType::Dyads,
            tonnetz::OrbifoldType::Triads,
        ];
        let mut orb_row = div().flex().gap_1();
        for orb in orbifold_types {
            let btn = if orbifold == orb {
                Button::new(SharedString::from(format!("orb-{:?}", orb)))
                    .label(orb.label())
                    .primary()
            } else {
                Button::new(SharedString::from(format!("orb-{:?}", orb)))
                    .label(orb.label())
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.tonnetz_state.set_orbifold(orb);
                        cx.notify();
                    }))
            };
            orb_row = orb_row.child(btn);
        }

        // ── Mute toggle ────────────────────────────────────────────────────
        let mute_btn = if self.tonnetz_muted {
            Button::new("orb-mute")
                .label("Unmute")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.tonnetz_muted = false;
                    this.play_tonnetz_chord();
                    cx.notify();
                }))
        } else {
            Button::new("orb-mute")
                .label("Mute")
                .danger()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.tonnetz_muted = true;
                    cx.notify();
                }))
        };

        let orb_label = match orbifold {
            tonnetz::OrbifoldType::Dyads => "T\u{00B2}/S\u{2082}",
            tonnetz::OrbifoldType::Triads => "T\u{00B3}/S\u{2083}",
        };

        // ── Status bar ──────────────────────────────────────────────────────
        let status = div()
            .flex()
            .items_center()
            .gap_4()
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::BOLD)
                    .text_color(gpui::hsla(0.58, 0.8, 0.7, 1.0))
                    .child(SharedString::from(format!(
                        "{}  \u{2014}  {}",
                        orb_label, current_chord_label
                    ))),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!(
                        "trail: {} | nav: [{:.2}, {:.2}]",
                        trail_len, nav_vel[0], nav_vel[1]
                    )),
            );

        // ── Canvas data ─────────────────────────────────────────────────────

        // Get allowed chord types from active profile (empty = all allowed)
        let allowed_types: Vec<&str> = self
            .sc_active_profile
            .and_then(|i| self.sc_profiles.get(i))
            .map(|p| p.allowed_chord_types.clone())
            .unwrap_or_default();

        // Node data: (ox, oy, oz, hue_idx, is_current, is_allowed)
        let node_data: Vec<(f32, f32, f32, u8, bool, bool)> = state
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let allowed = allowed_types.is_empty()
                    || n.chord.n() < 3  // don't filter dyads
                    || allowed_types.contains(&n.chord.type_label());
                (n.ox, n.oy, n.oz, n.chord.hue_index(), i == current_idx, allowed)
            })
            .collect();

        let edges: Vec<(usize, usize, f32)> = state
            .edges
            .iter()
            .map(|e| (e.from, e.to, e.distance))
            .collect();

        let trail: Vec<usize> = state.chord_trail.iter().copied().collect();
        let is_dyads = orbifold == tonnetz::OrbifoldType::Dyads;
        let yaw = state.yaw;
        let pitch_angle = state.pitch;
        let zoom = state.zoom;

        // Shared layout params for dyad click-to-play (left, top, side of the square)
        let dyad_layout = std::rc::Rc::new(std::cell::Cell::new((0.0_f32, 0.0_f32, 1.0_f32)));
        let layout_writer = dyad_layout.clone();

        // Clone node_data for the click handler (canvas closure will move the original)
        let node_data_for_click = node_data.clone();

        let orbifold_canvas = canvas(
            move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| bounds,
            move |_bounds: Bounds<Pixels>,
                  bounds: Bounds<Pixels>,
                  window: &mut Window,
                  _cx: &mut App| {
                let w: f32 = bounds.size.width.into();
                let h: f32 = bounds.size.height.into();
                let bx: f32 = bounds.origin.x.into();
                let by: f32 = bounds.origin.y.into();

                window.paint_quad(gpui::fill(bounds, c_canvas()));

                let hues: [f32; 6] = [0.58, 0.75, 0.0, 0.15, 0.45, 0.5];
                let margin = 50.0f32;

                if is_dyads {
                    // ── T²/S₂: Möbius strip [0,6]×[0,12] as square ──────
                    let side = (w - 2.0 * margin).min(h - 2.0 * margin);
                    let cx0 = bx + w / 2.0;
                    let cy0 = by + h / 2.0;
                    let left = cx0 - side / 2.0;
                    let top = cy0 - side / 2.0;

                    // Export layout params for click-to-play handler
                    layout_writer.set((left, top, side));

                    let to_screen = |ox: f32, oy: f32| -> (f32, f32) {
                        let sx = left + (ox / 6.0) * side;
                        let sy = top + side - (oy / 12.0) * side;
                        (sx, sy)
                    };

                    let domain = Bounds {
                        origin: point(px(left), px(top)),
                        size: size(px(side), px(side)),
                    };
                    window.paint_quad(gpui::fill(domain, gpui::hsla(0.6, 0.12, 0.09, 1.0)));
                    window.paint_quad(gpui::outline(
                        domain,
                        gpui::hsla(0.6, 0.4, 0.4, 0.6),
                        gpui::BorderStyle::Solid,
                    ));

                    // Grid lines
                    for iv in 0..=12 {
                        let (_, sy) = to_screen(0.0, iv as f32);
                        let alpha = if iv == 0 || iv == 6 || iv == 12 { 0.5 } else { 0.15 };
                        let stroke = if iv == 0 || iv == 6 || iv == 12 { 1.0 } else { 0.5 };
                        let mut gb = PathBuilder::stroke(px(stroke));
                        gb.move_to(point(px(left), px(sy)));
                        gb.line_to(point(px(left + side), px(sy)));
                        if let Ok(path) = gb.build() {
                            window.paint_path(path, gpui::hsla(0.0, 0.0, 0.3, alpha));
                        }
                        let mut tb = PathBuilder::stroke(px(1.0));
                        tb.move_to(point(px(left - 4.0), px(sy)));
                        tb.line_to(point(px(left), px(sy)));
                        if let Ok(path) = tb.build() {
                            window.paint_path(path, gpui::hsla(0.0, 0.0, 0.4, 0.8));
                        }
                    }
                    for t in 0..=6 {
                        let (sx, _) = to_screen(t as f32, 0.0);
                        let mut gb = PathBuilder::stroke(px(0.5));
                        gb.move_to(point(px(sx), px(top)));
                        gb.line_to(point(px(sx), px(top + side)));
                        if let Ok(path) = gb.build() {
                            window.paint_path(path, gpui::hsla(0.0, 0.0, 0.3, 0.15));
                        }
                    }

                    // Möbius gluing arrows
                    for i in 0..6 {
                        let frac = (i as f32 + 0.5) / 6.0;
                        let (_, sy) = to_screen(0.0, frac * 12.0);
                        let ax = left - 1.0;
                        let mut b = PathBuilder::stroke(px(2.0));
                        b.move_to(point(px(ax), px(sy + 8.0)));
                        b.line_to(point(px(ax), px(sy - 8.0)));
                        b.move_to(point(px(ax - 3.0), px(sy - 5.0)));
                        b.line_to(point(px(ax), px(sy - 8.0)));
                        b.line_to(point(px(ax + 3.0), px(sy - 5.0)));
                        if let Ok(path) = b.build() {
                            window.paint_path(path, gpui::hsla(0.08, 0.9, 0.6, 0.6));
                        }
                        let ax_r = left + side + 1.0;
                        let mut b2 = PathBuilder::stroke(px(2.0));
                        b2.move_to(point(px(ax_r), px(sy - 8.0)));
                        b2.line_to(point(px(ax_r), px(sy + 8.0)));
                        b2.move_to(point(px(ax_r - 3.0), px(sy + 5.0)));
                        b2.line_to(point(px(ax_r), px(sy + 8.0)));
                        b2.line_to(point(px(ax_r + 3.0), px(sy + 5.0)));
                        if let Ok(path) = b2.build() {
                            window.paint_path(path, gpui::hsla(0.08, 0.9, 0.6, 0.6));
                        }
                    }

                    // Edges, trail, nodes (2D)
                    for &(from, to, dist) in &edges {
                        if let (Some(&(ox1, oy1, _, _, c1, _)), Some(&(ox2, oy2, _, _, c2, _))) =
                            (node_data.get(from), node_data.get(to))
                        {
                            let (x1, y1) = to_screen(ox1, oy1);
                            let (x2, y2) = to_screen(ox2, oy2);
                            let alpha = if c1 || c2 { 0.5 } else {
                                (0.08 + 0.15 * (1.0 - dist / 3.0).max(0.0)).min(0.25)
                            };
                            let hue = if c1 || c2 { 0.33 } else { 0.58 };
                            let mut builder = PathBuilder::stroke(px(if c1 || c2 { 1.5 } else { 0.5 }));
                            builder.move_to(point(px(x1), px(y1)));
                            builder.line_to(point(px(x2), px(y2)));
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, gpui::hsla(hue, 0.5, 0.5, alpha));
                            }
                        }
                    }
                    for pair in trail.windows(2) {
                        if let (Some(&(ox1, oy1, _, _, _, _)), Some(&(ox2, oy2, _, _, _, _))) =
                            (node_data.get(pair[0]), node_data.get(pair[1]))
                        {
                            let (x1, y1) = to_screen(ox1, oy1);
                            let (x2, y2) = to_screen(ox2, oy2);
                            let mut builder = PathBuilder::stroke(px(1.5));
                            builder.move_to(point(px(x1), px(y1)));
                            builder.line_to(point(px(x2), px(y2)));
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, gpui::hsla(0.08, 0.9, 0.6, 0.5));
                            }
                        }
                    }
                    for &(ox, oy, _, hue_idx, is_current, is_allowed) in &node_data {
                        let (x, y) = to_screen(ox, oy);
                        let sz = if is_current { 16.0 } else if is_allowed { 8.0 } else { 5.0 };
                        let nb = Bounds {
                            origin: point(px(x - sz / 2.0), px(y - sz / 2.0)),
                            size: size(px(sz), px(sz)),
                        };
                        if is_current {
                            let gs = 26.0;
                            let glow = Bounds {
                                origin: point(px(x - gs / 2.0), px(y - gs / 2.0)),
                                size: size(px(gs), px(gs)),
                            };
                            window.paint_quad(gpui::fill(glow, gpui::hsla(0.33, 0.9, 0.6, 0.25)));
                            window.paint_quad(gpui::fill(nb, gpui::hsla(0.33, 0.9, 0.7, 1.0)));
                        } else if is_allowed {
                            let hue = hues[hue_idx as usize % hues.len()];
                            window.paint_quad(gpui::fill(nb, gpui::hsla(hue, 0.6, 0.5, 0.7)));
                        } else {
                            // Blocked: dim, desaturated, smaller
                            window.paint_quad(gpui::fill(nb, gpui::hsla(0.0, 0.0, 0.2, 0.3)));
                        }
                    }
                } else {
                    // ── T³/S₃: 3D triangular prism gluing diagram ────────
                    // The fundamental domain is a triangular prism:
                    //   - Transposition axis (length, period 4) along one direction
                    //   - Triangular cross-section from the interval simplex
                    // The two triangular end-faces are identified with a 120°
                    // rotation (cyclic permutation of voices).

                    // Use full 3D coords: ox = transposition [0,4),
                    // oy = barycentric y, oz = barycentric z
                    // Compute bounding box for normalization, padded so the
                    // prism wireframe visually encloses all chord nodes.
                    let (mut xmn, mut xmx) = (f32::INFINITY, f32::NEG_INFINITY);
                    let (mut ymn, mut ymx) = (f32::INFINITY, f32::NEG_INFINITY);
                    let (mut zmn, mut zmx) = (f32::INFINITY, f32::NEG_INFINITY);
                    for &(ox, oy, oz, _, _, _) in &node_data {
                        xmn = xmn.min(ox); xmx = xmx.max(ox);
                        ymn = ymn.min(oy); ymx = ymx.max(oy);
                        zmn = zmn.min(oz); zmx = zmx.max(oz);
                    }
                    // Pad so the prism extends well beyond the data
                    let ypad = (ymx - ymn).max(0.01) * 0.80;
                    let zpad = (zmx - zmn).max(0.01) * 0.80;
                    let xpad = (xmx - xmn).max(0.01) * 0.40;
                    xmn -= xpad; xmx += xpad;
                    ymn -= ypad; ymx += ypad;
                    zmn -= zpad; zmx += zpad;
                    let xr = (xmx - xmn).max(0.01);
                    let yr = (ymx - ymn).max(0.01);
                    let zr = (zmx - zmn).max(0.01);

                    let cx3 = bx + w / 2.0;
                    let cy3 = by + h / 2.0;
                    let r = (w.min(h) / 2.0 - margin) * 0.8 * zoom;

                    let project = |ox: f32, oy: f32, oz: f32| -> (f32, f32, f32) {
                        // Normalize to [-1, 1] with aspect: make transposition
                        // axis longer than the cross-section
                        let nx = (ox - xmn) / xr * 2.0 - 1.0;
                        let ny = (oy - ymn) / yr * 2.0 - 1.0;
                        let nz = (oz - zmn) / zr * 2.0 - 1.0;
                        let rotated = rotate_x(rotate_y([nx, ny, nz], yaw), pitch_angle);
                        (cx3 + rotated[0] * r, cy3 - rotated[1] * r, rotated[2])
                    };

                    // ── Prism wireframe (the gluing diagram) ─────────────
                    // Two triangular faces at x=0 and x=4 (normalized to
                    // x=-1 and x=+1). The triangle vertices are the three
                    // "pure" interval types: (12,0,0), (0,12,0), (0,0,12)
                    // in barycentric coords, but we use the actual data
                    // extent for the triangle corners.
                    // Equilateral triangle vertices in the yz-plane
                    let tri_verts = [
                        (0.0f32, 1.0f32),     // top
                        (-0.866, -0.5),        // bottom-left
                        (0.866, -0.5),         // bottom-right
                    ];
                    // Map triangle verts to data space
                    let tri_3d: Vec<[(f32, f32, f32); 2]> = tri_verts.iter().map(|&(ty, tz)| {
                        let y = ymn + (ty * 0.5 + 0.5) * yr;
                        let z = zmn + (tz * 0.5 + 0.5) * zr;
                        [(xmn, y, z), (xmx, y, z)]
                    }).collect();

                    // Draw the 3 longitudinal edges of the prism
                    for edge in &tri_3d {
                        let (x1, y1, _) = project(edge[0].0, edge[0].1, edge[0].2);
                        let (x2, y2, _) = project(edge[1].0, edge[1].1, edge[1].2);
                        let mut pb = PathBuilder::stroke(px(1.5));
                        pb.move_to(point(px(x1), px(y1)));
                        pb.line_to(point(px(x2), px(y2)));
                        if let Ok(path) = pb.build() {
                            window.paint_path(path, gpui::hsla(0.0, 0.0, 0.4, 0.5));
                        }
                    }

                    // Draw the two triangular end-faces
                    for face_x in [xmn, xmx] {
                        let verts: Vec<(f32, f32)> = tri_verts.iter().map(|&(ty, tz)| {
                            let y = ymn + (ty * 0.5 + 0.5) * yr;
                            let z = zmn + (tz * 0.5 + 0.5) * zr;
                            let (sx, sy, _) = project(face_x, y, z);
                            (sx, sy)
                        }).collect();
                        for i in 0..3 {
                            let j = (i + 1) % 3;
                            let mut pb = PathBuilder::stroke(px(1.5));
                            pb.move_to(point(px(verts[i].0), px(verts[i].1)));
                            pb.line_to(point(px(verts[j].0), px(verts[j].1)));
                            if let Ok(path) = pb.build() {
                                let hue = if face_x == xmn { 0.55 } else { 0.55 };
                                window.paint_path(path, gpui::hsla(hue, 0.5, 0.5, 0.6));
                            }
                        }
                    }

                    // ── 120° twist gluing arrows on triangular faces ─────
                    // Three color-coded arrows on each face showing the
                    // cyclic permutation: vertex A→B, B→C, C→A
                    let twist_hues = [0.0f32, 0.33, 0.66]; // red, green, blue
                    for (face_idx, face_x) in [xmn, xmx].iter().enumerate() {
                        let verts: Vec<(f32, f32)> = tri_verts.iter().map(|&(ty, tz)| {
                            let y = ymn + (ty * 0.5 + 0.5) * yr;
                            let z = zmn + (tz * 0.5 + 0.5) * zr;
                            let (sx, sy, _) = project(*face_x, y, z);
                            (sx, sy)
                        }).collect();
                        // Center of triangle
                        let tcx = (verts[0].0 + verts[1].0 + verts[2].0) / 3.0;
                        let tcy = (verts[0].1 + verts[1].1 + verts[2].1) / 3.0;
                        for k in 0..3 {
                            // Arrow from midpoint of edge k toward next vertex
                            // Direction depends on face: face 0 goes clockwise,
                            // face 1 goes counter-clockwise (the twist)
                            let (i, j) = if face_idx == 0 {
                                (k, (k + 1) % 3)
                            } else {
                                ((k + 1) % 3, k)
                            };
                            // Midpoint of edge
                            let mx = (verts[i].0 + verts[j].0) / 2.0;
                            let my = (verts[i].1 + verts[j].1) / 2.0;
                            // Shrink arrow toward center so it's visible
                            let ax = mx * 0.6 + tcx * 0.4;
                            let ay = my * 0.6 + tcy * 0.4;
                            // Arrow direction: toward the "next" vertex
                            let dx = verts[j].0 - verts[i].0;
                            let dy = verts[j].1 - verts[i].1;
                            let len = (dx * dx + dy * dy).sqrt().max(0.01);
                            let ux = dx / len;
                            let uy = dy / len;
                            let arrow_len = len * 0.25;

                            let mut pb = PathBuilder::stroke(px(2.5));
                            pb.move_to(point(px(ax - ux * arrow_len), px(ay - uy * arrow_len)));
                            pb.line_to(point(px(ax + ux * arrow_len), px(ay + uy * arrow_len)));
                            // Arrowhead
                            let hx = -ux * 5.0 + uy * 4.0;
                            let hy = -uy * 5.0 - ux * 4.0;
                            let hx2 = -ux * 5.0 - uy * 4.0;
                            let hy2 = -uy * 5.0 + ux * 4.0;
                            let tip_x = ax + ux * arrow_len;
                            let tip_y = ay + uy * arrow_len;
                            pb.move_to(point(px(tip_x + hx), px(tip_y + hy)));
                            pb.line_to(point(px(tip_x), px(tip_y)));
                            pb.line_to(point(px(tip_x + hx2), px(tip_y + hy2)));
                            if let Ok(path) = pb.build() {
                                window.paint_path(path, gpui::hsla(twist_hues[k], 0.9, 0.6, 0.8));
                            }
                        }
                    }

                    // ── Project and depth-sort chord nodes ───────────────
                    let mut screen: Vec<(usize, f32, f32, f32, u8, bool, bool)> = node_data
                        .iter()
                        .enumerate()
                        .map(|(i, &(ox, oy, oz, hi, ic, ia))| {
                            let (sx, sy, d) = project(ox, oy, oz);
                            (i, sx, sy, d, hi, ic, ia)
                        })
                        .collect();
                    screen.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

                    let mut spos = vec![(0.0f32, 0.0f32); node_data.len()];
                    let mut scur = vec![false; node_data.len()];
                    for &(i, sx, sy, _, _, ic, _) in &screen {
                        spos[i] = (sx, sy);
                        scur[i] = ic;
                    }

                    // Edges
                    for &(from, to, dist) in &edges {
                        let (x1, y1) = spos[from];
                        let (x2, y2) = spos[to];
                        let c1 = scur[from]; let c2 = scur[to];
                        let alpha = if c1 || c2 { 0.45 } else {
                            (0.04 + 0.12 * (1.0 - dist / 3.5).max(0.0)).min(0.18)
                        };
                        let hue = if c1 || c2 { 0.33 } else { 0.58 };
                        let mut builder = PathBuilder::stroke(px(if c1 || c2 { 1.5 } else { 0.4 }));
                        builder.move_to(point(px(x1), px(y1)));
                        builder.line_to(point(px(x2), px(y2)));
                        if let Ok(path) = builder.build() {
                            window.paint_path(path, gpui::hsla(hue, 0.5, 0.5, alpha));
                        }
                    }

                    // Trail
                    for pair in trail.windows(2) {
                        let (x1, y1) = spos[pair[0]];
                        let (x2, y2) = spos[pair[1]];
                        let mut builder = PathBuilder::stroke(px(1.5));
                        builder.move_to(point(px(x1), px(y1)));
                        builder.line_to(point(px(x2), px(y2)));
                        if let Ok(path) = builder.build() {
                            window.paint_path(path, gpui::hsla(0.08, 0.9, 0.6, 0.5));
                        }
                    }

                    // Nodes (depth-sorted, back-to-front)
                    for &(_i, x, y, depth, hue_idx, is_current, is_allowed) in &screen {
                        let ds = 0.7 + 0.3 * (depth + 1.0) / 2.0;
                        let sz = if is_current { 14.0 * ds }
                            else if is_allowed { 7.0 * ds }
                            else { 4.0 * ds };
                        let nb = Bounds {
                            origin: point(px(x - sz / 2.0), px(y - sz / 2.0)),
                            size: size(px(sz), px(sz)),
                        };
                        if is_current {
                            let gs = sz * 1.8;
                            let glow = Bounds {
                                origin: point(px(x - gs / 2.0), px(y - gs / 2.0)),
                                size: size(px(gs), px(gs)),
                            };
                            window.paint_quad(gpui::fill(glow, gpui::hsla(0.33, 0.9, 0.6, 0.2)));
                            window.paint_quad(gpui::fill(nb, gpui::hsla(0.33, 0.9, 0.7, 1.0)));
                        } else if is_allowed {
                            let hue = hues[hue_idx as usize % hues.len()];
                            let a = 0.4 + 0.4 * ds;
                            window.paint_quad(gpui::fill(nb, gpui::hsla(hue, 0.6, 0.5, a)));
                        } else {
                            // Blocked: dim, desaturated
                            let a = 0.15 + 0.1 * ds;
                            window.paint_quad(gpui::fill(nb, gpui::hsla(0.0, 0.0, 0.2, a)));
                        }
                    }
                }
            },
        )
        .w_full()
        .h(px(500.0));

        // For triads: wrap canvas with mouse drag for 3D rotation
        let orbifold_canvas = if !is_dyads {
            div()
                .id("triad-canvas-interact")
                .cursor(CursorStyle::PointingHand)
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, event: &MouseDownEvent, _window, _cx| {
                        this.tonnetz_state.dragging = true;
                        let pos = event.position;
                        this.tonnetz_state.last_drag_pos =
                            Some((pos.x.into(), pos.y.into()));
                    }),
                )
                .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                    if this.tonnetz_state.dragging {
                        if let Some((lx, ly)) = this.tonnetz_state.last_drag_pos {
                            let dx: f32 = f32::from(event.position.x) - lx;
                            let dy: f32 = f32::from(event.position.y) - ly;
                            this.tonnetz_state.yaw += dx * 0.01;
                            this.tonnetz_state.pitch =
                                (this.tonnetz_state.pitch + dy * 0.01)
                                    .clamp(
                                        -std::f32::consts::FRAC_PI_2,
                                        std::f32::consts::FRAC_PI_2,
                                    );
                            this.tonnetz_state.last_drag_pos =
                                Some((event.position.x.into(), event.position.y.into()));
                            cx.notify();
                        }
                    }
                }))
                .on_mouse_up(
                    MouseButton::Left,
                    cx.listener(|this, _, _window, _cx| {
                        this.tonnetz_state.dragging = false;
                        this.tonnetz_state.last_drag_pos = None;
                    }),
                )
                .on_mouse_up_out(
                    MouseButton::Left,
                    cx.listener(|this, _, _window, _cx| {
                        this.tonnetz_state.dragging = false;
                        this.tonnetz_state.last_drag_pos = None;
                    }),
                )
                .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _window, cx| {
                    let dy = match event.delta {
                        gpui::ScrollDelta::Lines(pt) => pt.y,
                        gpui::ScrollDelta::Pixels(pt) => f32::from(pt.y) / 40.0,
                    };
                    this.tonnetz_state.zoom =
                        (this.tonnetz_state.zoom * (1.0 + dy * 0.1)).clamp(0.3, 5.0);
                    cx.notify();
                }))
                .child(orbifold_canvas)
        } else {
            // Dyads: wrap canvas with click-to-navigate handler.
            let layout_reader = dyad_layout.clone();

            div()
                .id("dyad-canvas-click")
                .cursor(CursorStyle::PointingHand)
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, event: &MouseDownEvent, _window, cx| {
                        if !this.tonnetz_manual_nav {
                            return;
                        }
                        let (left, top, side) = layout_reader.get();
                        if side < 1.0 {
                            return;
                        }
                        let mx: f32 = event.position.x.into();
                        let my: f32 = event.position.y.into();

                        // Invert to_screen: sx = left + ox/6 * side, sy = top + side - oy/12 * side
                        let orb_x = (mx - left) / side * 6.0;
                        let orb_y = (1.0 - (my - top) / side) * 12.0;

                        // Bounds check
                        if orb_x < -0.5 || orb_x > 6.5 || orb_y < -0.5 || orb_y > 12.5 {
                            return;
                        }

                        // Find nearest node by orbifold distance
                        let mut best_dist = f32::MAX;
                        let mut best_idx = None;
                        let period = 6.0_f32;
                        for (i, &(ox, oy, _, _, _, _)) in node_data_for_click.iter().enumerate() {
                            let raw_dx = (orb_x - ox).rem_euclid(period);
                            let dx = raw_dx.min(period - raw_dx);
                            let dy = orb_y - oy;
                            let d = (dx * dx + dy * dy).sqrt();
                            if d < best_dist {
                                best_dist = d;
                                best_idx = Some(i);
                            }
                        }

                        // Only snap if click is reasonably close to a node
                        if let Some(target) = best_idx {
                            if best_dist < 1.5 {
                                let old = this.tonnetz_state.current_chord_idx;
                                if target != old {
                                    this.tonnetz_state.chord_trail.push_back(old);
                                    if this.tonnetz_state.chord_trail.len() > 64 {
                                        this.tonnetz_state.chord_trail.pop_front();
                                    }
                                }
                                this.prev_tonnetz_chord_idx = old;
                                this.tonnetz_state.current_chord_idx = target;
                                this.tonnetz_state.position = [
                                    this.tonnetz_state.nodes[target].ox,
                                    this.tonnetz_state.nodes[target].oy,
                                    this.tonnetz_state.nodes[target].oz,
                                ];
                                this.play_tonnetz_chord();
                                cx.notify();
                            }
                        }
                    }),
                )
                .child(orbifold_canvas)
        };

        // ── Neighbor data ──────────────────────────────────────────────────
        let current_edges = self.tonnetz_state.current_edges();
        let mut neighbors: Vec<(usize, String, String, f32)> = current_edges
            .iter()
            .map(|e| {
                let other_idx = if e.from == self.tonnetz_state.current_chord_idx {
                    e.to
                } else {
                    e.from
                };
                let other = &self.tonnetz_state.nodes[other_idx].chord;
                (
                    other_idx,
                    other.label(),
                    other.type_label().to_string(),
                    e.distance,
                )
            })
            .collect();
        neighbors.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

        // ── Manual nav toggle ─────────────────────────────────────────────
        let manual_nav_btn = if self.tonnetz_manual_nav {
            Button::new("orb-manual-nav")
                .label("Manual Nav")
                .primary()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.tonnetz_manual_nav = false;
                    cx.notify();
                }))
        } else {
            Button::new("orb-manual-nav")
                .label("Manual Nav")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.tonnetz_manual_nav = true;
                    cx.notify();
                }))
        };

        // ── Neighbor navigation panel (side panel) ────────────────────────
        let neighbor_panel = if self.tonnetz_manual_nav {
            // Precompute graph node positions for mini-graph canvas
            let graph_w = 220.0_f32;
            let graph_h = 220.0_f32;
            let gcx = graph_w / 2.0;
            let gcy = graph_h / 2.0;
            let grad = graph_w.min(graph_h) * 0.34;
            let n_nb = neighbors.len().max(1);

            // Get allowed types for the mini-graph
            let mg_allowed: Vec<&str> = self
                .sc_active_profile
                .and_then(|pi| self.sc_profiles.get(pi))
                .map(|p| p.allowed_chord_types.clone())
                .unwrap_or_default();

            // (canvas-local x, canvas-local y, node_index, hue, is_allowed)
            let graph_nodes: Vec<(f32, f32, usize, f32, bool)> = neighbors
                .iter()
                .enumerate()
                .map(|(i, (idx, _, _, _))| {
                    let angle = i as f32 * 2.0 * std::f32::consts::PI / n_nb as f32
                        - std::f32::consts::FRAC_PI_2;
                    let chord = &self.tonnetz_state.nodes[*idx].chord;
                    let hue_idx = chord.hue_index();
                    let hue_f = [0.58_f32, 0.75, 0.0, 0.15, 0.45, 0.5][hue_idx as usize % 6];
                    let allowed = mg_allowed.is_empty()
                        || chord.n() < 3
                        || mg_allowed.contains(&chord.type_label());
                    (gcx + grad * angle.cos(), gcy + grad * angle.sin(), *idx, hue_f, allowed)
                })
                .collect();

            // Shared canvas origin for click detection
            let canvas_origin = std::rc::Rc::new(std::cell::Cell::new((0.0_f32, 0.0_f32)));
            let origin_writer = canvas_origin.clone();
            let origin_reader = canvas_origin.clone();
            let nodes_for_paint = graph_nodes.clone();
            let nodes_for_click = graph_nodes.clone();

            // Current chord hue
            let cur_hue_idx = self.tonnetz_state.nodes[current_idx].chord.hue_index();
            let cur_hue = [0.58_f32, 0.75, 0.0, 0.15, 0.45, 0.5][cur_hue_idx as usize % 6];

            // Neighbor short labels for painting
            let nb_short_labels: Vec<String> = neighbors
                .iter()
                .map(|(idx, _, _, _)| self.tonnetz_state.nodes[*idx].chord.short_label())
                .collect();
            let cur_short_label = self.tonnetz_state.current_chord()
                .map(|c| c.short_label())
                .unwrap_or_default();

            let graph_canvas = canvas(
                move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
                    let ox: f32 = bounds.origin.x.into();
                    let oy: f32 = bounds.origin.y.into();
                    origin_writer.set((ox, oy));
                    (bounds, nodes_for_paint.clone(), ox, oy, gcx, gcy, cur_hue)
                },
                move |_bounds,
                      (bounds, nodes, ox, oy, gcx, gcy, cur_hue): (
                    Bounds<Pixels>,
                    Vec<(f32, f32, usize, f32, bool)>,
                    f32, f32, f32, f32, f32,
                ),
                      window: &mut Window,
                      _cx: &mut App| {
                    // Background
                    window.paint_quad(gpui::fill(bounds, c_canvas()));
                    window.paint_quad(gpui::outline(bounds, c_border(), gpui::BorderStyle::Solid));

                    // Draw edges from center to each neighbor (dim blocked)
                    for &(nx, ny, _, hue_f, nb_ok) in &nodes {
                        let edge_alpha = if nb_ok { 0.7 } else { 0.15 };
                        let mut path = PathBuilder::stroke(px(if nb_ok { 1.5 } else { 0.5 }));
                        path.move_to(point(px(ox + gcx), px(oy + gcy)));
                        path.line_to(point(px(ox + nx), px(oy + ny)));
                        if let Ok(p) = path.build() {
                            window.paint_path(p, hsla(hue_f, 0.30, 0.35, edge_alpha));
                        }
                    }

                    // Draw neighbor nodes (dim blocked ones)
                    let node_r = 10.0_f32;
                    for &(nx, ny, _, hue_f, nb_ok) in &nodes {
                        let (nr, sat, lum, alpha) = if nb_ok {
                            (node_r, 0.65, 0.50, 0.9)
                        } else {
                            (6.0, 0.1, 0.25, 0.4) // smaller, dim
                        };
                        let nb = Bounds {
                            origin: point(px(ox + nx - nr), px(oy + ny - nr)),
                            size: size(px(nr * 2.0), px(nr * 2.0)),
                        };
                        window.paint_quad(gpui::fill(nb, hsla(hue_f, sat, lum, alpha)));
                    }

                    // Draw current node (center, larger)
                    let cr = 14.0_f32;
                    let cb = Bounds {
                        origin: point(px(ox + gcx - cr), px(oy + gcy - cr)),
                        size: size(px(cr * 2.0), px(cr * 2.0)),
                    };
                    // Glow
                    let glow_r = cr + 6.0;
                    let gb = Bounds {
                        origin: point(px(ox + gcx - glow_r), px(oy + gcy - glow_r)),
                        size: size(px(glow_r * 2.0), px(glow_r * 2.0)),
                    };
                    window.paint_quad(gpui::fill(gb, hsla(cur_hue, 0.50, 0.50, 0.15)));
                    window.paint_quad(gpui::fill(cb, hsla(cur_hue, 0.75, 0.60, 1.0)));
                },
            )
            .w(px(graph_w))
            .h(px(graph_h));

            // Wrap canvas in a clickable container for graph node hit detection
            let graph_container = div()
                .id("graph-nav")
                .cursor(CursorStyle::PointingHand)
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, event: &MouseDownEvent, _window, cx| {
                        let (ox, oy) = origin_reader.get();
                        let local_x: f32 = event.position.x.into();
                        let local_y: f32 = event.position.y.into();
                        let local_x = local_x - ox;
                        let local_y = local_y - oy;

                        // Find closest neighbor node within hit radius
                        let mut best_dist = f32::MAX;
                        let mut best_idx = None;
                        for &(nx, ny, node_idx, _, _) in &nodes_for_click {
                            let dx = local_x - nx;
                            let dy = local_y - ny;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < 22.0 && dist < best_dist {
                                best_dist = dist;
                                best_idx = Some(node_idx);
                            }
                        }

                        if let Some(target_idx) = best_idx {
                            // Update trail
                            let old_idx = this.tonnetz_state.current_chord_idx;
                            this.tonnetz_state.chord_trail.push_back(old_idx);
                            if this.tonnetz_state.chord_trail.len() > 64 {
                                this.tonnetz_state.chord_trail.pop_front();
                            }
                            this.tonnetz_state.position_trail.push_back(
                                this.tonnetz_state.position,
                            );
                            if this.tonnetz_state.position_trail.len() > 64 {
                                this.tonnetz_state.position_trail.pop_front();
                            }

                            this.prev_tonnetz_chord_idx = old_idx;
                            this.tonnetz_state.current_chord_idx = target_idx;
                            this.tonnetz_state.position = [
                                this.tonnetz_state.nodes[target_idx].ox,
                                this.tonnetz_state.nodes[target_idx].oy,
                                this.tonnetz_state.nodes[target_idx].oz,
                            ];
                            this.play_tonnetz_chord();
                            cx.notify();
                        }
                    }),
                )
                .child(graph_canvas);

            // Build panel: graph + current chord + labeled neighbor list
            let mut panel = div()
                .id("neighbor-panel")
                .flex()
                .flex_col()
                .gap(px(4.0))
                .w(px(230.0))
                .flex_shrink_0()
                .p_2()
                .rounded(px(6.0))
                .bg(c_surface())
                .border_1()
                .border_color(c_border())
                .overflow_y_scroll()
                // Current chord
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(c_muted())
                                        .child("CURRENT"),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(FontWeight::BOLD)
                                        .text_color(c_accent_t())
                                        .child(SharedString::from(cur_short_label)),
                                ),
                        )
                        .child(
                            Button::new("orb-play")
                                .label("\u{266B}")
                                .primary()
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.tonnetz_muted = false;
                                    this.play_tonnetz_chord();
                                    cx.notify();
                                })),
                        ),
                )
                // Mini-graph
                .child(graph_container)
                // Separator + neighbor list header
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(c_muted())
                        .child(SharedString::from(format!(
                            "NEIGHBORS ({})",
                            neighbors.len()
                        ))),
                );

            // Labeled clickable neighbor list (matches graph node colors)
            // Get allowed types for dimming
            let nb_allowed_types: Vec<&str> = self
                .sc_active_profile
                .and_then(|i| self.sc_profiles.get(i))
                .map(|p| p.allowed_chord_types.clone())
                .unwrap_or_default();

            for (i, (idx, label, type_label, dist)) in neighbors.iter().enumerate() {
                let target_idx = *idx;
                let hue_f = graph_nodes.get(i).map(|n| n.3).unwrap_or(0.5);
                let _short = nb_short_labels.get(i).cloned().unwrap_or_default();
                let nb_chord = &self.tonnetz_state.nodes[target_idx].chord;
                let nb_is_allowed = nb_allowed_types.is_empty()
                    || nb_chord.n() < 3
                    || nb_allowed_types.contains(&nb_chord.type_label());

                // Dim blocked neighbors
                let (bg_sat, bg_light, border_sat, dot_sat, dot_light, text_alpha) =
                    if nb_is_allowed {
                        (0.15, 0.12, 0.30, 0.65, 0.50, 1.0)
                    } else {
                        (0.05, 0.08, 0.10, 0.15, 0.25, 0.35)
                    };

                panel = panel.child(
                    div()
                        .id(SharedString::from(format!("nb-{target_idx}")))
                        .flex()
                        .items_center()
                        .gap(px(6.0))
                        .px(px(6.0))
                        .py(px(4.0))
                        .rounded(px(4.0))
                        .bg(hsla(hue_f, bg_sat, bg_light, 1.0))
                        .border_1()
                        .border_color(hsla(hue_f, border_sat, 0.25, 0.5))
                        .cursor(CursorStyle::PointingHand)
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            let old_idx = this.tonnetz_state.current_chord_idx;
                            this.tonnetz_state.chord_trail.push_back(old_idx);
                            if this.tonnetz_state.chord_trail.len() > 64 {
                                this.tonnetz_state.chord_trail.pop_front();
                            }
                            this.prev_tonnetz_chord_idx = old_idx;
                            this.tonnetz_state.current_chord_idx = target_idx;
                            this.tonnetz_state.position = [
                                this.tonnetz_state.nodes[target_idx].ox,
                                this.tonnetz_state.nodes[target_idx].oy,
                                this.tonnetz_state.nodes[target_idx].oz,
                            ];
                            this.play_tonnetz_chord();
                            cx.notify();
                        }))
                        // Colored dot
                        .child(
                            div()
                                .size(px(10.0))
                                .rounded_full()
                                .bg(hsla(hue_f, dot_sat, dot_light, 0.9))
                                .flex_shrink_0(),
                        )
                        // Label
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .flex_1()
                                .min_w_0()
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(hsla(hue_f, 0.70 * text_alpha, 0.72, text_alpha))
                                        .child(SharedString::from(label.clone())),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(hsla(0.0, 0.0, 0.4, text_alpha))
                                        .child(SharedString::from(format!(
                                            "{}{}· d={dist:.1}",
                                            type_label,
                                            if nb_is_allowed { " " } else { " [blocked] " },
                                        ))),
                                ),
                        ),
                );
            }

            Some(panel)
        } else {
            None
        };

        // ── Controls row ──────────────────────────────────────────────────
        let nav_row = div()
            .flex()
            .items_center()
            .gap_2()
            .child(
                Button::new("orb-prev")
                    .label("\u{25C0} Prev")
                    .on_click(cx.listener(|this, _, _window, cx| {
                        let n = this.tonnetz_state.nodes.len();
                        if n > 0 {
                            this.prev_tonnetz_chord_idx =
                                this.tonnetz_state.current_chord_idx;
                            this.tonnetz_state.current_chord_idx =
                                (this.tonnetz_state.current_chord_idx + n - 1) % n;
                            this.play_tonnetz_chord();
                            cx.notify();
                        }
                    })),
            )
            .child(
                Button::new("orb-next")
                    .label("Next \u{25B6}")
                    .on_click(cx.listener(|this, _, _window, cx| {
                        let n = this.tonnetz_state.nodes.len();
                        if n > 0 {
                            this.prev_tonnetz_chord_idx =
                                this.tonnetz_state.current_chord_idx;
                            this.tonnetz_state.current_chord_idx =
                                (this.tonnetz_state.current_chord_idx + 1) % n;
                            this.play_tonnetz_chord();
                            cx.notify();
                        }
                    })),
            )
            .child(mute_btn)
            .child(manual_nav_btn);

        // ── Sound profiles & SC controls ──────────────────────────────────
        let sc_connected = self.sc_handle.is_some();
        let active_profile = self.sc_active_profile;
        let profile_names: Vec<(usize, &'static str, &'static str)> = self
            .sc_profiles
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.name, p.description))
            .collect();

        let mut voice_row = div()
            .flex()
            .items_center()
            .gap_2()
            .child(
                if sc_connected {
                    Button::new("sc-toggle")
                        .label("SC: On")
                        .primary()
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.sc_handle = None;
                            cx.notify();
                        }))
                } else {
                    Button::new("sc-toggle")
                        .label("Connect SC")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.ensure_sc();
                            cx.notify();
                        }))
                },
            );

        // Profile buttons
        for (idx, name, _desc) in &profile_names {
            let i = *idx;
            let is_active = active_profile == Some(i);
            let btn = if is_active {
                Button::new(SharedString::from(format!("prof-{i}")))
                    .label(*name)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("prof-{i}")))
                    .label(*name)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.sc_active_profile = Some(i);
                        let profile = this.sc_profiles[i].clone();
                        this.sc_voice = profile.voice;
                        this.sc_params = profile.params.clone();
                        this.ensure_sc();
                        // Start sequencer if profile has rhythm
                        if let Some(ref h) = this.sc_handle {
                            if profile.bpm > 0.0 && !profile.rhythm_pattern.is_empty() {
                                let midi = this.tonnetz_state.current_chord()
                                    .map(|c| crate::tonnetz::chord_to_midi_notes(c))
                                    .unwrap_or_default();
                                h.start_sequencer(profile, midi);
                            } else {
                                h.stop_sequencer();
                                h.set_reverb(
                                    profile.reverb_mix,
                                    profile.reverb_room,
                                    profile.reverb_damp,
                                );
                                this.play_tonnetz_chord();
                            }
                        }
                        cx.notify();
                    }))
            };
            voice_row = voice_row.child(btn);
        }

        // "None" button to deactivate profile
        voice_row = voice_row.child(
            if active_profile.is_none() {
                Button::new("prof-none").label("Custom").primary()
            } else {
                Button::new("prof-none")
                    .label("Custom")
                    .on_click(cx.listener(|this, _, _window, cx| {
                        this.sc_active_profile = None;
                        if let Some(ref h) = this.sc_handle {
                            h.stop_sequencer();
                        }
                        cx.notify();
                    }))
            },
        );

        // ── Final layout ──────────────────────────────────────────────────
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(c_muted())
                                    .child("ORBIFOLD"),
                            )
                            .child(orb_row),
                    )
                    .child(status),
            )
            .child(nav_row)
            .child(voice_row)
            .child(
                div()
                    .flex()
                    .gap_2()
                    .flex_1()
                    .min_h_0()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .child(orbifold_canvas),
                    )
                    .children(neighbor_panel),
            )
    }

    // ── Calibration ─────────────────────────────────────────────────────────

    fn render_calibration_view(&mut self, cx: &mut Context<Self>) -> Div {
        let step = self.calibration_state.step;
        let progress = self.calibration_state.progress();
        let has_profile = self.calibration_state.profile.is_some();
        let profile_name = self.calibration_state.profile.as_ref()
            .map(|p| p.user_name.clone())
            .unwrap_or_default();

        // Control state display
        let ctl = &self.control_state;
        let conf_c = ctl.confidence_continuous;
        let motion_x = ctl.motion_x;
        let motion_y = ctl.motion_y;
        let tension = ctl.tension;
        let stability = ctl.stability;
        let freeze = ctl.freeze;

        // Channel quality
        let diag = &self.calibration_state.channel_diag;
        let good_count = self.calibration_state.good_channel_count();
        let n_channels = diag.len();
        let warnings = self.calibration_state.warnings.clone();

        // Available profiles
        let profiles = self.calibration_state.available_profiles.clone();

        let mut col = div()
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .flex_1();

        // ── Header ───────────────────────────────────────────────────────
        col = col.child(
            div()
                .text_lg()
                .font_weight(FontWeight::BOLD)
                .text_color(cx.theme().foreground)
                .child("EEG Calibration & Control"),
        );

        // ── Step status ──────────────────────────────────────────────────
        col = col.child(
            div()
                .flex()
                .gap_2()
                .items_center()
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("Step: {}", step.label())),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child(step.instruction()),
                ),
        );

        // ── Progress bar ─────────────────────────────────────────────────
        if step != calibration::CalibrationStep::Idle
            && step != calibration::CalibrationStep::Complete
        {
            let bar_width = 300.0;
            col = col.child(
                div()
                    .w(px(bar_width))
                    .h(px(8.0))
                    .rounded(px(4.0))
                    .bg(cx.theme().muted)
                    .child(
                        div()
                            .h_full()
                            .rounded(px(4.0))
                            .bg(cx.theme().accent)
                            .w(px(bar_width * progress)),
                    ),
            );
        }

        // ── Action buttons ───────────────────────────────────────────────
        let mut btn_row = div().flex().gap_2();

        if step == calibration::CalibrationStep::Idle {
            // Profile name input
            btn_row = btn_row.child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_sm()
                            .text_color(c_muted())
                            .child("Profile name:"),
                    )
                    .child(
                        div()
                            .w(px(150.0))
                            .child(
                                Input::new(&self.profile_name_input)
                                    .small()
                                    .cleanable(true),
                            ),
                    ),
            );
            btn_row = btn_row.child(
                Button::new("calib-start")
                    .label("Start Calibration")
                    .primary()
                    .on_click(cx.listener(|this, _, _window, cx| {
                        // Read profile name from input
                        let name = this.profile_name_input.read(cx).value().to_string();
                        let name = if name.trim().is_empty() {
                            "default".to_string()
                        } else {
                            name.trim().to_string()
                        };
                        this.calibration_state.user_name = name.clone();
                        this.calibration_state.start();
                        this.session_log = Some(session_log::SessionLog::new(&name));
                        cx.notify();
                    })),
            );
            // Quick refresh for returning users with an existing profile
            if has_profile {
                btn_row = btn_row.child(
                    Button::new("calib-refresh")
                        .label("Quick Refresh")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.calibration_state.start_refresh();
                            let name = if this.calibration_state.user_name.is_empty() {
                                "default"
                            } else {
                                &this.calibration_state.user_name
                            };
                            this.session_log = Some(session_log::SessionLog::new(name));
                            cx.notify();
                        })),
                );
            }
        }

        if step == calibration::CalibrationStep::Complete || has_profile {
            btn_row = btn_row.child(
                div()
                    .text_sm()
                    .text_color(cx.theme().accent)
                    .child(format!("Profile: {}", profile_name)),
            );
        }

        col = col.child(btn_row);

        // ── Action training UI ──────────────────────────────────────────
        if step == calibration::CalibrationStep::ActionTraining {
            let at = &self.calibration_state.action_training;
            let is_running = at.running;
            let cue = at.current_cue().to_string();
            let cue_progress = at.cue_progress();
            let can_train = at.can_train();
            let model_trained = at.model_trained;
            let total_trials = at.total_trials;

            // Trial counts per action
            let counts: Vec<(String, usize)> = at.trial_counts.iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();

            let mut at_col = div().flex().flex_col().gap_3();

            if !is_running {
                // Action selection: show buttons to pick actions, then "Begin"
                let mut action_row = div().flex().flex_wrap().gap_2();
                for &action in calibration::TRAINABLE_ACTIONS {
                    let a = action.to_string();
                    action_row = action_row.child(
                        Button::new(SharedString::from(format!("at-sel-{a}")))
                            .label(SharedString::from(a.replace('_', " ")))
                            .on_click(cx.listener(move |this, _, _window, cx| {
                                let sel = &mut this.calibration_state.action_training.selected_actions;
                                let action_str = a.clone();
                                if sel.contains(&action_str) {
                                    sel.retain(|s| s != &action_str);
                                } else {
                                    sel.push(action_str);
                                }
                                cx.notify();
                            })),
                    );
                }

                at_col = at_col
                    .child(
                        div()
                            .text_sm()
                            .text_color(c_text())
                            .child("Select actions to train, then press Begin:"),
                    )
                    .child(action_row);

                // Show selected actions
                let selected = self.calibration_state.action_training.selected_actions.clone();
                if !selected.is_empty() {
                    at_col = at_col.child(
                        div()
                            .text_sm()
                            .text_color(c_accent_t())
                            .child(SharedString::from(format!(
                                "Selected: {}",
                                selected.join(", ")
                            ))),
                    );
                }

                at_col = at_col.child(
                    div()
                        .flex()
                        .gap_2()
                        .child(
                            Button::new("at-begin")
                                .label("Begin Training")
                                .primary()
                                .disabled(selected.len() < 2)
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    let actions = this
                                        .calibration_state
                                        .action_training
                                        .selected_actions
                                        .clone();
                                    this.calibration_state.start_action_training(actions);
                                    cx.notify();
                                })),
                        )
                        .child(
                            Button::new("at-skip")
                                .label("Skip")
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.calibration_state.skip_action_training();
                                    cx.notify();
                                })),
                        ),
                );
            } else {
                // Running: show cue display, countdown, trial counts
                let cue_color = if cue == "rest" {
                    c_green()
                } else {
                    c_accent_t()
                };
                at_col = at_col.child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap_2()
                        .py_4()
                        // Large cue label
                        .child(
                            div()
                                .text_3xl()
                                .font_weight(FontWeight::EXTRA_BOLD)
                                .text_color(cue_color)
                                .child(SharedString::from(cue.replace('_', " ").to_uppercase())),
                        )
                        // Countdown bar
                        .child(
                            div()
                                .w(px(300.0))
                                .h(px(8.0))
                                .rounded(px(4.0))
                                .bg(c_border())
                                .child(
                                    div()
                                        .h_full()
                                        .rounded(px(4.0))
                                        .bg(cue_color)
                                        .w(px(300.0 * cue_progress)),
                                ),
                        )
                        // Trial count
                        .child(
                            div()
                                .text_sm()
                                .text_color(c_muted())
                                .child(SharedString::from(format!(
                                    "{total_trials} trials recorded"
                                ))),
                        ),
                );

                // Per-class trial counts
                let mut count_row = div().flex().flex_wrap().gap_2();
                for (label, count) in &counts {
                    let min_needed = crate::recorder::fbcsp::MIN_EPOCHS;
                    let color = if *count >= min_needed {
                        c_green()
                    } else {
                        c_muted()
                    };
                    count_row = count_row.child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .px_2()
                            .py_1()
                            .rounded(px(4.0))
                            .bg(c_surface())
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(color)
                                    .child(SharedString::from(format!(
                                        "{}: {}/{}",
                                        label.replace('_', " "),
                                        count,
                                        min_needed
                                    ))),
                            ),
                    );
                }
                at_col = at_col.child(count_row);

                // Control buttons
                let mut ctrl_row = div().flex().gap_2();
                if can_train {
                    ctrl_row = ctrl_row.child(
                        Button::new("at-train")
                            .label("Train FBCSP")
                            .primary()
                            .on_click(cx.listener(|this, _, _window, cx| {
                                // Clone trials for background training
                                let trials = this
                                    .calibration_state
                                    .action_training
                                    .trials
                                    .clone();
                                eprintln!(
                                    "[calibration] training FBCSP on {} trials (background)...",
                                    trials.len()
                                );

                                // Run training on a background thread
                                cx.spawn(async move |this, cx| {
                                    let model = smol::unblock(move || {
                                        recorder::fbcsp::FbcspModel::train(&trials, 300.0)
                                    })
                                    .await;

                                    this.update(cx, |this, cx| {
                                        match model {
                                            Some(m) => {
                                                eprintln!(
                                                    "[calibration] FBCSP trained: {} classes, {} features",
                                                    m.labels.len(),
                                                    m.n_features()
                                                );
                                                this.rec.fbcsp_model = Some(m.clone());
                                                this.calibration_state.action_training.trained_model =
                                                    Some(m);
                                                this.calibration_state.action_training.model_trained =
                                                    true;
                                            }
                                            None => {
                                                eprintln!(
                                                    "[calibration] FBCSP training failed"
                                                );
                                            }
                                        }
                                        cx.notify();
                                    })
                                    .ok();
                                })
                                .detach();
                                cx.notify();
                            })),
                    );
                }
                if model_trained {
                    ctrl_row = ctrl_row.child(
                        div()
                            .text_sm()
                            .text_color(c_green())
                            .child("Model trained"),
                    );
                }
                ctrl_row = ctrl_row.child(
                    Button::new("at-finish")
                        .label("Finish & Continue")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            // Copy trained model to recorder
                            if let Some(ref m) =
                                this.calibration_state.action_training.trained_model
                            {
                                this.rec.fbcsp_model = Some(m.clone());
                            }
                            this.calibration_state.finish_action_training();
                            cx.notify();
                        })),
                );

                at_col = at_col.child(ctrl_row);
            }

            col = col.child(at_col);
        }

        // ── Load existing profiles ───────────────────────────────────────
        if !profiles.is_empty() && step == calibration::CalibrationStep::Idle {
            let mut profile_row = div()
                .flex()
                .flex_wrap()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child("Load profile:"),
                );

            for name in profiles {
                let name_clone = name.clone();
                profile_row = profile_row.child(
                    Button::new(SharedString::from(format!("load-{}", &name)))
                        .label(SharedString::from(name))
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.calibration_state.load_profile(&name_clone);
                            // Load saved FBCSP model if present
                            if let Some(ref profile) = this.calibration_state.profile {
                                if let Some(ref model) = profile.trained_model {
                                    this.rec.fbcsp_model = Some(model.clone());
                                    eprintln!("[profile] loaded FBCSP model: {} classes", model.labels.len());
                                }
                            }
                            // Start session logging with loaded profile
                            this.session_log = Some(session_log::SessionLog::new(&name_clone));
                            if let Some(ref mut log) = this.session_log {
                                log.log_calibration(&name_clone);
                            }
                            cx.notify();
                        })),
                );
            }
            col = col.child(profile_row);
        }

        // ── Channel quality display (only during active calibration) ─────
        let is_active_step = step != calibration::CalibrationStep::Idle
            && step != calibration::CalibrationStep::Complete;
        if !is_active_step {
            // On Idle/Complete, just show a compact summary
            if has_profile {
                col = col.child(
                    div()
                        .text_sm()
                        .text_color(c_green())
                        .child(format!(
                            "Profile loaded: {} ({} channels usable)",
                            profile_name, good_count
                        )),
                );
            }
            return col;
        }
        col = col.child(
            div()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(cx.theme().foreground)
                .child(format!("Channels: {}/{} usable", good_count, n_channels)),
        );

        // Per-channel quality grid (colored rectangles)
        {
            let diag_data: Vec<(f32, bool)> = self.calibration_state.channel_diag
                .iter().map(|d| (d.quality, d.flat)).collect();
            let rows = ((n_channels + 15) / 16) as f32;
            let ch_canvas = canvas(
                move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| { bounds },
                move |_actual: Bounds<Pixels>,
                      bounds: Bounds<Pixels>,
                      window: &mut Window,
                      _cx: &mut App| {
                    let cell = 10.0f32;
                    let gap = 2.0f32;
                    let ox: f32 = bounds.origin.x.into();
                    let oy: f32 = bounds.origin.y.into();
                    for (i, &(quality, flat)) in diag_data.iter().enumerate() {
                        let col_i = i % 16;
                        let row_i = i / 16;
                        let x = ox + col_i as f32 * (cell + gap);
                        let y = oy + row_i as f32 * (cell + gap);
                        let color = if flat {
                            gpui::hsla(0.0, 0.0, 0.15, 1.0)
                        } else if quality > 0.6 {
                            gpui::hsla(0.33, 0.8, 0.45, 1.0)
                        } else if quality > 0.3 {
                            gpui::hsla(0.13, 0.8, 0.5, 1.0)
                        } else {
                            gpui::hsla(0.0, 0.75, 0.45, 1.0)
                        };
                        window.paint_quad(gpui::fill(
                            Bounds::new(
                                point(px(x), px(y)),
                                size(px(cell), px(cell)),
                            ),
                            color,
                        ));
                    }
                },
            )
            .w(px(16.0 * 12.0))
            .h(px(rows * 12.0));
            col = col.child(ch_canvas);
        }

        // Warnings
        for w in &warnings {
            col = col.child(
                div()
                    .text_sm()
                    .text_color(gpui_component::yellow_500())
                    .child(w.clone()),
            );
        }

        // ── Control state visualization ─────────────────────────────────
        let mx = motion_x;
        let my = motion_y;
        let cc = conf_c;
        let tn = tension;
        let st = stability;
        let fz = freeze;
        let motion_canvas = canvas(
            move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
                bounds
            },
            move |_actual_bounds: Bounds<Pixels>,
                  bounds: Bounds<Pixels>,
                  window: &mut Window,
                  _cx: &mut App| {
                // Background
                window.paint_quad(gpui::fill(bounds, c_canvas()));
                let border_color = if fz {
                    gpui::hsla(0.55, 0.8, 0.5, 0.8) // cyan border when frozen
                } else {
                    gpui::hsla(0.0, 0.0, 0.2, 1.0)
                };
                window.paint_quad(gpui::outline(
                    bounds, border_color, gpui::BorderStyle::Solid,
                ));

                let w: f32 = bounds.size.width.into();
                let h: f32 = bounds.size.height.into();
                let ox: f32 = bounds.origin.x.into();
                let oy: f32 = bounds.origin.y.into();
                let cx_f = ox + w * 0.5;
                let cy_f = oy + h * 0.5;
                let radius = w.min(h) * 0.4;

                // Crosshairs
                let cross_color = gpui::hsla(0.0, 0.0, 0.3, 0.5);
                window.paint_quad(gpui::fill(
                    Bounds::new(
                        point(px(cx_f - 0.5), px(oy)),
                        size(px(1.0), px(h)),
                    ),
                    cross_color,
                ));
                window.paint_quad(gpui::fill(
                    Bounds::new(
                        point(px(ox), px(cy_f - 0.5)),
                        size(px(w), px(1.0)),
                    ),
                    cross_color,
                ));

                // Cursor dot — size and color reflect confidence
                let dot_x = cx_f + mx * radius;
                let dot_y = cy_f - my * radius;
                let dot_r = 3.0 + cc * 7.0;
                let dot_hue = if cc > 0.6 { 0.33 } else if cc > 0.3 { 0.15 } else { 0.0 };
                let dot_color = gpui::hsla(dot_hue, 0.8, 0.55, 1.0);
                window.paint_quad(gpui::fill(
                    Bounds::new(
                        point(px(dot_x - dot_r), px(dot_y - dot_r)),
                        size(px(dot_r * 2.0), px(dot_r * 2.0)),
                    ),
                    dot_color,
                ));

                // Tension bar (bottom)
                let bar_h = 4.0;
                let bar_y = oy + h - bar_h - 2.0;
                window.paint_quad(gpui::fill(
                    Bounds::new(
                        point(px(ox + 2.0), px(bar_y)),
                        size(px((w - 4.0) * tn), px(bar_h)),
                    ),
                    gpui::hsla(0.0, 0.7, 0.5, 0.8),
                ));

                // Stability bar (right side, vertical)
                let sbar_w = 4.0;
                let sbar_x = ox + w - sbar_w - 2.0;
                let sbar_h = (h - 4.0) * st;
                window.paint_quad(gpui::fill(
                    Bounds::new(
                        point(px(sbar_x), px(oy + h - 2.0 - sbar_h)),
                        size(px(sbar_w), px(sbar_h)),
                    ),
                    gpui::hsla(0.58, 0.7, 0.5, 0.8),
                ));
            },
        )
        .w(px(160.0))
        .h(px(160.0));

        let mut legend = div()
            .flex()
            .flex_col()
            .gap_1()
            .text_xs()
            .text_color(cx.theme().muted_foreground)
            .child("X: relaxation (α/β)")
            .child("Y: arousal (θ)")
            .child("Bottom: tension")
            .child("Right: stability")
            .child(format!("Confidence {:.0}%", conf_c * 100.0));
        if freeze {
            legend = legend.child(
                div()
                    .text_xs()
                    .font_weight(FontWeight::BOLD)
                    .text_color(gpui::hsla(0.55, 0.8, 0.6, 1.0))
                    .child("FROZEN"),
            );
        }

        col = col.child(
            div()
                .flex()
                .gap_4()
                .mt_2()
                .child(motion_canvas)
                .child(legend),
        );

        // ── Live Detection ────────────────────────────────────────────────
        col = col.child(
            div()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(cx.theme().foreground)
                .mt_2()
                .child("Live Detection"),
        );

        // Hardcoded artifact detectors (blink / jaw clench)
        {
            let blink_color = if self.detecting_blink {
                gpui::hsla(0.33, 0.9, 0.5, 1.0)
            } else {
                gpui::hsla(0.0, 0.0, 0.3, 1.0)
            };
            let jaw_color = if self.detecting_jaw_clench {
                gpui::hsla(0.08, 0.9, 0.55, 1.0)
            } else {
                gpui::hsla(0.0, 0.0, 0.3, 1.0)
            };

            let mut indicators = div().flex().flex_wrap().gap_3();

            // Hardcoded detectors
            indicators = indicators
                .child(
                    div().flex().gap_1().items_center()
                        .child(div().w(px(10.0)).h(px(10.0)).rounded(px(5.0)).bg(blink_color))
                        .child(div().text_sm().text_color(cx.theme().muted_foreground).child("Blink")),
                )
                .child(
                    div().flex().gap_1().items_center()
                        .child(div().w(px(10.0)).h(px(10.0)).rounded(px(5.0)).bg(jaw_color))
                        .child(div().text_sm().text_color(cx.theme().muted_foreground).child("Jaw Clench")),
                );

            // Classifier-detected actions (all trained stimuli)
            if let Some(ref pred) = self.rec.last_prediction {
                let mut sorted: Vec<(&String, &f32)> = pred.similarities.iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                for &(label, &sim) in &sorted {
                    // Skip entries that duplicate the hardcoded detectors
                    if label.contains("blink") || label == "jaw_clench" {
                        continue;
                    }
                    let is_top = *label == pred.predicted_label && pred.confidence > 0.3;
                    let hue = stimulus_hue(label);
                    let dot_color = if is_top {
                        gpui::hsla(hue, 0.9, 0.55, 1.0)
                    } else {
                        gpui::hsla(0.0, 0.0, 0.3, 1.0)
                    };
                    let text_color = if is_top {
                        gpui::hsla(hue, 0.9, 0.75, 1.0)
                    } else {
                        cx.theme().muted_foreground
                    };
                    indicators = indicators.child(
                        div().flex().gap_1().items_center()
                            .child(div().w(px(10.0)).h(px(10.0)).rounded(px(5.0)).bg(dot_color))
                            .child(
                                div().text_sm().text_color(text_color)
                                    .child(format!("{} {:.0}%", label.replace('_', " "), sim * 100.0)),
                            ),
                    );
                }
            } else if self.rec.classifier.is_some() {
                indicators = indicators.child(
                    div().text_xs().text_color(cx.theme().muted_foreground)
                        .child("Classifier idle — start prediction below"),
                );
            }

            col = col.child(indicators);
        }

        // Band power bars
        {
            let bands = &self.live_band_powers;
            let band_data: [(&str, f32, f32); 5] = [
                ("δ delta",  bands.delta,  0.58),
                ("θ theta",  bands.theta,  0.75),
                ("α alpha",  bands.alpha,  0.33),
                ("β beta",   bands.beta,   0.15),
                ("γ gamma",  bands.gamma,  0.0),
            ];

            let mut bands_col = div().flex().flex_col().gap_1();
            for (label, power, hue) in band_data {
                let bar_frac = power.clamp(0.0, 1.0);
                let bar_color = gpui::hsla(hue, 0.7, 0.5, 0.9);
                bands_col = bands_col.child(
                    div()
                        .flex()
                        .gap_2()
                        .items_center()
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .w(px(60.0))
                                .child(label.to_string()),
                        )
                        .child(
                            div()
                                .w(px(150.0))
                                .h(px(8.0))
                                .rounded(px(4.0))
                                .bg(cx.theme().muted)
                                .child(
                                    div()
                                        .h_full()
                                        .rounded(px(4.0))
                                        .bg(bar_color)
                                        .w(px(150.0 * bar_frac)),
                                ),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(format!("{:.2}", power)),
                        ),
                );
            }
            col = col.child(bands_col);
        }

        // Recent detected events log (includes both hardcoded + classifier events)
        {
            let mut events_col = div()
                .flex()
                .flex_col()
                .gap_0p5()
                .mt_1();

            let now = std::time::Instant::now();
            for (t, name) in self.detected_events.iter().rev().take(8) {
                let ago = now.duration_since(*t).as_secs_f32();
                let alpha = (1.0 - ago / 5.0).clamp(0.1, 1.0);
                events_col = events_col.child(
                    div()
                        .text_xs()
                        .text_color(gpui::hsla(0.0, 0.0, 0.7, alpha))
                        .child(format!("{} ({:.1}s ago)", name, ago)),
                );
            }

            if self.detected_events.is_empty() {
                events_col = events_col.child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child("No events detected yet — try blinking or clenching jaw"),
                );
            }

            col = col.child(events_col);
        }

        // ── Recorder (integrated) ────────────────────────────────────────
        col = col.child(
            div()
                .mt_4()
                .border_t_1()
                .border_color(gpui::hsla(0.0, 0.0, 0.2, 1.0))
                .pt_4()
                .child(self.render_recorder_view(cx)),
        );

        col
    }

    // ── Soundboard ────────────────────────────────────────────────────────────

    fn sb_ensure_engine(&mut self) {
        if self.soundboard_handle.is_none() {
            match soundboard::spawn_soundboard_engine() {
                Ok(h) => self.soundboard_handle = Some(h),
                Err(e) => eprintln!("soundboard engine error: {e}"),
            }
        }
    }

    fn sb_play_note(&mut self) {
        self.sb_ensure_engine();
        if let Some(ref h) = self.soundboard_handle {
            let _ = h.cmd_tx.try_send(soundboard::SbCommand::PlayNote {
                midi: self.sb.root_midi,
                waveform: self.sb.waveform,
                instrument: self.sb.instrument,
                chord: self.sb.chord,
                volume: self.sb.volume,
            });
        }
        // Mark soundboard stimulus in recorder (auto-epoch)
        let label = format!("{}_wave", self.sb.waveform.label().to_lowercase().replace(' ', "_"));
        if let Some(ep) = self.rec_ring_to_epoch(&label) {
            if self.rec.pending_epoch.is_none() {
                self.rec.pending_epoch = Some(ep);
            }
        }
    }

    fn sb_start(&mut self, cx: &mut Context<Self>) {
        self.sb_ensure_engine();
        self.sb.is_playing = true;
        self.sb.current_step = 0;

        // Fire first beat immediately
        self.sb_play_note();
        self.sb.trigger_count += 1;
        self.sb.current_step = 1 % self.sb.n_triggers;
        cx.notify();

        cx.spawn(async |this, cx| {
            loop {
                let (interval_ms, still_playing) = this
                    .update(cx, |this, _cx| {
                        let ms = 60_000 / this.sb.bpm as u64;
                        (ms, this.sb.is_playing)
                    })
                    .unwrap_or((500, false));

                if !still_playing {
                    break;
                }

                smol::Timer::after(std::time::Duration::from_millis(interval_ms)).await;

                let cont = this
                    .update(cx, |this, cx| {
                        if !this.sb.is_playing {
                            return false;
                        }
                        this.sb.trigger_count += 1;
                        this.sb.current_step =
                            (this.sb.current_step + 1) % this.sb.n_triggers;
                        if let Some(ref h) = this.soundboard_handle {
                            let _ = h.cmd_tx.try_send(soundboard::SbCommand::PlayNote {
                                midi: this.sb.root_midi,
                                waveform: this.sb.waveform,
                                instrument: this.sb.instrument,
                                chord: this.sb.chord,
                                volume: this.sb.volume,
                            });
                        }
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);

                if !cont {
                    break;
                }
            }
        })
        .detach();
    }

    // ── Recorder tab UI ───────────────────────────────────────────────────────

    fn render_recorder_view(&mut self, cx: &mut Context<Self>) -> Div {
        let mode = self.rec.mode.clone();
        let active_stim = self.rec.active_stimulus.clone();
        let epoch_count = self.rec.session.epochs.len();
        let has_pending = self.rec.pending_epoch.is_some();
        let has_classifier = self.rec.classifier.is_some();
        let prediction = self.rec.last_prediction.clone();
        let pred_history = self.rec.prediction_history.iter().cloned().collect::<Vec<_>>();
        let session_labels = self.rec.session.labels();
        let thresholds = self.rec.thresholds.clone();

        // ── All stimulus labels (built-in + custom) ───────────────────────────
        let all_stimuli: Vec<String> = BUILT_IN_STIMULI
            .iter()
            .map(|s| s.to_string())
            .chain(self.rec.custom_stimuli.iter().cloned())
            .collect();

        // ── LEFT: Stimulus Library ────────────────────────────────────────────
        let mut stim_list = div()
            .flex()
            .flex_col()
            .gap(px(1.0))
            .w(px(170.0))
            .flex_shrink_0();

        stim_list = stim_list.child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(cx.theme().muted_foreground)
                .mb_2()
                .child("STIMULUS LIBRARY"),
        );

        for (i, stim) in all_stimuli.iter().enumerate() {
            let count = self.rec.session.count_for(stim);
            let is_active = *stim == active_stim;
            let hue = stimulus_hue(stim);
            let stim_clone = stim.clone();
            let shortcut = if i < 9 { format!(" [{}]", i + 1) } else { String::new() };
            let label_text = format!("{}{}", stim.replace('_', " "), shortcut);

            let row = div()
                .flex()
                .items_center()
                .justify_between()
                .p(px(4.0))
                .rounded_sm()
                .cursor_pointer()
                .bg(if is_active {
                    gpui::hsla(hue, 0.4, 0.18, 1.0)
                } else {
                    gpui::hsla(0.0, 0.0, 0.0, 0.0)
                })
                .border_1()
                .border_color(if is_active {
                    gpui::hsla(hue, 0.7, 0.5, 0.8)
                } else {
                    gpui::hsla(0.0, 0.0, 0.0, 0.0)
                })
                .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _window, cx| {
                    this.rec.active_stimulus = stim_clone.clone();
                    // If there are recorded epochs for this stimulus, load the last one into review
                    let last_ep = this.rec.session.epochs.iter()
                        .filter(|e| e.label == stim_clone)
                        .last()
                        .cloned();
                    this.rec.review_epoch = last_ep;
                    cx.notify();
                }))
                .child(
                    div()
                        .text_xs()
                        .text_color(if is_active {
                            gpui::hsla(hue, 0.9, 0.75, 1.0)
                        } else {
                            cx.theme().foreground
                        })
                        .child(label_text),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                        .child(format!("{count}")),
                );
            stim_list = stim_list.child(row);
        }

        // New stimulus input row
        stim_list = stim_list.child(
            div()
                .mt_2()
                .flex()
                .gap_1()
                .child(
                    Input::new(&self.stimulus_input)
                        .flex_1()
                        .small(),
                )
                .child(
                    Button::new("rec-add-stim")
                        .label("+")
                        .on_click(cx.listener(|this, _, window, cx| {
                            let name = this.stimulus_input.read(cx).value().to_string();
                            let name = name.trim().to_string();
                            if !name.is_empty() {
                                let slug = name.to_lowercase().replace(' ', "_");
                                if !this.rec.custom_stimuli.contains(&slug) {
                                    this.rec.custom_stimuli.push(slug.clone());
                                }
                                this.rec.active_stimulus = slug;
                                this.stimulus_input.update(cx, |s, cx| {
                                    s.set_value("", window, cx);
                                });
                                cx.notify();
                            }
                        })),
                ),
        );

        // ── MIDDLE: Epoch Preview + Controls ─────────────────────────────────
        let mut middle = div().flex().flex_col().flex_1().gap_3();

        // Determine what data source we're previewing and compute display duration
        let is_reviewing = self.rec.review_epoch.is_some();
        let preview_sample_count = if is_reviewing {
            self.rec.review_epoch.as_ref().map(|e| e.samples.len()).unwrap_or(0)
        } else if has_pending {
            self.rec.pending_epoch.as_ref().map(|e| e.samples.len()).unwrap_or(0)
        } else {
            self.rec_ring.len()
        };
        let preview_dur_ms = preview_sample_count as f32 / 300.0 * 1000.0;

        // Header row: title + duration tag + optional "← Live" button
        let header_row = {
            let mode_label = if is_reviewing {
                let lbl = self.rec.review_epoch.as_ref().map(|e| e.label.replace('_', " ")).unwrap_or_default();
                format!("REVIEWING: {}", lbl.to_uppercase())
            } else if has_pending {
                "EPOCH PREVIEW — captured".to_string()
            } else {
                "EPOCH PREVIEW — live buffer".to_string()
            };
            let dur_tag = format!("{:.0}ms / {} samples", preview_dur_ms, preview_sample_count);

            let mut row = div()
                .flex()
                .items_center()
                .justify_between()
                .gap_2();

            row = row.child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(if is_reviewing {
                                gpui::hsla(stimulus_hue(&active_stim), 0.9, 0.7, 1.0)
                            } else {
                                cx.theme().muted_foreground
                            })
                            .child(mode_label),
                    )
                    .child(
                        div()
                            .text_xs()
                            .px(px(5.0))
                            .py(px(2.0))
                            .rounded_sm()
                            .bg(gpui::hsla(0.0, 0.0, 0.12, 1.0))
                            .text_color(cx.theme().muted_foreground)
                            .child(dur_tag),
                    ),
            );

            if is_reviewing {
                row = row.child(
                    Button::new("rec-exit-review")
                        .label("← Live")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.rec.review_epoch = None;
                            cx.notify();
                        })),
                );
            }
            row
        };
        middle = middle.child(header_row);

        // Channel labels: anatomical region names, not signal type labels.
        // Ch0 ≈ frontal (near eye/forehead — sensitive to blink artifacts)
        // Ch10 ≈ temporal (near jaw/temple — sensitive to jaw-clench artifacts)
        // Ch20 ≈ central (motor cortex region)
        // Without a confirmed Cognionics HD-72 pin-out these are approximations.
        let preview_channels = [0usize, 10, 20];
        let preview_labels = ["Ch0 — frontal", "Ch10 — temporal", "Ch20 — central"];
        for (i, &ch) in preview_channels.iter().enumerate() {
            let data = if is_reviewing {
                self.rec.review_epoch.as_ref().map(|e| e.channel(ch)).unwrap_or_default()
            } else if let Some(ref ep) = self.rec.pending_epoch {
                ep.channel(ch)
            } else {
                self.rec_ring.iter().map(|f| f[ch]).collect()
            };
            middle = middle.child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .w(px(80.0))
                            .flex_shrink_0()
                            .text_color(cx.theme().muted_foreground)
                            .child(preview_labels[i]),
                    )
                    .child(waveform_canvas(&data, 300.0)),
            );
        }

        // Time axis ruler — tick labels proportional to the preview duration
        {
            // Choose tick interval: 200ms for ≤1s, 500ms for >1s
            let tick_interval_ms: f32 = if preview_dur_ms <= 1050.0 { 200.0 } else { 500.0 };
            let num_ticks = (preview_dur_ms / tick_interval_ms).floor() as usize;
            let mut ruler = div()
                .flex()
                .items_center()
                .ml(px(82.0)) // align with canvas area (label column width + gap)
                .mb(px(2.0));
            // "0ms" at the start
            ruler = ruler.child(
                div()
                    .text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0))
                    .text_xs()
                    .child("0ms"),
            );
            // Spacers + tick labels
            for t in 1..=num_ticks {
                let t_ms = t as f32 * tick_interval_ms;
                let label = if t_ms >= 1000.0 {
                    format!("{}s", t_ms / 1000.0)
                } else {
                    format!("{:.0}ms", t_ms)
                };
                ruler = ruler.child(div().flex_1()); // push tick to proportional position
                ruler = ruler.child(
                    div()
                        .text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0))
                        .text_xs()
                        .child(label),
                );
            }
            // End label showing total duration
            ruler = ruler.child(div().flex_1());
            ruler = ruler.child(
                div()
                    .text_color(gpui::hsla(0.0, 0.0, 0.4, 1.0))
                    .text_xs()
                    .child(if preview_dur_ms >= 1000.0 {
                        format!("{}s", preview_dur_ms / 1000.0)
                    } else {
                        format!("{:.0}ms", preview_dur_ms)
                    }),
            );
            middle = middle.child(ruler);
        }

        // Pending epoch info + accept/reject
        if has_pending {
            if let Some(ref ep) = self.rec.pending_epoch {
                let ep_label = ep.label.clone();
                let ep_samples = ep.samples.len();
                middle = middle.child(
                    div()
                        .mt_2()
                        .p_2()
                        .rounded_md()
                        .border_1()
                        .border_color(gpui::hsla(0.58, 0.7, 0.5, 0.5))
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .text_color(gpui::hsla(0.58, 0.9, 0.7, 1.0))
                                .child(format!(
                                    "Captured: \"{}\" — {} samples ({:.0} ms)",
                                    ep_label,
                                    ep_samples,
                                    ep_samples as f32 / 300.0 * 1000.0
                                )),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(
                                    Button::new("rec-accept")
                                        .primary()
                                        .label("✓ Accept")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.rec_accept_epoch(cx);
                                        })),
                                )
                                .child(
                                    Button::new("rec-reject")
                                        .danger()
                                        .label("✗ Reject")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.rec_reject_epoch(cx);
                                        })),
                                ),
                        ),
                );
            }
        }

        // Record / ARM controls
        let record_btn = Button::new("rec-record")
            .primary()
            .label("◉ Record")
            .on_click(cx.listener(|this, _, _window, cx| {
                this.rec.review_epoch = None; // return to live view on record
                this.rec_capture_epoch(cx);
            }));

        let arm_btn = if mode == RecorderMode::Armed {
            Button::new("rec-arm")
                .danger()
                .label("▣ Armed — click to cancel")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.rec.mode = RecorderMode::Idle;
                    cx.notify();
                }))
        } else {
            Button::new("rec-arm")
                .label("▶ ARM")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.rec.review_epoch = None; // return to live view on arm
                    this.rec.mode = RecorderMode::Armed;
                    cx.notify();
                }))
        };

        middle = middle.child(
            div()
                .mt_2()
                .flex()
                .gap_2()
                .child(record_btn)
                .child(arm_btn),
        );

        // Threshold sliders (ARM mode settings)
        let blink_thresh = thresholds.blink_uv;
        let jaw_thresh = thresholds.jaw_power;
        middle = middle.child(
            div()
                .mt_1()
                .flex()
                .gap_3()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .child(div().text_xs().text_color(cx.theme().muted_foreground).child("Blink µV:"))
                        .child(
                            Button::new("th-blink-dn").label("−").on_click(cx.listener(|this, _, _, cx| {
                                this.rec.thresholds.blink_uv = (this.rec.thresholds.blink_uv - 10.0).max(10.0);
                                cx.notify();
                            })),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().foreground)
                                .w(px(36.0))
                                .child(format!("{:.0}", blink_thresh)),
                        )
                        .child(
                            Button::new("th-blink-up").label("+").on_click(cx.listener(|this, _, _, cx| {
                                this.rec.thresholds.blink_uv = (this.rec.thresholds.blink_uv + 10.0).min(500.0);
                                cx.notify();
                            })),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .child(div().text_xs().text_color(cx.theme().muted_foreground).child("Jaw pwr:"))
                        .child(
                            Button::new("th-jaw-dn").label("−").on_click(cx.listener(|this, _, _, cx| {
                                this.rec.thresholds.jaw_power = (this.rec.thresholds.jaw_power - 5.0).max(5.0);
                                cx.notify();
                            })),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().foreground)
                                .w(px(36.0))
                                .child(format!("{:.0}", jaw_thresh)),
                        )
                        .child(
                            Button::new("th-jaw-up").label("+").on_click(cx.listener(|this, _, _, cx| {
                                this.rec.thresholds.jaw_power = (this.rec.thresholds.jaw_power + 5.0).min(200.0);
                                cx.notify();
                            })),
                        ),
                ),
        );

        // Session stats + save/export
        middle = middle.child(
            div()
                .mt_3()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!("Session: {} epochs", epoch_count)),
                )
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .child(
                            Button::new("rec-save")
                                .label("💾 Save")
                                .on_click(cx.listener(|this, _, _, _cx| {
                                    this.rec_save_session();
                                })),
                        )
                        .child(
                            Button::new("rec-export")
                                .label("📤 CSV")
                                .on_click(cx.listener(|this, _, _, _cx| {
                                    this.rec_export_csv();
                                })),
                        ),
                ),
        );

        // ── RIGHT: Live Classifier ────────────────────────────────────────────
        let mut right = div()
            .flex()
            .flex_col()
            .gap_2()
            .w(px(230.0))
            .flex_shrink_0();

        right = right.child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(cx.theme().muted_foreground)
                .child("LIVE CLASSIFIER"),
        );

        // Start/stop prediction button
        let can_predict = has_classifier
            || self.rec.session.min_class_count() >= MIN_EPOCHS_PER_CLASS;

        let pred_btn = if mode == RecorderMode::Predicting {
            Button::new("rec-pred-stop")
                .danger()
                .label("● Stop Prediction")
                .on_click(cx.listener(|this, _, _, cx| {
                    this.rec.mode = RecorderMode::Idle;
                    this.rec.last_prediction = None;
                    cx.notify();
                }))
        } else if can_predict {
            Button::new("rec-pred-start")
                .primary()
                .label("▶ Start Prediction")
                .on_click(cx.listener(|this, _, _, cx| {
                    // Force retrain before starting (both classifiers)
                    this.rec.classifier = TrainedClassifier::train(&this.rec.session.epochs);
                    this.rec_train_fbcsp();
                    this.rec.mode = RecorderMode::Predicting;
                    cx.notify();
                }))
        } else {
            let min = self.rec.session.min_class_count();
            Button::new("rec-pred-start")
                .label(format!("Need {}/{} min epochs", min, MIN_EPOCHS_PER_CLASS))
                .disabled(true)
        };
        right = right.child(pred_btn);

        // Confidence bars
        if let Some(ref pred) = prediction {
            right = right.child(
                div()
                    .mt_2()
                    .flex()
                    .flex_col()
                    .gap(px(2.0))
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!(
                                "▶ {} ({:.0}%)",
                                pred.predicted_label,
                                pred.confidence * 100.0
                            )),
                    ),
            );

            // Sort classes by similarity descending
            let mut sorted: Vec<(String, f32)> = pred.similarities.iter()
                .map(|(l, &s)| (l.clone(), s))
                .collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (label, sim) in &sorted {
                let hue = stimulus_hue(label);
                let bar_frac = sim.clamp(0.0, 1.0);
                let is_top = *label == pred.predicted_label;
                right = right.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .child(
                            div()
                                .text_xs()
                                .w(px(90.0))
                                .text_color(if is_top {
                                    gpui::hsla(hue, 0.9, 0.75, 1.0)
                                } else {
                                    cx.theme().muted_foreground
                                })
                                .child(label.replace('_', " ")),
                        )
                        .child(
                            // Background track
                            div()
                                .flex_1()
                                .h(px(10.0))
                                .rounded_sm()
                                .bg(gpui::hsla(0.0, 0.0, 0.15, 1.0))
                                .child(
                                    // Filled bar
                                    div()
                                        .h_full()
                                        .rounded_sm()
                                        .bg(gpui::hsla(hue, 0.75, 0.5, 0.9))
                                        .w(relative(bar_frac)),
                                ),
                        )
                        .child(
                            div()
                                .text_xs()
                                .w(px(30.0))
                                .text_color(cx.theme().muted_foreground)
                                .child(format!("{:.2}", sim)),
                        ),
                );
            }

            if pred.is_novel {
                right = right.child(
                    div()
                        .mt_1()
                        .text_xs()
                        .text_color(gpui::hsla(0.1, 0.8, 0.65, 1.0))
                        .child("⚠ Novel / unrecognised signal"),
                );
            }
        } else if mode == RecorderMode::Predicting {
            right = right.child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Waiting for signal…"),
            );
        } else {
            right = right.child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Collect ≥5 epochs per class, then start prediction."),
            );
        }

        // Radar canvas
        if has_classifier || !session_labels.is_empty() {
            let similarities_for_radar: Vec<(String, f32)> = if let Some(ref pred) = prediction {
                session_labels
                    .iter()
                    .map(|l| {
                        let sim = pred.similarities.get(l).copied().unwrap_or(0.0);
                        (l.clone(), sim)
                    })
                    .collect()
            } else {
                session_labels.iter().map(|l| (l.clone(), 0.0)).collect()
            };

            if !similarities_for_radar.is_empty() {
                right = right.child(
                    div()
                        .mt_2()
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child("DEVIATION MAP"),
                        )
                        .child(radar_canvas(&similarities_for_radar)),
                );
            }
        }

        // Prediction history
        if !pred_history.is_empty() {
            right = right.child(
                div()
                    .mt_2()
                    .flex()
                    .flex_col()
                    .gap(px(1.0))
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child("HISTORY"),
                    )
                    .children(pred_history.iter().rev().take(8).map(|p| {
                        let hue = stimulus_hue(&p.predicted_label);
                        div()
                            .text_xs()
                            .text_color(gpui::hsla(hue, 0.8, 0.65, 1.0))
                            .child(format!(
                                "{} {:.0}%",
                                p.predicted_label.replace('_', " "),
                                p.confidence * 100.0
                            ))
                            .into_any_element()
                    })),
            );
        }

        // ── Assemble three-column layout ──────────────────────────────────────
        let baseline_section = self.render_baseline_section(cx);

        div()
            .flex()
            .flex_col()
            .flex_1()
            .p_2()
            .gap_2()
            .child(baseline_section)
            .child(
                div()
                    .flex()
                    .gap_4()
                    .flex_1()
                    .child(stim_list)
                    .child(middle)
                    .child(right),
            )
    }

    // ── Baseline dashboard ────────────────────────────────────────────────────

    fn render_baseline_section(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let has_baseline = self.rec.baseline.is_some();
        let is_recording = self.rec.baseline_rec.is_some();
        let progress = self.rec.baseline_rec.as_ref().map(|r| r.progress()).unwrap_or(0.0);
        let windows_done = self.rec.baseline_rec.as_ref().map(|r| r.windows_done).unwrap_or(0);
        let target = self.rec.baseline_rec.as_ref().map(|r| r.target_windows).unwrap_or(30);
        let normalize = self.rec.normalize_with_baseline;
        let dashboard_open = self.rec.baseline_dashboard_open;
        // Clone baseline data so we can pass it without borrow conflicts.
        let baseline = self.rec.baseline.clone();

        // ── Status strip ─────────────────────────────────────────────────────
        let status_color = if is_recording {
            gpui::hsla(0.17, 0.9, 0.65, 1.0) // amber while recording
        } else if has_baseline {
            gpui::hsla(0.33, 0.8, 0.55, 1.0) // green when done
        } else {
            cx.theme().muted_foreground
        };

        let rejected = self.rec.baseline_rec.as_ref().map(|r| r.windows_rejected).unwrap_or(0);
        let status_text = if is_recording {
            let rej_str = if rejected > 0 { format!(" ({} artifact windows rejected)", rejected) } else { String::new() };
            format!("Recording resting EEG… {}/{}s{}", windows_done, target, rej_str)
        } else if let Some(ref bl) = baseline {
            format!("✓ {}s baseline — IAF {:.1} Hz — FAA {:+.2} ({})",
                bl.duration_s as u32, bl.iaf_hz, bl.faa, bl.faa_label())
        } else {
            "No baseline — record 30 s of resting EEG to unlock normalised classification".to_string()
        };

        // Button row
        let btn_row = div().flex().items_center().gap_2()
            // Record 30s
            .child(if is_recording {
                Button::new("bl-stop")
                    .danger()
                    .label("✗ Stop")
                    .on_click(cx.listener(|this, _, _, cx| {
                        // Finalise whatever was collected
                        if let Some(rec) = this.rec.baseline_rec.take() {
                            this.rec.baseline = rec.finalize();
                        }
                        cx.notify();
                    }))
            } else {
                Button::new("bl-30")
                    .label("Record 30s")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_rec = Some(BaselineRecorder::new(30, 300.0));
                        this.rec.baseline_dashboard_open = false;
                        cx.notify();
                    }))
            })
            // Record 60s (disabled while recording)
            .child(
                Button::new("bl-60")
                    .label("60s")
                    .disabled(is_recording)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_rec = Some(BaselineRecorder::new(60, 300.0));
                        this.rec.baseline_dashboard_open = false;
                        cx.notify();
                    })),
            )
            // Normalise toggle
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child("Normalise:"),
                    )
                    .child(
                        Button::new("bl-norm")
                            .label(if normalize && has_baseline { "● ON" } else { "○ OFF" })
                            .disabled(!has_baseline)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.rec.normalize_with_baseline = !this.rec.normalize_with_baseline;
                                cx.notify();
                            })),
                    ),
            )
            // Dashboard toggle (only when baseline exists)
            .children(has_baseline.then(|| {
                Button::new("bl-dash")
                    .label(if dashboard_open { "▲ Hide" } else { "▼ Dashboard" })
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_dashboard_open = !this.rec.baseline_dashboard_open;
                        cx.notify();
                    }))
            }))
            // Clear
            .children(has_baseline.then(|| {
                Button::new("bl-clear")
                    .label("Clear")
                    .danger()
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline = None;
                        this.rec.baseline_dashboard_open = false;
                        cx.notify();
                    }))
            }));

        let mut section = div()
            .flex()
            .flex_col()
            .gap_2()
            .px_2()
            .py(px(6.0))
            .rounded_md()
            .border_1()
            .border_color(if has_baseline {
                gpui::hsla(0.33, 0.5, 0.3, 0.6)
            } else {
                gpui::hsla(0.0, 0.0, 0.2, 1.0)
            })
            // Header row: label + status + buttons
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(cx.theme().muted_foreground)
                                    .child("BASELINE REFERENCE"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(status_color)
                                    .child(status_text),
                            ),
                    )
                    .child(btn_row),
            );

        // Progress bar while recording
        if is_recording {
            let pct_w = (progress * 400.0) as u32; // approximate px width
            section = section.child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(400.0))
                            .h(px(4.0))
                            .rounded_full()
                            .bg(gpui::hsla(0.0, 0.0, 0.15, 1.0))
                            .child(
                                div()
                                    .h(px(4.0))
                                    .rounded_full()
                                    .bg(gpui::hsla(0.17, 0.9, 0.55, 1.0))
                                    .w(px(pct_w as f32)),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!("{}%", (progress * 100.0) as u32)),
                    ),
            );
        }

        // MNE subprocess status line
        if let Some(ref mne_status) = self.rec.baseline_mne_status {
            let is_running = mne_status.starts_with('⏳');
            section = section.child(
                div()
                    .text_xs()
                    .text_color(if is_running {
                        gpui::hsla(0.15, 0.8, 0.65, 1.0)
                    } else if mne_status.starts_with('✓') {
                        gpui::hsla(0.33, 0.7, 0.55, 1.0)
                    } else {
                        gpui::hsla(0.08, 0.8, 0.65, 1.0)
                    })
                    .child(mne_status.clone()),
            );
        }

        // ── Profile save / load ───────────────────────────────────────────────
        // Save row: only shown when a baseline is loaded
        if has_baseline {
            let save_row = div()
                .flex()
                .items_center()
                .gap_1()
                .child(
                    div().text_xs().text_color(cx.theme().muted_foreground).child("Save as:"),
                )
                .child(
                    Input::new(&self.profile_name_input)
                        .small()
                        .flex_1(),
                )
                .child(
                    Button::new("bl-save-profile")
                        .label("Save + MNE")
                        .on_click(cx.listener(|this, _, window, cx| {
                            let name = this.profile_name_input.read(cx).value().to_string();
                            let name = name.trim().to_string();
                            if name.is_empty() {
                                return;
                            }
                            if let Some(ref bl) = this.rec.baseline {
                                // 1. Save the quick Rust-computed baseline as a fallback
                                match recorder::storage::save_baseline_profile(&name, bl) {
                                    Ok(p) => eprintln!("[profiles] saved rust baseline to {}", p.display()),
                                    Err(e) => eprintln!("[profiles] save error: {e}"),
                                }
                                // 2. Save raw frames for MNE (take from the recorder if still available)
                                let raw_path_ok = if let Some(ref mut rec) = this.rec.baseline_rec {
                                    let frames = rec.take_raw_frames();
                                    match recorder::storage::save_raw_baseline(&name, &frames, rec.sample_rate) {
                                        Ok(_) => true,
                                        Err(e) => { eprintln!("[profiles] raw save error: {e}"); false }
                                    }
                                } else {
                                    false
                                };

                                this.saved_profiles = recorder::storage::list_baseline_profiles();
                                this.profile_name_input.update(cx, |s, cx| s.set_value("", window, cx));

                                // 3. Spawn MNE subprocess if raw data was saved
                                if raw_path_ok {
                                    this.rec.baseline_mne_status = Some("⏳ MNE processing…".to_string());
                                    cx.notify();
                                    let name2 = name.clone();
                                    let name3 = name.clone();
                                    cx.spawn(async move |this, cx| {
                                        let result = smol::unblock(move || {
                                            std::process::Command::new("python3")
                                                .args(["scripts/compute_baseline.py", &name2])
                                                .output()
                                        }).await;
                                        this.update(cx, |this, cx| {
                                            match result {
                                                Ok(out) if out.status.success() => {
                                                    // Reload the MNE-enhanced profile
                                                    match recorder::storage::load_baseline_profile(&name3) {
                                                        Ok(bl) => {
                                                            this.rec.baseline = Some(bl);
                                                            this.rec.baseline_mne_status = Some("✓ MNE processed".to_string());
                                                        }
                                                        Err(e) => {
                                                            this.rec.baseline_mne_status = Some(format!("⚠ reload error: {e}"));
                                                        }
                                                    }
                                                }
                                                Ok(out) => {
                                                    let stderr = String::from_utf8_lossy(&out.stderr);
                                                    this.rec.baseline_mne_status = Some(format!("⚠ MNE error: {}", stderr.lines().last().unwrap_or("unknown")));
                                                }
                                                Err(e) => {
                                                    this.rec.baseline_mne_status = Some(format!("⚠ spawn error: {e}"));
                                                }
                                            }
                                            cx.notify();
                                        }).ok();
                                    }).detach();
                                } else {
                                    this.rec.baseline_mne_status = Some("⚠ raw data unavailable — re-record baseline to enable MNE".to_string());
                                }
                                cx.notify();
                            }
                        })),
                );
            section = section.child(save_row);
        }

        // Load row: list of saved profiles as clickable buttons
        if !self.saved_profiles.is_empty() {
            let profiles = self.saved_profiles.clone();
            let load_row = profiles.iter().fold(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .flex_wrap()
                    .child(
                        div().text_xs().text_color(cx.theme().muted_foreground).child("Load:"),
                    ),
                |row, name| {
                    let n = name.clone();
                    row.child(
                        Button::new(SharedString::from(format!("bl-load-{n}")))
                            .label(SharedString::from(n.clone()))
                            .small()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                match recorder::storage::load_baseline_profile(&n) {
                                    Ok(bl) => {
                                        this.rec.baseline = Some(bl);
                                        this.rec.baseline_dashboard_open = true;
                                        eprintln!("[profiles] loaded '{n}'");
                                    }
                                    Err(e) => eprintln!("[profiles] load error: {e}"),
                                }
                                cx.notify();
                            })),
                    )
                },
            );
            section = section.child(load_row);
        }

        // Expanded dashboard
        if dashboard_open {
            if let Some(ref bl) = baseline {
                let selected = self.rec.baseline_selected_band;

                // Band selector — which band the topo map displays
                let band_sel = div().flex().items_center().gap_1()
                    .child(div().text_xs().text_color(cx.theme().muted_foreground).child("Topo band:"))
                    .child(Button::new("bl-b0").label("δ").on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_selected_band = 0; cx.notify();
                    })))
                    .child(Button::new("bl-b1").label("θ").on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_selected_band = 1; cx.notify();
                    })))
                    .child(Button::new("bl-b2").label("α").on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_selected_band = 2; cx.notify();
                    })))
                    .child(Button::new("bl-b3").label("β").on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_selected_band = 3; cx.notify();
                    })))
                    .child(Button::new("bl-b4").label("γ").on_click(cx.listener(|this, _, _, cx| {
                        this.rec.baseline_selected_band = 4; cx.notify();
                    })))
                    .child(
                        div().text_xs().font_weight(FontWeight::SEMIBOLD)
                            .text_color(gpui::hsla(BAND_HUES[selected], 0.8, 0.65, 1.0))
                            .child(format!("▶ {}", BAND_NAMES[selected])),
                    );

                section = section.child(band_sel);
                section = section.child(baseline_dashboard_expanded(bl, selected, cx));
            }
        }

        section
    }

    fn render_soundboard_view(&mut self, cx: &mut Context<Self>) -> Div {
        let waveform = self.sb.waveform;
        let instrument = self.sb.instrument;
        let root_midi = self.sb.root_midi;
        let chord = self.sb.chord;
        let bpm = self.sb.bpm;
        let n_triggers = self.sb.n_triggers;
        let volume = self.sb.volume;
        let is_playing = self.sb.is_playing;
        let current_step = self.sb.current_step;
        let trigger_count = self.sb.trigger_count;

        // ── Transport ────────────────────────────────────────────────────────
        let play_stop_btn = if is_playing {
            Button::new("sb-stop")
                .label("■ Stop")
                .danger()
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.sb.is_playing = false;
                    cx.notify();
                }))
        } else {
            Button::new("sb-play")
                .primary()
                .label("▶ Play")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.sb_start(cx);
                }))
        };

        let trigger_btn = Button::new("sb-trigger-now")
            .label("▷ Trigger")
            .on_click(cx.listener(|this, _, _window, cx| {
                this.sb_play_note();
                cx.notify();
            }));

        let bpm_ctrl = div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("BPM"),
            )
            .child(Button::new("sb-bpm-dn").label("−").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.bpm = this.sb.bpm.saturating_sub(5).max(20);
                    cx.notify();
                },
            )))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(cx.theme().foreground)
                    .w(px(36.0))
                    .child(format!("{bpm}")),
            )
            .child(Button::new("sb-bpm-up").label("+").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.bpm = (this.sb.bpm + 5).min(240);
                    cx.notify();
                },
            )));

        let n_ctrl = div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("N"),
            )
            .child(Button::new("sb-n-dn").label("−").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.n_triggers = this.sb.n_triggers.saturating_sub(1).max(1);
                    cx.notify();
                },
            )))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(cx.theme().foreground)
                    .w(px(24.0))
                    .child(format!("{n_triggers}")),
            )
            .child(Button::new("sb-n-up").label("+").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.n_triggers = (this.sb.n_triggers + 1).min(16);
                    cx.notify();
                },
            )));

        let vol_ctrl = div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Vol"),
            )
            .child(Button::new("sb-vol-dn").label("−").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.volume = (this.sb.volume - 0.05).max(0.0);
                    cx.notify();
                },
            )))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().foreground)
                    .w(px(38.0))
                    .child(format!("{:.0}%", volume * 100.0)),
            )
            .child(Button::new("sb-vol-up").label("+").on_click(cx.listener(
                |this, _, _window, cx| {
                    this.sb.volume = (this.sb.volume + 0.05).min(1.0);
                    cx.notify();
                },
            )));

        let transport_row = div()
            .flex()
            .items_center()
            .gap_3()
            .child(play_stop_btn)
            .child(trigger_btn)
            .child(bpm_ctrl)
            .child(n_ctrl)
            .child(vol_ctrl);

        // ── Waveform grid (2×2) ──────────────────────────────────────────────
        let all_waves = [
            soundboard::SbWaveform::Sine,
            soundboard::SbWaveform::Sawtooth,
            soundboard::SbWaveform::Triangle,
            soundboard::SbWaveform::Square,
        ];
        let mut wave_row1 = div().flex().gap_2();
        let mut wave_row2 = div().flex().gap_2();
        for (i, &w) in all_waves.iter().enumerate() {
            let label = w.label().to_string();
            let btn = if waveform == w {
                Button::new(SharedString::from(format!("sb-w-{i}")))
                    .label(label)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("sb-w-{i}")))
                    .label(label)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.sb.waveform = w;
                        cx.notify();
                    }))
            };
            if i < 2 {
                wave_row1 = wave_row1.child(btn);
            } else {
                wave_row2 = wave_row2.child(btn);
            }
        }
        let wave_section = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("WAVEFORM"),
            )
            .child(wave_row1)
            .child(wave_row2);

        // ── Instrument grid (2×2) ────────────────────────────────────────────
        let all_insts = [
            soundboard::SbInstrument::Kick,
            soundboard::SbInstrument::Snare,
            soundboard::SbInstrument::Piano,
            soundboard::SbInstrument::Strings,
        ];
        let mut inst_row1 = div().flex().gap_2();
        let mut inst_row2 = div().flex().gap_2();
        for (i, &inst) in all_insts.iter().enumerate() {
            let label = inst.label().to_string();
            let btn = if instrument == inst {
                Button::new(SharedString::from(format!("sb-i-{i}")))
                    .label(label)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("sb-i-{i}")))
                    .label(label)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.sb.instrument = inst;
                        cx.notify();
                    }))
            };
            if i < 2 {
                inst_row1 = inst_row1.child(btn);
            } else {
                inst_row2 = inst_row2.child(btn);
            }
        }
        let inst_section = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("INSTRUMENT"),
            )
            .child(inst_row1)
            .child(inst_row2);

        // ── Root note ────────────────────────────────────────────────────────
        const SB_NOTES: &[(&str, u8)] = &[
            ("C4", 60), ("D4", 62), ("E4", 64), ("F4", 65), ("G4", 67),
            ("A4", 69), ("B4", 71), ("C5", 72), ("D5", 74), ("E5", 76),
        ];
        let mut note_row = div().flex().gap_1();
        for &(name, midi) in SB_NOTES {
            let label = name.to_string();
            let btn = if root_midi == midi {
                Button::new(SharedString::from(format!("sb-note-{midi}")))
                    .label(label)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("sb-note-{midi}")))
                    .label(label)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.sb.root_midi = midi;
                        this.sb_play_note();
                        cx.notify();
                    }))
            };
            note_row = note_row.child(btn);
        }

        // ── Chord ────────────────────────────────────────────────────────────
        let all_chords = [
            soundboard::SbChord::Single,
            soundboard::SbChord::Major,
            soundboard::SbChord::Minor,
            soundboard::SbChord::Dom7,
            soundboard::SbChord::Sus4,
        ];
        let mut chord_row = div().flex().gap_1();
        for (i, &ch) in all_chords.iter().enumerate() {
            let label = ch.label().to_string();
            let btn = if chord == ch {
                Button::new(SharedString::from(format!("sb-ch-{i}")))
                    .label(label)
                    .primary()
            } else {
                Button::new(SharedString::from(format!("sb-ch-{i}")))
                    .label(label)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.sb.chord = ch;
                        cx.notify();
                    }))
            };
            chord_row = chord_row.child(btn);
        }

        // ── Sequence display ─────────────────────────────────────────────────
        let mut seq_row = div().flex().gap_1();
        for step in 0..n_triggers {
            let is_active = is_playing && step == current_step;
            let step_el = div()
                .w(px(28.0))
                .h(px(28.0))
                .flex()
                .items_center()
                .justify_center()
                .text_xs()
                .rounded_sm()
                .border_1()
                .border_color(if is_active {
                    gpui::hsla(0.33, 0.7, 0.5, 1.0)
                } else {
                    cx.theme().border
                })
                .bg(if is_active {
                    gpui::hsla(0.33, 0.7, 0.25, 1.0)
                } else {
                    cx.theme().background
                })
                .text_color(if is_active {
                    gpui::hsla(0.33, 0.9, 0.75, 1.0)
                } else {
                    cx.theme().muted_foreground
                })
                .child(format!("{}", step + 1));
            seq_row = seq_row.child(step_el);
        }

        // ── Assemble ─────────────────────────────────────────────────────────
        div()
            .flex()
            .flex_col()
            .gap_4()
            .child(transport_row)
            .child(div().flex().gap_6().child(wave_section).child(inst_section))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child("ROOT NOTE"),
                    )
                    .child(note_row),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child("CHORD"),
                    )
                    .child(chord_row),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!(
                                "SEQUENCE  ·  {trigger_count} triggers fired"
                            )),
                    )
                    .child(seq_row),
            )
    }
}

fn rotate_y(p: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = angle.sin_cos();
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

fn rotate_x(p: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = angle.sin_cos();
    [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]
}

fn project_ortho(p: [f32; 3], cx: f32, cy: f32, radius: f32) -> (f32, f32, f32) {
    (cx + p[0] * radius, cy - p[1] * radius, p[2])
}

struct PcaPrepaint {
    bounds: Bounds<Pixels>,
    lat_lines: Vec<Vec<(f32, f32, f32)>>,
    lon_lines: Vec<Vec<(f32, f32, f32)>>,
    trail_points: Vec<(f32, f32, f32, f32)>,
    current_point: Option<(f32, f32)>,
}

fn pca_sphere_canvas(
    current_point: [f32; 3],
    trail: &VecDeque<[f32; 3]>,
    yaw: f32,
    pitch: f32,
) -> impl IntoElement {
    let trail: Vec<[f32; 3]> = trail.iter().copied().collect();

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();
            let center_x = ox + w / 2.0;
            let center_y = oy + h / 2.0;
            let radius = (w.min(h) / 2.0) * 0.85;

            let segments = 48;
            let rotate = |p: [f32; 3]| rotate_x(rotate_y(p, yaw), pitch);

            // Generate latitude circles (7)
            let mut lat_lines = Vec::new();
            for lat_i in 1..=7 {
                let phi = std::f32::consts::PI * lat_i as f32 / 8.0;
                let r = phi.sin();
                let y_pos = phi.cos();
                let mut line = Vec::new();
                for seg in 0..=segments {
                    let theta =
                        2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
                    let p = [r * theta.cos(), y_pos, r * theta.sin()];
                    let rotated = rotate(p);
                    let (sx, sy, depth) =
                        project_ortho(rotated, center_x, center_y, radius);
                    line.push((sx, sy, depth));
                }
                lat_lines.push(line);
            }

            // Generate longitude meridians (12)
            let mut lon_lines = Vec::new();
            for lon_i in 0..12 {
                let theta =
                    2.0 * std::f32::consts::PI * lon_i as f32 / 12.0;
                let mut line = Vec::new();
                for seg in 0..=segments {
                    let phi =
                        std::f32::consts::PI * seg as f32 / segments as f32;
                    let p = [
                        phi.sin() * theta.cos(),
                        phi.cos(),
                        phi.sin() * theta.sin(),
                    ];
                    let rotated = rotate(p);
                    let (sx, sy, depth) =
                        project_ortho(rotated, center_x, center_y, radius);
                    line.push((sx, sy, depth));
                }
                lon_lines.push(line);
            }

            // Project trail points
            let trail_len = trail.len();
            let trail_points: Vec<(f32, f32, f32, f32)> = trail
                .iter()
                .enumerate()
                .map(|(i, &pt)| {
                    let rotated = rotate(pt);
                    let (sx, sy, depth) =
                        project_ortho(rotated, center_x, center_y, radius);
                    let age_factor = (i + 1) as f32 / trail_len.max(1) as f32;
                    (sx, sy, depth, age_factor)
                })
                .collect();

            // Project current point
            let rotated = rotate(current_point);
            let (sx, sy, _depth) =
                project_ortho(rotated, center_x, center_y, radius);
            let cp = if current_point[0] != 0.0
                || current_point[1] != 0.0
                || current_point[2] != 0.0
            {
                Some((sx, sy))
            } else {
                None
            };

            PcaPrepaint {
                bounds,
                lat_lines,
                lon_lines,
                trail_points,
                current_point: cp,
            }
        },
        move |_bounds: Bounds<Pixels>,
              state: PcaPrepaint,
              window: &mut Window,
              _cx: &mut App| {
            let bounds = state.bounds;

            // Dark background + outline
            window.paint_quad(gpui::fill(bounds, c_canvas()));
            window.paint_quad(gpui::outline(
                bounds,
                c_border(),
                gpui::BorderStyle::Solid,
            ));

            // Wireframe: latitude circles
            for line in &state.lat_lines {
                for pair in line.windows(2) {
                    let (x1, y1, d1) = pair[0];
                    let (x2, y2, d2) = pair[1];
                    let avg_depth = (d1 + d2) / 2.0;
                    let alpha = 0.15 + 0.15 * (avg_depth + 1.0) / 2.0;
                    let mut builder = PathBuilder::stroke(px(0.5));
                    builder.move_to(point(px(x1), px(y1)));
                    builder.line_to(point(px(x2), px(y2)));
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, gpui::hsla(0.58, 0.2, 0.5, alpha));
                    }
                }
            }

            // Wireframe: longitude meridians
            for line in &state.lon_lines {
                for pair in line.windows(2) {
                    let (x1, y1, d1) = pair[0];
                    let (x2, y2, d2) = pair[1];
                    let avg_depth = (d1 + d2) / 2.0;
                    let alpha = 0.15 + 0.15 * (avg_depth + 1.0) / 2.0;
                    let mut builder = PathBuilder::stroke(px(0.5));
                    builder.move_to(point(px(x1), px(y1)));
                    builder.line_to(point(px(x2), px(y2)));
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, gpui::hsla(0.58, 0.2, 0.5, alpha));
                    }
                }
            }

            // Trail segments + points
            for pair in state.trail_points.windows(2) {
                let (x1, y1, d1, age1) = pair[0];
                let (x2, y2, d2, age2) = pair[1];
                let avg_depth = ((d1 + d2) / 2.0 + 1.0) / 2.0;
                let avg_age = (age1 + age2) / 2.0;
                let alpha = avg_age * (0.3 + 0.7 * avg_depth);
                let mut builder = PathBuilder::stroke(px(1.5));
                builder.move_to(point(px(x1), px(y1)));
                builder.line_to(point(px(x2), px(y2)));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, gpui::hsla(0.33, 0.8, 0.5, alpha));
                }
            }
            for &(sx, sy, depth, age_factor) in &state.trail_points {
                let depth_factor = (depth + 1.0) / 2.0;
                let alpha = age_factor * (0.3 + 0.7 * depth_factor);
                let sz = 3.0;
                let trail_bounds = Bounds {
                    origin: point(px(sx - sz / 2.0), px(sy - sz / 2.0)),
                    size: size(px(sz), px(sz)),
                };
                window.paint_quad(gpui::fill(
                    trail_bounds,
                    gpui::hsla(0.33, 0.8, 0.5, alpha),
                ));
            }

            // Current point
            if let Some((sx, sy)) = state.current_point {
                // Glow halo
                let glow_sz = 14.0;
                let glow_bounds = Bounds {
                    origin: point(px(sx - glow_sz / 2.0), px(sy - glow_sz / 2.0)),
                    size: size(px(glow_sz), px(glow_sz)),
                };
                window.paint_quad(gpui::fill(
                    glow_bounds,
                    gpui::hsla(0.33, 0.9, 0.6, 0.3),
                ));

                // Bright dot
                let dot_sz = 8.0;
                let dot_bounds = Bounds {
                    origin: point(px(sx - dot_sz / 2.0), px(sy - dot_sz / 2.0)),
                    size: size(px(dot_sz), px(dot_sz)),
                };
                window.paint_quad(gpui::fill(
                    dot_bounds,
                    gpui::hsla(0.33, 0.9, 0.7, 1.0),
                ));
            }
        },
    )
    .w_full()
    .h(px(400.0))
}

fn main() {
    let demo_mode = std::env::args().any(|a| a == "--demo");

    Application::new().run(move |cx: &mut App| {
        gpui_component::init(cx);
        gpui_component::theme::Theme::change(gpui_component::theme::ThemeMode::Dark, None, cx);

        cx.open_window(WindowOptions::default(), |window, cx| {
            let stimulus_input = cx.new(|cx| InputState::new(window, cx).placeholder("new stimulus…"));
            let profile_name_input = cx.new(|cx| InputState::new(window, cx).placeholder("profile name…"));
            let view = cx.new(|cx| {
                let mut daw = MindDaw::new(stimulus_input, profile_name_input);
                if demo_mode {
                    daw.cog_demo(cx);
                }
                daw
            });
            cx.new(|cx| Root::new(view, window, cx))
        })
        .unwrap();
    });
}
