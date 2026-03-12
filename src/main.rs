mod audio;
mod cognionics;
mod recorder;
mod soundboard;
mod streams;
mod word_read;

use audio::{AudioCommand, AudioHandle, EegFrame};
use cognionics::{CogCommand, CogHandle, CogState};
use recorder::auto_detect::detect_event;
use recorder::baseline::{BaselineProfile, BaselineRecorder, BAND_HUES, BAND_NAMES, BAND_SYMS, REGION_NAMES};
use recorder::baseline::normalize_features as baseline_normalize;
use recorder::classifier::{predict_features, TrainedClassifier, MIN_EPOCHS_PER_CLASS, RETRAIN_EVERY};
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
    Recorder,
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
const SPECTRUM_SAMPLE_RATE: f32 = 300.0;

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
        let end = (hi_hz / bin_hz).floor() as usize; // inclusive bin index, but we use it as exclusive slice end
        let max_bin = SPECTRUM_FFT_SIZE / 2 - 1; // max output index
        (start.min(max_bin), end.min(max_bin + 1))
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
    sb: SoundboardUiState,

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
            sb: SoundboardUiState::default(),

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
                                    if this.audio_enabled {
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
            }
            cx.notify();
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

                            // Recorder: ARM auto-detect
                            if this.rec.mode == RecorderMode::Armed {
                                if detect_event(&this.rec_ring, &this.rec.thresholds).is_some() {
                                    this.rec_capture_epoch(cx);
                                }
                            }

                            // Recorder: live prediction
                            if this.rec.mode == RecorderMode::Predicting {
                                if let Some(clf) = this.rec.classifier.clone() {
                                    if let Some(ep) = this.rec_ring_to_epoch("live") {
                                        let feat = extract_features(&ep);
                                        let feat = if this.rec.normalize_with_baseline {
                                            if let Some(ref bl) = this.rec.baseline {
                                                baseline_normalize(&feat, bl)
                                            } else { feat }
                                        } else { feat };
                                        let pred = predict_features(&feat, &clf);
                                        if this.rec.prediction_history.len() >= 10 {
                                            this.rec.prediction_history.pop_front();
                                        }
                                        this.rec.prediction_history.push_back(pred.clone());
                                        this.rec.last_prediction = Some(pred);
                                    }
                                }
                            }

                            cx.notify();
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
            .bg(cx.theme().background)
            .p_4()
            .gap_4()
            // Header
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::BOLD)
                            .text_color(cx.theme().foreground)
                            .child("mind-daw — EEG Streams"),
                    )
                    .child(if scanning {
                        Button::new("scan")
                            .label("Scanning...")
                            .disabled(true)
                    } else {
                        Button::new("scan")
                            .primary()
                            .label("Scan LSL")
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.scan(cx);
                            }))
                    }),
            )
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
                            .text_color(cx.theme().muted_foreground)
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
                                                        "Type: {} | Channels: {} | Rate: {:.0} Hz",
                                                        stream.stream_type,
                                                        stream.channel_count,
                                                        stream.sample_rate,
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
        cx: &mut Context<Self>,
    ) -> Option<Div> {
        let meta = self.paired_meta.as_ref()?.clone();
        let ch_count = meta.channel_count as usize;

        let audio_btn = if self.audio_enabled {
            Button::new("lsl-audio-toggle")
                .danger()
                .label("Stop Audio")
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.stop_audio(cx);
                }))
        } else {
            Button::new("lsl-audio-toggle")
                .primary()
                .label("Start Audio")
                .on_click(cx.listener(move |this, _, _window, cx| {
                    this.start_audio(ch_count, cx);
                }))
        };

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
                        .justify_between()
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
                        .child(audio_btn),
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

    fn render_cog_panel(
        &mut self,
        cog_state: &CogState,
        cog_waveform_data: &[Vec<f32>],
        cx: &mut Context<Self>,
    ) -> Div {
        let panel = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().border);

        match cog_state {
            CogState::Disconnected => panel.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(cx.theme().foreground)
                            .child("Cognionics HD-72"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                Button::new("cog-scan")
                                    .primary()
                                    .label("Connect Cognionics")
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

            CogState::Scanning => panel.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(cx.theme().foreground)
                            .child("Cognionics HD-72"),
                    )
                    .child(
                        Button::new("cog-scanning")
                            .label("Scanning...")
                            .disabled(true),
                    ),
            ),

            CogState::Found { id, name } => {
                let device_id = id.clone();
                let label = format!("Connect to {name}");
                panel.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(cx.theme().foreground)
                                .child("Cognionics HD-72"),
                        )
                        .child(
                            Button::new("cog-connect")
                                .primary()
                                .label(label)
                                .on_click(cx.listener(move |this, _, _window, cx| {
                                    this.cog_connect(device_id.clone(), cx);
                                })),
                        ),
                )
            }

            CogState::Connecting => panel.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(cx.theme().foreground)
                            .child("Cognionics HD-72"),
                    )
                    .child(
                        Button::new("cog-connecting")
                            .label("Connecting...")
                            .disabled(true),
                    ),
            ),

            CogState::Streaming => {
                let audio_btn = if self.audio_enabled {
                    Button::new("cog-audio-toggle")
                        .danger()
                        .label("Stop Audio")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.stop_audio(cx);
                        }))
                } else {
                    Button::new("cog-audio-toggle")
                        .primary()
                        .label("Start Audio")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.start_audio(cognionics::NUM_CHANNELS, cx);
                        }))
                };

                let active_tab = self.active_tab;

                let waves_btn = if active_tab == Tab::Waves {
                    Button::new("tab-waves").label("Waves").primary()
                } else {
                    Button::new("tab-waves")
                        .label("Waves")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Waves;
                            cx.notify();
                        }))
                };
                let spectrum_btn = if active_tab == Tab::Spectrum {
                    Button::new("tab-spectrum").label("Spectrum").primary()
                } else {
                    Button::new("tab-spectrum")
                        .label("Spectrum")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Spectrum;
                            cx.notify();
                        }))
                };
                let pca_btn = if active_tab == Tab::Pca {
                    Button::new("tab-pca").label("PCA").primary()
                } else {
                    Button::new("tab-pca")
                        .label("PCA")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Pca;
                            cx.notify();
                        }))
                };
                let words_btn = if active_tab == Tab::Words {
                    Button::new("tab-words").label("Words").primary()
                } else {
                    Button::new("tab-words")
                        .label("Words")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Words;
                            cx.notify();
                        }))
                };
                let soundboard_btn = if active_tab == Tab::Soundboard {
                    Button::new("tab-soundboard").label("Soundboard").primary()
                } else {
                    Button::new("tab-soundboard")
                        .label("Soundboard")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Soundboard;
                            cx.notify();
                        }))
                };
                let recorder_btn = if active_tab == Tab::Recorder {
                    Button::new("tab-recorder").label("Recorder").primary()
                } else {
                    Button::new("tab-recorder")
                        .label("Recorder")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Recorder;
                            cx.notify();
                        }))
                };

                let content: Div = if active_tab == Tab::Recorder {
                    self.render_recorder_view(cx)
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
                                        .child(waveform_canvas(data, 300.0))
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
                .border_color(gpui_component::green_500())
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
                                        .size(px(8.0))
                                        .rounded_full()
                                        .bg(gpui_component::green_500()),
                                )
                                .child(
                                    div()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(cx.theme().foreground)
                                        .child("Cognionics HD-72 — 64ch @ 300 Hz"),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(waves_btn)
                                .child(spectrum_btn)
                                .child(pca_btn)
                                .child(words_btn)
                                .child(soundboard_btn)
                                .child(recorder_btn)
                                .child(audio_btn)
                                .child(
                                    Button::new("cog-disconnect")
                                        .danger()
                                        .label("Disconnect")
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.cog_disconnect(cx);
                                        })),
                                ),
                        ),
                )
                .child(content)
            }

            CogState::Error(msg) => panel.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .text_color(gpui_component::red_500())
                            .child(msg.clone()),
                    )
                    .child(
                        Button::new("cog-retry")
                            .label("Retry")
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.cog_scan(cx);
                            })),
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
    /// Labels with angle-projected positions.
    labels: Vec<(String, f32, f32)>,
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
                    labels: vec![],
                };
            }

            let mut axes = Vec::with_capacity(n);
            let mut poly = Vec::with_capacity(n);
            let mut labels = Vec::with_capacity(n);

            for (i, (label, sim)) in classes.iter().enumerate() {
                let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32
                    - std::f32::consts::FRAC_PI_2; // start from top
                let ax = cx + radius * angle.cos();
                let ay = cy + radius * angle.sin();
                axes.push((ax, ay));

                let pr = sim.clamp(0.0, 1.0) * radius;
                poly.push((cx + pr * angle.cos(), cy + pr * angle.sin()));

                // Label outside ring
                let lx = cx + (radius + 10.0) * angle.cos();
                let ly = cy + (radius + 10.0) * angle.sin();
                labels.push((label.replace('_', " "), lx, ly));
            }

            RadarPrepaint { bounds, axes, poly, cx, cy, labels }
        },
        move |_bounds, state: RadarPrepaint, window: &mut Window, _cx: &mut App| {
            if state.axes.is_empty() {
                return;
            }

            // Background
            window.paint_quad(gpui::fill(state.bounds, gpui::hsla(0.0, 0.0, 0.06, 1.0)));

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
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.08, 1.0)));
            window.paint_quad(gpui::outline(
                bounds,
                gpui::hsla(0.0, 0.0, 0.25, 1.0),
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
        .enumerate()
        .map(|(i, c)| {
            let mag = c.norm() / fft_size as f32;
            mag  as f32
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
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.06, 1.0)));
            window.paint_quad(gpui::outline(
                bounds,
                gpui::hsla(0.0, 0.0, 0.2, 1.0),
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
                                gpui::hsla(0.33, 0.9, 0.6, 1.0)
                            } else {
                                gpui::hsla(0.0, 0.0, 0.5, 1.0)
                            })
                            .child(format!("Ch{ch}")),
                    )
                    .child(spectrum_canvas(&data, ch, band));

                if is_selected {
                    cell = cell
                        .rounded(px(3.0))
                        .border_1()
                        .border_color(gpui::hsla(0.33, 0.9, 0.55, 0.8));
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
                    // Force retrain before starting
                    this.rec.classifier = TrainedClassifier::train(&this.rec.session.epochs);
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
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.06, 1.0)));
            window.paint_quad(gpui::outline(
                bounds,
                gpui::hsla(0.0, 0.0, 0.2, 1.0),
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
    Application::new().run(|cx: &mut App| {
        gpui_component::init(cx);
        gpui_component::theme::Theme::change(gpui_component::theme::ThemeMode::Dark, None, cx);

        cx.open_window(WindowOptions::default(), |window, cx| {
            let stimulus_input = cx.new(|cx| InputState::new(window, cx).placeholder("new stimulus…"));
            let profile_name_input = cx.new(|cx| InputState::new(window, cx).placeholder("profile name…"));
            let view = cx.new(|_cx| MindDaw::new(stimulus_input, profile_name_input));
            cx.new(|cx| Root::new(view, window, cx))
        })
        .unwrap();
    });
}
