mod audio;
mod cognionics;
mod soundboard;
mod streams;
#[allow(dead_code)]
mod tonnetz;
mod word_read;

use audio::{AudioCommand, AudioHandle, EegFrame};
use cognionics::{CogCommand, CogHandle, CogState};
use word_read::WordReadState;
use gpui::*;
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::{ActiveTheme, Disableable, Root};
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

    // Tonnetz / Orbifold
    tonnetz_state: tonnetz::TonnetzState,
    prev_tonnetz_chord_idx: usize,
    tonnetz_muted: bool,
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
    fn new() -> Self {
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

            tonnetz_state: tonnetz::TonnetzState::new(tonnetz::OrbifoldType::Dyads),
            prev_tonnetz_chord_idx: 0,
            tonnetz_muted: true,
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

                                    // Send audio frame (build inline to avoid borrow conflict)
                                    // Disable EEG sonification on the Tonnetz tab (it
                                    // produces static noise that drowns out chord audio).
                                    if this.audio_enabled
                                        && this.active_tab != Tab::Tonnetz
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

                            // Tonnetz navigation from brain waves
                            let nav = tonnetz::eeg_to_nav_signal(&features);
                            this.tonnetz_state.update_from_brain(nav);

                            // Play chord when it changes
                            if this.tonnetz_state.current_chord_idx
                                != this.prev_tonnetz_chord_idx
                            {
                                this.prev_tonnetz_chord_idx =
                                    this.tonnetz_state.current_chord_idx;
                                this.play_tonnetz_chord();
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
                        Button::new("cog-scan")
                            .primary()
                            .label("Connect Cognionics")
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.cog_scan(cx);
                            })),
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
                let tonnetz_btn = if active_tab == Tab::Tonnetz {
                    Button::new("tab-tonnetz").label("Tonnetz").primary()
                } else {
                    Button::new("tab-tonnetz")
                        .label("Tonnetz")
                        .on_click(cx.listener(|this, _, _window, cx| {
                            this.active_tab = Tab::Tonnetz;
                            cx.notify();
                        }))
                };

                let content: Div = if active_tab == Tab::Tonnetz {
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
                                .child(tonnetz_btn)
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

    // ── Tonnetz / Orbifold ──────────────────────────────────────────────────

    fn play_tonnetz_chord(&mut self) {
        if self.tonnetz_muted {
            return;
        }
        self.sb_ensure_engine();
        if let Some(chord) = self.tonnetz_state.current_chord() {
            let midi_notes = tonnetz::chord_to_midi_notes(chord);
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
            tonnetz::OrbifoldType::Tetrads => "T\u{2074}/S\u{2084}",
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

        // Node data: (ox, oy, oz, hue_idx, is_current)
        let node_data: Vec<(f32, f32, f32, u8, bool)> = state
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.ox, n.oy, n.oz, n.chord.hue_index(), i == current_idx))
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

                window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.06, 1.0)));

                let hues: [f32; 6] = [0.58, 0.75, 0.0, 0.15, 0.45, 0.5];
                let margin = 50.0f32;

                if is_dyads {
                    // ── T²/S₂: Möbius strip [0,6]×[0,12] as square ──────
                    let side = (w - 2.0 * margin).min(h - 2.0 * margin);
                    let cx0 = bx + w / 2.0;
                    let cy0 = by + h / 2.0;
                    let left = cx0 - side / 2.0;
                    let top = cy0 - side / 2.0;

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
                        if let (Some(&(ox1, oy1, _, _, c1)), Some(&(ox2, oy2, _, _, c2))) =
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
                        if let (Some(&(ox1, oy1, _, _, _)), Some(&(ox2, oy2, _, _, _))) =
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
                    for &(ox, oy, _, hue_idx, is_current) in &node_data {
                        let (x, y) = to_screen(ox, oy);
                        let sz = if is_current { 16.0 } else { 8.0 };
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
                        } else {
                            let hue = hues[hue_idx as usize % hues.len()];
                            window.paint_quad(gpui::fill(nb, gpui::hsla(hue, 0.6, 0.5, 0.7)));
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
                    for &(ox, oy, oz, _, _) in &node_data {
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
                    let mut screen: Vec<(usize, f32, f32, f32, u8, bool)> = node_data
                        .iter()
                        .enumerate()
                        .map(|(i, &(ox, oy, oz, hi, ic))| {
                            let (sx, sy, d) = project(ox, oy, oz);
                            (i, sx, sy, d, hi, ic)
                        })
                        .collect();
                    screen.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

                    let mut spos = vec![(0.0f32, 0.0f32); node_data.len()];
                    let mut scur = vec![false; node_data.len()];
                    for &(i, sx, sy, _, _, ic) in &screen {
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
                    for &(_i, x, y, depth, hue_idx, is_current) in &screen {
                        let ds = 0.7 + 0.3 * (depth + 1.0) / 2.0;
                        let sz = if is_current { 14.0 * ds } else { 7.0 * ds };
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
                        } else {
                            let hue = hues[hue_idx as usize % hues.len()];
                            let a = 0.4 + 0.4 * ds;
                            window.paint_quad(gpui::fill(nb, gpui::hsla(hue, 0.6, 0.5, a)));
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
            div().child(orbifold_canvas)
        };

        // ── Voice leading info ──────────────────────────────────────────────
        let current_edges = self.tonnetz_state.current_edges();
        let mut vl_items: Vec<(String, f32)> = current_edges
            .iter()
            .map(|e| {
                let other_idx = if e.from == self.tonnetz_state.current_chord_idx {
                    e.to
                } else {
                    e.from
                };
                let other = &self.tonnetz_state.nodes[other_idx].chord;
                (
                    format!("{} ({})", other.label(), other.type_label()),
                    e.distance,
                )
            })
            .collect();
        vl_items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        vl_items.truncate(8);

        let mut vl_row = div().flex().flex_wrap().gap_2();
        for (label, dist) in &vl_items {
            vl_row = vl_row.child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .border_1()
                    .border_color(cx.theme().border)
                    .child(
                        div()
                            .text_sm()
                            .text_color(cx.theme().foreground)
                            .child(label.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(format!("d={dist:.2}")),
                    ),
            );
        }

        let vl_section = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().muted_foreground)
                    .child("VOICE LEADINGS (nearest)"),
            )
            .child(vl_row);

        // Navigation buttons
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
                Button::new("orb-play")
                    .label("\u{266B} Play")
                    .primary()
                    .on_click(cx.listener(|this, _, _window, cx| {
                        this.tonnetz_muted = false;
                        this.play_tonnetz_chord();
                        cx.notify();
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
            .child(mute_btn);

        div()
            .flex()
            .flex_col()
            .gap_3()
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
                                    .text_color(cx.theme().muted_foreground)
                                    .child("ORBIFOLD"),
                            )
                            .child(orb_row),
                    )
                    .child(status),
            )
            .child(nav_row)
            .child(orbifold_canvas)
            .child(vl_section)
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
            let view = cx.new(|_cx| MindDaw::new());
            cx.new(|cx| Root::new(view, window, cx))
        })
        .unwrap();
    });
}
