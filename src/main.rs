mod audio;
mod cognionics;
mod streams;
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

    fn cog_connect(&mut self, address: [u8; 6], cx: &mut Context<Self>) {
        if let Some(ref handle) = self.cog_handle {
            let _ = handle.cmd_tx.send(CogCommand::Connect(address));
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

            CogState::Found { address, name } => {
                let addr = *address;
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
                                    this.cog_connect(addr, cx);
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

                let content: Div = if active_tab == Tab::Words {
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

/// Prepaint state for waveform canvas.
struct WaveformPrepaint {
    bounds: Bounds<Pixels>,
    points: Vec<(f32, f32)>,
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

            // Draw the waveform trace
            let mut builder = PathBuilder::stroke(px(1.5));
            builder.move_to(point(px(state.points[0].0), px(state.points[0].1)));
            for &(x, y) in &state.points[1..] {
                builder.line_to(point(px(x), px(y)));
            }
            if let Ok(path) = builder.build() {
                window.paint_path(path, gpui::hsla(0.33, 0.9, 0.55, 1.0)); // green trace
            }

            // Draw flat (disconnected) segments in red over the green trace
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

/// Render an FFT spectrum plot for one channel using gpui canvas.
fn spectrum_canvas(data: &[f32], ch: usize) -> impl IntoElement {
    let spectrum = compute_spectrum(data, 32);

    canvas(
        move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
            let w: f32 = bounds.size.width.into();
            let h: f32 = bounds.size.height.into();
            let ox: f32 = bounds.origin.x.into();
            let oy: f32 = bounds.origin.y.into();

            if spectrum.is_empty() || w < 2.0 || h < 2.0 {
                return (bounds, Vec::new(), ch);
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

            (bounds, bars, ch)
        },
        move |_bounds: Bounds<Pixels>,
              (bounds, bars, ch): (Bounds<Pixels>, Vec<(f32, f32, f32, f32)>, usize),
              window: &mut Window,
              _cx: &mut App| {
            // Background
            window.paint_quad(gpui::fill(bounds, gpui::hsla(0.0, 0.0, 0.06, 1.0)));
            window.paint_quad(gpui::outline(
                bounds,
                gpui::hsla(0.0, 0.0, 0.2, 1.0),
                gpui::BorderStyle::Solid,
            ));

            // Channel label (draw as a small colored indicator in top-left)
            // Hue varies by channel for visual distinction
            let hue = (ch as f32 / 64.0) * 0.8;

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
                    .child(spectrum_canvas(&data, ch));

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

        grid
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
