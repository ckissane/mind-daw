mod audio;
mod cognionics;
mod streams;

use audio::{AudioCommand, AudioHandle, EegFrame};
use cognionics::{CogCommand, CogHandle, CogState};
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

    // UI
    active_tab: Tab,
}

const COG_BUFFER_CAPACITY: usize = 512;

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

            active_tab: Tab::Waves,
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
                        smol::Timer::after(std::time::Duration::from_millis(33)).await;

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
            let frame = EegFrame {
                channels: self
                    .cog_buffer
                    .iter()
                    .map(|buf| {
                        let n = buf.len().min(64);
                        buf.iter().rev().take(n).rev().copied().collect()
                    })
                    .collect(),
            };
            let _ = handle.cmd_tx.try_send(AudioCommand::Frame(frame));
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
        cx.notify();
    }

    /// Start a ~30fps async poll loop that drains samples and state from the BT worker.
    fn start_cog_poll(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async |this, cx| {
            loop {
                smol::Timer::after(std::time::Duration::from_millis(33)).await;

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
                                .map(|buf| buf.iter().copied().collect())
                                .collect();

                            this.send_audio_frame_from_cog();
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
                                    .child(waveform_canvas(data))
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

                let content: Div = if active_tab == Tab::Spectrum {
                    render_spectrum_grid(cog_waveform_data)
                } else {
                    div().flex().flex_col().gap_1().children(
                        cog_waveform_data
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
                                            .text_color(cx.theme().muted_foreground)
                                            .child(format!("Ch{ch}")),
                                    )
                                    .child(waveform_canvas(data))
                                    .into_any_element()
                            })
                            .collect::<Vec<_>>(),
                    )
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
}

/// Render an oscilloscope-style waveform trace using gpui canvas with stroked paths.
/// Draws vertical bars at the detected autocorrelation period interval.
fn waveform_canvas(data: &[f32]) -> impl IntoElement {
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

            WaveformPrepaint {
                bounds,
                points,
                period_xs,
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
                    window.paint_path(path, gpui::hsla(0.6, 0.5, 0.45, 0.5)); // blue-ish, semi-transparent
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
    let spectrum = compute_spectrum(data, 128);

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

/// Render the 8x8 spectrum grid for all 64 channels.
fn render_spectrum_grid(waveform_data: &[Vec<f32>]) -> Div {
    let cols = 8;
    let rows = 8;

    let mut grid = div().flex().flex_col().gap(px(2.0));

    for row in 0..rows {
        let mut row_div = div().flex().gap(px(2.0));
        for col in 0..cols {
            let ch = row * cols + col;
            let data = waveform_data.get(ch).cloned().unwrap_or_default();

            row_div = row_div.child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(gpui::hsla(0.0, 0.0, 0.5, 1.0))
                            .child(format!("Ch{ch}")),
                    )
                    .child(spectrum_canvas(&data, ch)),
            );
        }
        grid = grid.child(row_div);
    }

    grid
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
