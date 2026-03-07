mod audio;
mod cognionics;
mod streams;

use audio::{AudioCommand, AudioHandle, EegFrame};
use cognionics::{CogCommand, CogHandle, CogState};
use gpui::*;
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::{ActiveTheme, Disableable, Root};
use std::collections::VecDeque;
use streams::{PairedStream, StreamMeta};

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
                            .take(8)
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
                                    .child(waveform_bar(data))
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
                .child(
                    div().flex().flex_col().gap_1().children(
                        cog_waveform_data
                            .iter()
                            .enumerate()
                            .take(8)
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
                                    .child(waveform_bar(data))
                                    .into_any_element()
                            })
                            .collect::<Vec<_>>(),
                    ),
                )
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

/// Render a simple ASCII-style waveform bar from sample data.
fn waveform_bar(data: &[f32]) -> Div {
    let display_width = 60;
    let text = if data.is_empty() {
        "—".repeat(display_width)
    } else {
        let step = (data.len() as f32 / display_width as f32).max(1.0);
        let mut chars = String::with_capacity(display_width);
        for i in 0..display_width {
            let idx = ((i as f32) * step) as usize;
            let val = data.get(idx).copied().unwrap_or(0.0);
            let ch = if val > 0.5 {
                '█'
            } else if val > 0.2 {
                '▓'
            } else if val > 0.0 {
                '▒'
            } else if val > -0.2 {
                '░'
            } else if val > -0.5 {
                '▒'
            } else {
                '▓'
            };
            chars.push(ch);
        }
        chars
    };

    div()
        .flex_1()
        .font_family("monospace")
        .text_xs()
        .rounded_sm()
        .px_1()
        .child(text)
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
