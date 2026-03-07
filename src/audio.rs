use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::mpsc;

// ── Types ────────────────────────────────────────────────────────────────────

/// One snapshot of EEG data (source-agnostic).
pub struct EegFrame {
    /// channels[ch] = last `fft_size` samples for that channel.
    pub channels: Vec<Vec<f32>>,
}

/// Commands sent to the audio engine thread.
pub enum AudioCommand {
    Frame(EegFrame),
    Stop,
}

/// Handle held by the main thread to send frames to the audio engine.
pub struct AudioHandle {
    pub cmd_tx: mpsc::SyncSender<AudioCommand>,
}

// ── Sonification pipeline ────────────────────────────────────────────────────

struct SonificationPipeline {
    fft_size: usize,
    concat_len: usize,
    hann_window: Vec<f32>,
    overlap_buf: Vec<f32>,
    peak_level: f32,
    output_len: usize,
    fwd_scratch: Vec<Complex<f32>>,
    inv_scratch: Vec<Complex<f32>>,
    fft_planner: FftPlanner<f32>,
}

impl SonificationPipeline {
    fn new(num_channels: usize, fft_size: usize, audio_sample_rate: u32, poll_rate_hz: f32) -> Self {
        let concat_len = num_channels * fft_size;
        let output_len = (audio_sample_rate as f32 / poll_rate_hz) as usize;

        // Pre-compute Hann window
        let hann_window: Vec<f32> = (0..fft_size)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / fft_size as f32;
                x.sin().powi(2)
            })
            .collect();

        let mut planner = FftPlanner::new();
        // Pre-plan both FFTs so internal caches are warm
        let _ = planner.plan_fft_forward(fft_size);
        let _ = planner.plan_fft_inverse(concat_len);

        let fwd_scratch_len = planner.plan_fft_forward(fft_size).get_inplace_scratch_len();
        let inv_scratch_len = planner.plan_fft_inverse(concat_len).get_inplace_scratch_len();
        let scratch_len = fwd_scratch_len.max(inv_scratch_len);

        Self {
            fft_size,
            concat_len,
            hann_window,
            overlap_buf: vec![0.0; 64], // crossfade length
            peak_level: 1.0,
            output_len,
            fwd_scratch: vec![Complex::default(); scratch_len],
            inv_scratch: vec![Complex::default(); scratch_len],
            fft_planner: planner,
        }
    }

    fn process(&mut self, frame: &EegFrame) -> Vec<f32> {
        let fwd = self.fft_planner.plan_fft_forward(self.fft_size);
        let inv = self.fft_planner.plan_fft_inverse(self.concat_len);

        // 1. FFT each channel and concatenate
        let mut concat_spectrum = Vec::with_capacity(self.concat_len);

        for ch_data in &frame.channels {
            let n = ch_data.len().min(self.fft_size);
            let mut buf: Vec<Complex<f32>> = vec![Complex::default(); self.fft_size];

            // Copy samples with Hann window, zero-pad if short
            for i in 0..n {
                let idx = ch_data.len().saturating_sub(self.fft_size) + i;
                buf[i] = Complex::new(ch_data.get(idx).copied().unwrap_or(0.0) * self.hann_window[i], 0.0);
            }

            fwd.process_with_scratch(&mut buf, &mut self.fwd_scratch);
            concat_spectrum.extend_from_slice(&buf);
        }

        // Pad if fewer channels than expected
        concat_spectrum.resize(self.concat_len, Complex::default());

        // 2. Inverse FFT the concatenated spectrum
        inv.process_with_scratch(&mut concat_spectrum, &mut self.inv_scratch);

        // 3. Take real parts, normalize by 1/concat_len
        let inv_norm = 1.0 / self.concat_len as f32;
        let time_domain: Vec<f32> = concat_spectrum.iter().map(|c| c.re * inv_norm).collect();

        // 4. Resample to output_len
        let mut resampled = resample_linear(&time_domain, self.output_len);

        // 5. Peak-tracking normalization
        let frame_peak = resampled.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if frame_peak > self.peak_level {
            self.peak_level = frame_peak; // instant rise
        } else {
            self.peak_level *= 0.999; // slow decay
        }

        if self.peak_level > 1e-10 {
            let gain = 0.8 / self.peak_level;
            for s in &mut resampled {
                *s *= gain;
            }
        }

        // 6. Crossfade with previous frame's tail
        let crossfade_len = self.overlap_buf.len().min(resampled.len());
        for i in 0..crossfade_len {
            let t = i as f32 / crossfade_len as f32;
            // Raised-cosine crossfade
            let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            let fade_out = 1.0 - fade_in;
            resampled[i] = resampled[i] * fade_in + self.overlap_buf[i] * fade_out;
        }

        // Store tail for next frame's crossfade
        let tail_start = resampled.len().saturating_sub(self.overlap_buf.len());
        let tail = &resampled[tail_start..];
        self.overlap_buf.clear();
        self.overlap_buf.extend_from_slice(tail);

        resampled
    }
}

/// Linear interpolation resampler.
fn resample_linear(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    if input.len() == target_len {
        return input.to_vec();
    }

    let ratio = (input.len() - 1) as f64 / (target_len - 1).max(1) as f64;
    (0..target_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f64;
            let a = input[idx];
            let b = input[(idx + 1).min(input.len() - 1)];
            a + (b - a) * frac as f32
        })
        .collect()
}

// ── Audio engine ─────────────────────────────────────────────────────────────

/// Spawn the audio engine on a dedicated OS thread.
///
/// Returns an `AudioHandle` for sending EEG frames from the main thread.
pub fn spawn_audio_engine(num_channels: usize, fft_size: usize) -> anyhow::Result<AudioHandle> {
    let (cmd_tx, cmd_rx) = mpsc::sync_channel::<AudioCommand>(64);

    std::thread::Builder::new()
        .name("audio-engine".into())
        .spawn(move || {
            if let Err(e) = audio_thread(cmd_rx, num_channels, fft_size) {
                eprintln!("audio-engine error: {e}");
            }
        })?;

    Ok(AudioHandle { cmd_tx })
}

fn audio_thread(
    cmd_rx: mpsc::Receiver<AudioCommand>,
    num_channels: usize,
    fft_size: usize,
) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no audio output device found"))?;

    let supported = device.default_output_config()?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;

    let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(sample_rate as usize); // 1 sec buffer

    let config = cpal::StreamConfig {
        channels: supported.channels(),
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    // Wrap consumer in a mutex-free holder for the callback
    let consumer = std::cell::UnsafeCell::new(consumer);

    // SAFETY: The consumer is only accessed from the cpal callback thread.
    // cpal guarantees the callback runs on a single thread.
    struct ConsumerHolder(std::cell::UnsafeCell<rtrb::Consumer<f32>>);
    unsafe impl Send for ConsumerHolder {}

    let holder = ConsumerHolder(consumer);

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // SAFETY: Only accessed from this single callback thread.
            let consumer = unsafe { &mut *holder.0.get() };
            for frame in data.chunks_mut(channels) {
                let sample = consumer.pop().unwrap_or(0.0);
                // Write mono sample to all output channels
                for s in frame.iter_mut() {
                    *s = sample;
                }
            }
        },
        |err| {
            eprintln!("cpal stream error: {err}");
        },
        None,
    )?;

    stream.play()?;

    let mut pipeline = SonificationPipeline::new(num_channels, fft_size, sample_rate, 30.0);

    loop {
        match cmd_rx.recv() {
            Ok(AudioCommand::Frame(frame)) => {
                let samples = pipeline.process(&frame);

                // Skip if ring buffer is >75% full (backpressure)
                let capacity = sample_rate as usize;
                if producer.slots() < capacity / 4 {
                    continue;
                }

                for &s in &samples {
                    if producer.push(s).is_err() {
                        break; // buffer full
                    }
                }
            }
            Ok(AudioCommand::Stop) | Err(_) => break,
        }
    }

    drop(stream);
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_output_length() {
        let mut pipeline = SonificationPipeline::new(64, 64, 48000, 30.0);
        let frame = EegFrame {
            channels: vec![vec![0.0; 64]; 64],
        };
        let output = pipeline.process(&frame);
        assert_eq!(output.len(), 1600);
    }

    #[test]
    fn pipeline_nonzero_from_sine() {
        let mut pipeline = SonificationPipeline::new(4, 64, 48000, 30.0);
        let frame = EegFrame {
            channels: (0..4)
                .map(|ch| {
                    (0..64)
                        .map(|i| {
                            let freq = 10.0 * (ch + 1) as f32;
                            (2.0 * std::f32::consts::PI * freq * i as f32 / 300.0).sin()
                        })
                        .collect()
                })
                .collect(),
        };
        let output = pipeline.process(&frame);
        let energy: f32 = output.iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "output should have nonzero energy from sine input");
    }

    #[test]
    fn resample_identity() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let output = resample_linear(&input, 100);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_downsample() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let output = resample_linear(&input, 50);
        assert_eq!(output.len(), 50);
        // First and last should match
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[49] - 99.0).abs() < 1e-6);
    }
}
