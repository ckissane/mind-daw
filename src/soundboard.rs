use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SbWaveform {
    Sine,
    Sawtooth,
    Triangle,
    Square,
}

impl SbWaveform {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sine => "Sine",
            Self::Sawtooth => "Saw",
            Self::Triangle => "Triangle",
            Self::Square => "Square",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SbInstrument {
    Kick,
    Snare,
    Piano,
    Strings,
}

impl SbInstrument {
    pub fn label(self) -> &'static str {
        match self {
            Self::Kick => "Kick",
            Self::Snare => "Snare",
            Self::Piano => "Piano",
            Self::Strings => "Strings",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SbChord {
    Single,
    Major,
    Minor,
    Dom7,
    Sus4,
}

impl SbChord {
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "Single",
            Self::Major => "Major",
            Self::Minor => "Minor",
            Self::Dom7 => "Dom7",
            Self::Sus4 => "Sus4",
        }
    }

    pub fn intervals(self) -> &'static [i32] {
        match self {
            Self::Single => &[0],
            Self::Major => &[0, 4, 7],
            Self::Minor => &[0, 3, 7],
            Self::Dom7 => &[0, 4, 7, 10],
            Self::Sus4 => &[0, 5, 7],
        }
    }
}

pub enum SbCommand {
    PlayNote {
        midi: u8,
        waveform: SbWaveform,
        instrument: SbInstrument,
        chord: SbChord,
        volume: f32,
    },
    Stop,
}

pub struct SoundboardHandle {
    pub cmd_tx: mpsc::SyncSender<SbCommand>,
}

// ── Internal voice (pre-rendered samples) ────────────────────────────────────

struct Voice {
    samples: Vec<f32>,
    pos: usize,
}

impl Voice {
    fn next_sample(&mut self) -> f32 {
        if self.pos >= self.samples.len() {
            return 0.0;
        }
        let s = self.samples[self.pos];
        self.pos += 1;
        s
    }

    fn is_done(&self) -> bool {
        self.pos >= self.samples.len()
    }
}

// ── Synthesis helpers ─────────────────────────────────────────────────────────

fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0)
}

fn osc_sample(phase: f32, waveform: SbWaveform) -> f32 {
    match waveform {
        SbWaveform::Sine => (2.0 * std::f32::consts::PI * phase).sin(),
        SbWaveform::Sawtooth => 2.0 * phase - 1.0,
        SbWaveform::Triangle => {
            let p = 2.0 * phase;
            if p < 1.0 { 2.0 * p - 1.0 } else { 3.0 - 2.0 * p }
        }
        SbWaveform::Square => if phase < 0.5 { 1.0 } else { -1.0 },
    }
}

fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0xcafef00dd15ea5e5);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    STATE.store(s, Ordering::Relaxed);
    (s >> 11) as f32 / (1u64 << 53) as f32
}

fn render_kick(midi: u8, waveform: SbWaveform, sample_rate: f32, vol: f32) -> Vec<f32> {
    let base_freq = midi_to_hz(midi);
    let start_freq = (base_freq * 3.0).max(150.0);
    let end_freq = (base_freq * 0.3).max(30.0);
    let dur = 0.55_f32;
    let n = (dur * sample_rate) as usize;
    let mut phase = 0.0_f32;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let t_norm = i as f32 / n as f32;
        // Exponential frequency sweep downward
        let freq = start_freq * (end_freq / start_freq).powf(t_norm);
        let env = (1.0 - t_norm).powi(2);
        out.push(osc_sample(phase, waveform) * env * vol);
        phase += freq / sample_rate;
        if phase >= 1.0 { phase -= 1.0; }
    }
    out
}

fn render_snare(midi: u8, waveform: SbWaveform, sample_rate: f32, vol: f32) -> Vec<f32> {
    let base_freq = midi_to_hz(midi);
    let dur = 0.28_f32;
    let n = (dur * sample_rate) as usize;

    // Generate white noise and apply one-pole low-pass at center freq
    let center_freq = (base_freq * 4.0).min(4000.0);
    let alpha = {
        let w = 2.0 * std::f32::consts::PI * center_freq / sample_rate;
        w / (w + 1.0)
    };
    let mut noise: Vec<f32> = Vec::with_capacity(n);
    let mut lp = 0.0_f32;
    for _ in 0..n {
        let raw = rand_f32() * 2.0 - 1.0;
        lp += alpha * (raw - lp);
        noise.push(lp);
    }

    // Mix noise + oscillator tone body
    let mut osc_phase = 0.0_f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let noise_env = (-t * 14.0).exp();
        let osc_env = (-t * 28.0).exp();
        let osc_s = osc_sample(osc_phase, waveform) * osc_env * 0.4;
        out.push((noise[i] * noise_env * 0.9 + osc_s) * vol);
        osc_phase += base_freq / sample_rate;
        if osc_phase >= 1.0 { osc_phase -= 1.0; }
    }
    out
}

fn render_piano(midi: u8, waveform: SbWaveform, sample_rate: f32, vol: f32) -> Vec<f32> {
    let freq = midi_to_hz(midi);
    let dur = 1.8_f32;
    let n = (dur * sample_rate) as usize;
    let attack_n = (0.01 * sample_rate) as usize;
    let decay_n = (0.25 * sample_rate) as usize;
    let sustain = 0.5_f32;
    let rel_start_n = (1.1 * sample_rate) as usize;
    let rel_dur_n = (0.7 * sample_rate) as usize;

    let mut p1 = 0.0_f32;
    let mut p2 = 0.0_f32;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let env = if i < attack_n {
            i as f32 / attack_n as f32
        } else if i < attack_n + decay_n {
            1.0 - (i - attack_n) as f32 / decay_n as f32 * (1.0 - sustain)
        } else if i < rel_start_n {
            sustain
        } else {
            let t = (i - rel_start_n) as f32 / rel_dur_n as f32;
            (sustain * (1.0 - t)).max(0.0)
        };

        let s = osc_sample(p1, waveform) + osc_sample(p2, waveform) * 0.18;
        out.push(s * env * vol * 0.65);

        p1 += freq / sample_rate;
        p2 += freq * 2.0 / sample_rate;
        if p1 >= 1.0 { p1 -= 1.0; }
        if p2 >= 1.0 { p2 -= 1.0; }
    }
    out
}

fn render_strings(midi: u8, waveform: SbWaveform, sample_rate: f32, vol: f32) -> Vec<f32> {
    let base_freq = midi_to_hz(midi);
    // Strings work best with sawtooth; fall back if sine selected
    let wave = if waveform == SbWaveform::Sine { SbWaveform::Sawtooth } else { waveform };
    let dur = 3.0_f32;
    let n = (dur * sample_rate) as usize;
    let attack_n = (0.5 * sample_rate) as usize;
    let rel_start_n = (dur * 0.72 * sample_rate) as usize;
    let rel_dur_n = (0.8 * sample_rate) as usize;

    // Three detuned oscillators: -8, 0, +8 cents
    let detune_cents = [-8.0_f32, 0.0, 8.0];
    let freqs: [f32; 3] = [
        base_freq * 2.0f32.powf(detune_cents[0] / 1200.0),
        base_freq * 2.0f32.powf(detune_cents[1] / 1200.0),
        base_freq * 2.0f32.powf(detune_cents[2] / 1200.0),
    ];
    let mut phases = [0.0_f32; 3];
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let env = if i < attack_n {
            i as f32 / attack_n as f32
        } else if i < rel_start_n {
            1.0
        } else {
            let t = (i - rel_start_n) as f32 / rel_dur_n as f32;
            (1.0 - t).max(0.0)
        };

        let mut s = 0.0_f32;
        for j in 0..3 {
            s += osc_sample(phases[j], wave) / 3.0;
            phases[j] += freqs[j] / sample_rate;
            if phases[j] >= 1.0 { phases[j] -= 1.0; }
        }
        out.push(s * env * vol * 0.5);
    }

    // Gentle low-pass warm-up
    let alpha = {
        let w = 2.0 * std::f32::consts::PI * 3500.0 / sample_rate;
        w / (w + 1.0)
    };
    let mut lp = 0.0_f32;
    for s in out.iter_mut() {
        lp += alpha * (*s - lp);
        *s = lp;
    }
    out
}

fn make_voices(
    midi: u8,
    waveform: SbWaveform,
    instrument: SbInstrument,
    chord: SbChord,
    sample_rate: f32,
    vol: f32,
) -> Vec<Voice> {
    let intervals = chord.intervals();
    let per_vol = vol / intervals.len() as f32;
    intervals
        .iter()
        .map(|&semi| {
            let note = ((midi as i32) + semi).clamp(0, 127) as u8;
            let samples = match instrument {
                SbInstrument::Kick => render_kick(note, waveform, sample_rate, per_vol),
                SbInstrument::Snare => render_snare(note, waveform, sample_rate, per_vol),
                SbInstrument::Piano => render_piano(note, waveform, sample_rate, per_vol),
                SbInstrument::Strings => render_strings(note, waveform, sample_rate, per_vol),
            };
            Voice { samples, pos: 0 }
        })
        .collect()
}

// ── Engine ────────────────────────────────────────────────────────────────────

pub fn spawn_soundboard_engine() -> anyhow::Result<SoundboardHandle> {
    let (cmd_tx, cmd_rx) = mpsc::sync_channel::<SbCommand>(64);
    std::thread::Builder::new()
        .name("soundboard".into())
        .spawn(move || {
            if let Err(e) = soundboard_thread(cmd_rx) {
                eprintln!("soundboard engine error: {e}");
            }
        })?;
    Ok(SoundboardHandle { cmd_tx })
}

fn soundboard_thread(cmd_rx: mpsc::Receiver<SbCommand>) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no output device"))?;
    let supported = device.default_output_config()?;
    let sample_rate = supported.sample_rate().0 as f32;
    let channels = supported.channels() as usize;

    let voices: Arc<Mutex<Vec<Voice>>> = Arc::new(Mutex::new(Vec::new()));
    let voices_cb = Arc::clone(&voices);

    // SAFETY: Arc<Mutex<Vec<Voice>>> is Send; wrapper needed only because
    // cpal requires the closure to be 'static + Send.
    struct VHolder(Arc<Mutex<Vec<Voice>>>);
    unsafe impl Send for VHolder {}
    let holder = VHolder(voices_cb);

    let config = cpal::StreamConfig {
        channels: supported.channels(),
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            if let Ok(mut lock) = holder.0.try_lock() {
                for frame in data.chunks_mut(channels) {
                    let mut s = 0.0_f32;
                    for v in lock.iter_mut() {
                        s += v.next_sample();
                    }
                    let out = s.clamp(-1.0, 1.0);
                    for ch in frame.iter_mut() {
                        *ch = out;
                    }
                }
                lock.retain(|v| !v.is_done());
            } else {
                // Mutex contention — output silence for this callback
                for s in data.iter_mut() {
                    *s = 0.0;
                }
            }
        },
        |e| eprintln!("soundboard cpal: {e}"),
        None,
    )?;
    stream.play()?;

    // Keep stream alive while processing commands
    let _stream = stream;

    loop {
        match cmd_rx.recv() {
            Ok(SbCommand::PlayNote { midi, waveform, instrument, chord, volume }) => {
                let new_voices = make_voices(midi, waveform, instrument, chord, sample_rate, volume);
                if let Ok(mut lock) = voices.lock() {
                    if lock.len() < 32 {
                        lock.extend(new_voices);
                    }
                }
            }
            Ok(SbCommand::Stop) | Err(_) => break,
        }
    }

    Ok(())
}
