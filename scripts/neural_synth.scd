/*
  neural_synth.scd  —  Neural Instrument backend for mind-daw
  ─────────────────────────────────────────────────────────────────────────────
  Started by ScProcess.spawn() from the Rust frontend.
  Receives OSC from Rust on port 57120 (sclang default).
  Sends metering + status back to Rust on port 57111.

  OSC IN  (Rust → SC)
    /instrument/tonnetz/axis       f  −1.0 … 1.0  (minor/dark ↔ major/bright)
    /instrument/tonnetz/tension    f   0.0 … 1.0  (tonic → distant harmony)
    /instrument/arp/speed          f   0.5 … 8.0  notes/sec
    /instrument/arp/direction      f   0.0 … 1.0  (up / up-down / random)
    /instrument/synth/waveform     f   0.0 … 3.0  (sine / tri / saw / square)
    /instrument/bass/filter        f   200 … 8000 Hz  filter cutoff
    /instrument/drums/toggle       i   1           trigger: toggle drums on/off
    /instrument/drums/pattern      i   1           trigger: advance drum pattern
    /instrument/bass/toggle        i   1           trigger: toggle bass on/off
    /instrument/bass/pattern       i   1           trigger: advance bass pattern

  OSC IN  (master params)
    /sc/master/volume              f   0.0 … 1.0
    /sc/master/reverb              f   0.0 … 1.0
    /sc/rhythm/bpm                 f  60.0 … 200.0
    /sc/master/tuning              f  430.0 … 460.0  (A4 Hz)

  OSC OUT (SC → Rust, port 57111)
    /sc/meter/left                 f   0.0 … 1.0  (linear RMS)
    /sc/meter/right                f   0.0 … 1.0
    /sc/status                     s   "booted" | "error: …"
*/

// ── Boot ─────────────────────────────────────────────────────────────────────

s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels  = 0;
s.options.sampleRate           = 44100;
s.options.memSize              = 65536;

s.waitForBoot({

    // ── Shared state ─────────────────────────────────────────────────────────

    ~rustAddr   = NetAddr("127.0.0.1", 57111); // Rust OscReceiver listens here

    ~masterVol  = 0.7;
    ~masterRev  = 0.2;
    ~bpm        = 120.0;
    ~tuningHz   = 440.0;   // A4

    // Tonnetz state
    ~tonnetzAxis    = 0.0;   // −1=minor/dark, +1=major/bright
    ~tonnetzTension = 0.0;   // 0=tonic, 1=distant

    // Arpeggio state
    ~arpSpeed    = 2.0;      // notes/sec
    ~arpDir      = 0.0;      // 0=up, 0.5=up-down, 1=random
    ~arpRunning  = false;
    ~arpWave     = 2.0;      // separate timbre from the Tonnetz synth (0=sine…3=square)

    // Waveform synth (Tonnetz chord voice only)
    ~waveShape   = 2.0;      // 0=sine, 1=tri, 2=saw, 3=square

    // Bass state
    ~bassOn      = false;
    ~bassFilter  = 800.0;    // Hz
    ~bassPatIdx  = 0;
    ~bassPatterns = [
        [0,  0,  0,  0],         // root pump
        [0,  7,  5,  3],         // walking up
        [0, -12, 7, -5],         // octave dive
        [0,  3,  5,  7],         // ascending
        [0,  7, 12,  7],         // bouncy 5th
    ];

    // Drums state — master toggle + per-instrument cueing
    ~drumsOn    = false;
    ~drumPatIdx = 0;
    ~kickOn     = true;    // independently cue each voice
    ~snareOn    = true;
    ~hihatOn    = true;
    ~clapOn     = false;   // off by default (accent colour)
    // Per-voice synthesis parameters (tweakable via OSC)
    ~kickFreq   = 55.0;    // base pitch  Hz    (30–120)
    ~kickDecay  = 0.4;     // body decay  s     (0.1–1.0)
    ~kickTone   = 0.0;     // FM crunch   0–1   (0=pure sine)
    ~snareSnap  = 0.5;     // noise ratio 0–1   (0=all body, 1=all noise)
    ~snareDecay = 0.18;    // body decay  s     (0.05–0.5)
    ~hihatOpen  = 0.0;     // open amount 0–1   (0=closed, 1=open)
    // Transition (riser → impact) pending flag
    ~riserPending = false;
    // Pad state
    ~padOn      = false;
    // ── Drum patterns [step][kick, snare, hihat, clap] ──────────────────────
    ~drumPatterns = [
        // 0 – Four-on-floor
        Array.fill(16, {|i|
            [(i%4)==0, (i%8)==4, (i%2)==0, false].collect(_.asInteger)
        }),
        // 1 – Breakbeat
        Array.fill(16, {|i|
            [(i==0)||(i==10), (i==4)||(i==12), (i%2)==0, false].collect(_.asInteger)
        }),
        // 2 – Half-time
        Array.fill(16, {|i|
            [(i==0)||(i==8), (i==4)||(i==12), (i%4)==0, false].collect(_.asInteger)
        }),
        // 3 – Trap (syncopated kick, fast hi-hats, clap on 2&4)
        Array.fill(16, {|i|
            [(i==0)||(i==6)||(i==9),
             (i==4)||(i==12),
             (i%2==0)||(i==3)||(i==7)||(i==11),
             (i==4)||(i==12)].collect(_.asInteger)
        }),
        // 4 – Dembow / Reggaeton
        Array.fill(16, {|i|
            [(i==0)||(i==6)||(i==8)||(i==14),
             (i==4)||(i==11)||(i==13),
             (i%4)==0,
             (i==4)].collect(_.asInteger)
        }),
        // 5 – Samba / Afro
        Array.fill(16, {|i|
            [(i==0)||(i==3)||(i==6)||(i==10),
             (i==2)||(i==10),
             (i%2)==0,
             (i==6)||(i==14)].collect(_.asInteger)
        }),
    ];

    // ── Buses ─────────────────────────────────────────────────────────────────

    ~mainBus = Bus.audio(s, 2);   // pre-master stereo bus
    ~revBus  = Bus.audio(s, 2);   // reverb send

    // ── SynthDefs ─────────────────────────────────────────────────────────────

    // Master output + reverb
    SynthDef(\masterOut, {|inBus, revBus, vol=0.7, revMix=0.2|
        var dry  = In.ar(inBus, 2);
        var revIn= In.ar(revBus, 2);
        var verb = FreeVerb2.ar(revIn[0], revIn[1], mix:1, room:0.6, damp:0.5);
        var sig  = (dry + (verb * revMix)) * vol;
        var peak = Amplitude.ar(sig, attackTime:0.01, releaseTime:0.1);
        // Send metering back to Rust every ~100ms
        SendTrig.ar(Impulse.ar(10), 0, peak[0].lag(0.05));
        SendTrig.ar(Impulse.ar(10), 1, peak[1].lag(0.05));
        Out.ar(0, sig);
    }).add;

    // Tonnetz voice: a simple triadic chord whose root, mode and voicing
    // follow the axis (major/minor colour) and tension (distance from tonic).
    // axis ∈ [−1, +1]: −1 = minor/dark, +1 = major/bright
    // tension ∈ [0, 1]: 0 = tonic triad, 1 = dominant 7th region
    // gate=0 → silent on boot; set gate=1 via /instrument/tonnetz/on to unmute.
    SynthDef(\tonnetzVoice, {|out, revOut, axis=0, tension=0, tuning=440, amp=0.2, gate=0|
        // Map axis → mode: negative = minor 3rd (3 semitones), positive = major (4)
        var third = axis.linlin(-1.0, 1.0, 3.0, 4.0).round(1.0);
        // Map tension → chord root offset (0=tonic, 7=dominant, 5=subdominant)
        var rootOffset = Select.kr(tension.linlin(0.0, 1.0, 0.0, 2.9).floor, [0, 7, 5]);
        // A4 tuning reference
        var a4 = tuning;
        var rootFreq = (a4 / 16.0) * (2 ** (rootOffset / 12.0));  // bass octave
        var freq1 = rootFreq * 4;                    // root
        var freq2 = rootFreq * 4 * (2 ** (third/12));     // 3rd
        var freq3 = rootFreq * 4 * (2 ** (7.0/12.0));      // 5th
        // gate controls on/off with a smooth fade in/out (no snap)
        var env   = EnvGen.kr(Env.asr(0.15, 1, 0.3), gate, doneAction:0);
        var osc   = SinOsc.ar([freq1, freq2, freq3], 0, [0.3, 0.25, 0.25]) * env;
        var sig   = Mix(osc) * amp;
        Out.ar(out,    [sig, sig]);
        Out.ar(revOut, [sig, sig] * 0.5);
    }).add;

    // Arpeggio: cycles through chord tones at speed (notes/sec).
    // direction: 0=up, 0.5=up-down, 1=random
    // wave: 0=sine…3=square, fractional values crossfade (independent of Tonnetz synth).
    SynthDef(\arpNote, {|out, revOut, freq=440, wave=2, amp=0.15, dur=0.25|
        var env  = EnvGen.kr(Env.perc(0.005, dur * 0.9), doneAction:2);
        var sine = SinOsc.ar(freq);
        var tri  = LFTri.ar(freq);
        var saw  = Saw.ar(freq);
        var sq   = Pulse.ar(freq, 0.5);
        // SelectX gives continuous crossfade between adjacent waveforms
        var sig  = SelectX.ar(wave / 3.0, [sine, tri, saw, sq]) * env * amp;
        Out.ar(out,    [sig, sig]);
        Out.ar(revOut, [sig, sig] * 0.4);
    }).add;

    // Bass: simple subby bass with resonant filter
    SynthDef(\bassNote, {|out, revOut, freq=55, filt=800, amp=0.35, dur=0.45|
        var env  = EnvGen.kr(Env.perc(0.005, dur, 1, -4), doneAction:2);
        var osc  = SinOsc.ar(freq) + (SinOsc.ar(freq*2) * 0.3);
        var fil  = RLPF.ar(osc, filt.clip(60, 18000), 0.4);
        var sig  = fil * env * amp;
        Out.ar(out,    [sig, sig] * 0.7);
        Out.ar(revOut, [sig, sig] * 0.15);
    }).add;

    // ── Drum SynthDefs ────────────────────────────────────────────────────────
    // Each has tweakable synthesis parameters — controllable from Rust via OSC.

    // Kick: FM sine with pitched transient.  freq=base pitch, decay=body length,
    //       tone=FM index (0=clean sub, 1=crunchy/aggressive).
    SynthDef(\kick, {|out, amp=0.8, freq=55, decay=0.4, tone=0|
        var env   = EnvGen.kr(Env.perc(0.001, decay, 1, -4), doneAction:2);
        var pitch = EnvGen.kr(Env.perc(0.001, 0.08), levelScale: freq * 3.5) + freq;
        var mod   = SinOsc.ar(pitch * 2) * tone * freq * 0.5;
        var sig   = SinOsc.ar(pitch + mod) * env * amp;
        Out.ar(out, [sig, sig]);
    }).add;

    // Snare: body sine + noise crack.  snap=noise ratio, decay=length, tone=body freq.
    SynthDef(\snare, {|out, amp=0.6, snap=0.5, decay=0.18, tone=200|
        var env   = EnvGen.kr(Env.perc(0.002, decay, 1, -4), doneAction:2);
        var body  = SinOsc.ar(tone);
        var crack = HPF.ar(WhiteNoise.ar, 800);
        var sig   = (body * (1 - snap) + crack * snap) * env * amp;
        Out.ar(out, [sig, sig]);
    }).add;

    // HiHat: filtered noise.  open=0 → 50 ms closed; open=1 → 550 ms open hat.
    SynthDef(\hihat, {|out, amp=0.3, open=0|
        var dur = open.linlin(0, 1, 0.05, 0.55);
        var env = EnvGen.kr(Env.perc(0.001, dur, 1, -5), doneAction:2);
        var sig = HPF.ar(WhiteNoise.ar, 8000) * env * amp;
        Out.ar(out, [sig, sig]);
    }).add;

    // Clap: three layered noise bursts (gives that "hands slapping together" flutter).
    SynthDef(\clap, {|out, amp=0.5|
        var e1  = EnvGen.kr(Env.perc(0.001, 0.04));
        var e2  = EnvGen.kr(Env.perc(0.010, 0.07));
        var e3  = EnvGen.kr(Env.perc(0.050, 0.12), doneAction:2);
        var sig = BPF.ar(
            WhiteNoise.ar * e1 + WhiteNoise.ar * e2 * 0.7 + WhiteNoise.ar * e3 * 0.4,
            1400, 0.55) * amp;
        Out.ar(out, [sig, sig]);
    }).add;

    // Riser: high-pass noise sweep + rising sine tone, lasts exactly 'dur' seconds.
    // Triggered at a bar boundary; dur is computed from BPM so it fills one bar exactly.
    SynthDef(\riser, {|out, revOut, dur=4.0, amp=0.45|
        var env    = EnvGen.kr(Env.linen(0.1, dur - 0.3, 0.2), doneAction:2);
        var hpfreq = Line.kr(300, 15000, dur);
        var noise  = HPF.ar(WhiteNoise.ar * 0.4, hpfreq);
        var sweep  = Line.kr(60, 2800, dur);
        var tone   = SinOsc.ar(sweep) * 0.25;
        var sig    = (noise + tone) * env * amp;
        Out.ar(out,    [sig, sig]);
        Out.ar(revOut, [sig, sig] * 0.35);
    }).add;

    // Impact: sub-boom + noise crack — fires at the downbeat after the riser.
    SynthDef(\impact, {|out, revOut, amp=0.95|
        var env   = EnvGen.kr(Env.perc(0.001, 0.75, 1, -4), doneAction:2);
        var sub   = SinOsc.ar(EnvGen.kr(Env.perc(0.001, 0.18), levelScale: 160) + 35) * 0.9;
        var crack = WhiteNoise.ar * EnvGen.kr(Env.perc(0.001, 0.12)) * 0.6;
        var sig   = (sub + crack) * env * amp;
        Out.ar(out,    [sig, sig]);
        Out.ar(revOut, [sig, sig] * 0.2);
    }).add;

    // Pad: detuned chord pad, same harmony as Tonnetz but one octave lower & softer.
    // Slow attack / release = ambient counterchord texture.
    SynthDef(\padVoice, {|out, revOut, axis=0, tension=0, tuning=440, amp=0.12, gate=0|
        var third      = axis.linlin(-1.0, 1.0, 3.0, 4.0).round(1.0);
        var rootOffset = Select.kr(tension.linlin(0.0, 1.0, 0.0, 2.9).floor, [0, 7, 5]);
        var rootFreq   = (tuning / 32.0) * (2 ** (rootOffset / 12.0)); // octave below tonnetz
        var freqs      = [rootFreq * 2,
                          rootFreq * 2 * (2 ** (third / 12.0)),
                          rootFreq * 2 * (2 ** (7.0 / 12.0))];
        // Three detuned copies per chord tone for warmth
        var osc = Mix.fill(3, {|j|
            var f   = freqs[j.min(2)];
            var det = [-0.018, 0, 0.018][j];
            LFTri.ar(f * (1 + det), 0, 0.12);
        });
        var env = EnvGen.kr(Env.asr(1.8, 1, 2.5), gate, doneAction:0);
        var sig = LPF.ar(osc, 2800) * env * amp;
        Out.ar(out,    [sig * 0.9, sig * 1.1]);   // gentle stereo spread
        Out.ar(revOut, [sig, sig] * 0.75);
    }).add;

    // ── Start master ─────────────────────────────────────────────────────────

    s.sync;

    ~masterSynth = Synth(\masterOut, [
        \inBus, ~mainBus, \revBus, ~revBus,
        \vol, ~masterVol, \revMix, ~masterRev
    ], addAction:\addToTail);

    // ── Metering: receive SendTrig from masterOut ──────────────────────────

    OSCFunc({ |msg|
        var id  = msg[2].asInteger;   // 0=left, 1=right
        var val = msg[3].asFloat.clip(0,1);
        if(id == 0) { ~rustAddr.sendMsg("/sc/meter/left",  val) };
        if(id == 1) { ~rustAddr.sendMsg("/sc/meter/right", val) };
    }, '/tr', s.addr);

    // ── Tonnetz chord engine ───────────────────────────────────────────────

    // Start silent (gate=0) — Rust sends /instrument/tonnetz/on 1 to unmute.
    ~tonnetzSynth = Synth(\tonnetzVoice, [
        \out, ~mainBus, \revOut, ~revBus,
        \axis, ~tonnetzAxis, \tension, ~tonnetzTension,
        \tuning, ~tuningHz, \gate, 0
    ]);

    ~updateTonnetz = {
        ~tonnetzSynth.set(\axis, ~tonnetzAxis, \tension, ~tonnetzTension, \tuning, ~tuningHz);

        // Keep pad voice in sync with chord
        if(~padSynth.notNil) {
            ~padSynth.set(\axis, ~tonnetzAxis, \tension, ~tonnetzTension, \tuning, ~tuningHz);
        };

        // Arp chord-following: minor (axis<0) vs major; add flat-7 when tension > 0.5.
        // This is what makes an arp feel "intelligent" — the notes change with the harmony.
        var minor   = ~tonnetzAxis < 0.0;
        var seventh = ~tonnetzTension > 0.5;
        var third   = if(minor, 3, 4);
        ~arpNotes = if(seventh,
            [0, third, 7, 10, 12 + third],  // 7th chord voicing
            [0, third, 7, 12]               // simple triad + octave
        );
    };

    // Beat-quantized chord clock: snap chord to current EEG state once per bar.
    // Also fires riser+impact transition when queued.
    ~barTask = Task({
        loop {
            ~updateTonnetz.value;
            // Queue a riser that fills the current bar, impact lands on next downbeat
            if(~riserPending) {
                var barDur = 60.0 / ~bpm * 4.0;
                Synth(\riser, [\out, ~mainBus, \revOut, ~revBus, \dur, barDur * 0.97]);
                SystemClock.sched(barDur * 0.97, {
                    Synth(\impact, [\out, ~mainBus, \revOut, ~revBus]);
                    nil;
                });
                ~riserPending = false;
                "Riser fired".postln;
            };
            (60.0 / ~bpm * 4.0).wait;
        }
    });
    ~barTask.play;

    // Pad voice — starts silent (gate=0), toggled via /instrument/pad/toggle
    ~padSynth = Synth(\padVoice, [
        \out, ~mainBus, \revOut, ~revBus,
        \axis, ~tonnetzAxis, \tension, ~tonnetzTension,
        \tuning, ~tuningHz, \gate, 0
    ], addAction:\addToHead);

    // ── Arpeggio engine ───────────────────────────────────────────────────

    ~arpStep = 0;
    ~arpNotes = [0, 4, 7, 12];   // major triad + octave (in semitones above root)

    ~arpTick = {
        if(~arpRunning) {
            var rootMidi = 60;  // C4, will shift with Tonnetz axis later
            var noteOffset = case
                { ~arpDir < 0.35 } { ~arpNotes[~arpStep % ~arpNotes.size] }
                { ~arpDir < 0.65 } {
                    var up = ~arpNotes ++ ~arpNotes.reverse;
                    up[~arpStep % up.size]
                }
                { ~arpNotes.choose };
            var midi = rootMidi + noteOffset;
            var freq = ~tuningHz * (2 ** ((midi - 69.0) / 12.0));
            var dur  = 1.0 / ~arpSpeed;
            Synth(\arpNote, [
                \out, ~mainBus, \revOut, ~revBus,
                \freq, freq, \wave, ~arpWave, \amp, 0.15, \dur, dur * 0.85
            ]);
            ~arpStep = ~arpStep + 1;
        };
    };

    ~arpTask = Task({
        loop {
            ~arpTick.value;
            (1.0 / ~arpSpeed).wait;
        }
    });
    ~arpTask.play;

    // ── Drum sequencer ─────────────────────────────────────────────────────

    ~drumStep = 0;

    ~drumTick = {
        if(~drumsOn) {
            var pat  = ~drumPatterns[~drumPatIdx];
            var step = pat[~drumStep % 16];
            // Each instrument is independently cued and uses its synthesis parameters
            if((step[0] == 1) && ~kickOn)  {
                Synth(\kick,  [\out, ~mainBus,
                    \amp, 0.8, \freq, ~kickFreq, \decay, ~kickDecay, \tone, ~kickTone]) };
            if((step[1] == 1) && ~snareOn) {
                Synth(\snare, [\out, ~mainBus,
                    \amp, 0.6, \snap, ~snareSnap, \decay, ~snareDecay]) };
            if((step[2] == 1) && ~hihatOn) {
                Synth(\hihat, [\out, ~mainBus, \amp, 0.3, \open, ~hihatOpen]) };
            if((step[3] == 1) && ~clapOn)  {
                Synth(\clap,  [\out, ~mainBus, \amp, 0.5]) };
            ~drumStep = ~drumStep + 1;
        };
    };

    ~drumTask = Task({
        loop {
            ~drumTick.value;
            (60.0 / (~bpm * 4.0)).wait;  // 16th note
        }
    });
    ~drumTask.play;

    // ── Bass sequencer ─────────────────────────────────────────────────────

    ~bassStep = 0;

    ~bassTick = {
        if(~bassOn) {
            var pat     = ~bassPatterns[~bassPatIdx];
            var offset  = pat[~bassStep % pat.size];
            var rootMidi= 36;   // C2
            var midi    = rootMidi + offset;
            var freq    = ~tuningHz * (2 ** ((midi - 69.0) / 12.0));
            var beatDur = 60.0 / ~bpm;
            Synth(\bassNote, [
                \out, ~mainBus, \revOut, ~revBus,
                \freq, freq, \filt, ~bassFilter, \amp, 0.35, \dur, beatDur * 0.9
            ]);
            ~bassStep = ~bassStep + 1;
        };
    };

    ~bassTask = Task({
        loop {
            ~bassTick.value;
            (60.0 / ~bpm).wait;  // quarter note
        }
    });
    ~bassTask.play;

    // ── OSC handlers (Rust → SC) ───────────────────────────────────────────

    // Tonnetz axis (continuous) — updates state only; chord snaps on next bar clock.
    OSCdef(\tonnetzAxis, { |msg|
        ~tonnetzAxis = msg[1].asFloat.clip(-1.0, 1.0);
    }, '/instrument/tonnetz/axis');

    // Tonnetz tension (continuous) — updates state only.
    OSCdef(\tonnetzTension, { |msg|
        ~tonnetzTension = msg[1].asFloat.clip(0.0, 1.0);
    }, '/instrument/tonnetz/tension');

    // Manual chord shift — snap immediately without waiting for the bar clock.
    OSCdef(\tonnetzShift, { |msg|
        ~updateTonnetz.value;
        "Chord shifted".postln;
    }, '/instrument/tonnetz/shift');

    // Arp speed (continuous) — only updates speed, does NOT start the arp.
    // Use /instrument/arp/toggle to start/stop.
    OSCdef(\arpSpeed, { |msg|
        ~arpSpeed = msg[1].asFloat.clip(0.5, 8.0);
    }, '/instrument/arp/speed');

    // Arp direction (continuous)
    OSCdef(\arpDir, { |msg|
        ~arpDir = msg[1].asFloat.clip(0.0, 1.0);
    }, '/instrument/arp/direction');

    // Arp wave morph (independent of Tonnetz synth)
    OSCdef(\arpWave, { |msg|
        ~arpWave = msg[1].asFloat.clip(0.0, 3.0);
    }, '/instrument/arp/wave');

    // Waveform morph (continuous) — matches Rust /instrument/synth/morph
    OSCdef(\waveShape, { |msg|
        ~waveShape = msg[1].asFloat.clip(0.0, 3.0);
    }, '/instrument/synth/morph');

    // Tonnetz on/off (gate)
    OSCdef(\tonnetzOn, { |msg|
        var on = msg[1].asInteger;
        ~tonnetzSynth.set(\gate, on);
        ("Tonnetz: " ++ if(on == 1, "ON", "OFF")).postln;
    }, '/instrument/tonnetz/on');

    // Arpeggio toggle
    OSCdef(\arpToggle, { |msg|
        ~arpRunning = ~arpRunning.not;
        ("Arp: " ++ if(~arpRunning, "ON", "OFF")).postln;
    }, '/instrument/arp/toggle');

    // Bass filter (continuous)
    OSCdef(\bassFilter, { |msg|
        ~bassFilter = msg[1].asFloat.clip(200.0, 8000.0);
    }, '/instrument/bass/filter');

    // Drums toggle (event trigger)
    OSCdef(\drumsToggle, { |msg|
        ~drumsOn = ~drumsOn.not;
        ~drumStep = 0;
        ("Drums: " ++ if(~drumsOn, "ON", "OFF")).postln;
    }, '/instrument/drums/toggle');

    // Drums pattern advance (event trigger)
    OSCdef(\drumsPattern, { |msg|
        ~drumPatIdx = (~drumPatIdx + 1) % ~drumPatterns.size;
        ("Drum pattern: " ++ ~drumPatIdx).postln;
    }, '/instrument/drums/pattern');

    // Bass toggle (event trigger)
    OSCdef(\bassToggle, { |msg|
        ~bassOn = ~bassOn.not;
        ~bassStep = 0;
        ("Bass: " ++ if(~bassOn, "ON", "OFF")).postln;
    }, '/instrument/bass/toggle');

    // Bass pattern advance (event trigger)
    OSCdef(\bassPattern, { |msg|
        ~bassPatIdx = (~bassPatIdx + 1) % ~bassPatterns.size;
        ("Bass pattern: " ++ ~bassPatIdx).postln;
    }, '/instrument/bass/pattern');

    // ── Per-instrument drum cueing ─────────────────────────────────────────
    OSCdef(\drumKickOn,  { |msg| ~kickOn  = msg[1].asInteger > 0 }, '/instrument/drums/kick/on');
    OSCdef(\drumSnareOn, { |msg| ~snareOn = msg[1].asInteger > 0 }, '/instrument/drums/snare/on');
    OSCdef(\drumHihatOn, { |msg| ~hihatOn = msg[1].asInteger > 0 }, '/instrument/drums/hihat/on');
    OSCdef(\drumClapOn,  { |msg| ~clapOn  = msg[1].asInteger > 0 }, '/instrument/drums/clap/on');

    // ── Drum synthesis parameters ─────────────────────────────────────────
    OSCdef(\drumKickFreq,   { |msg| ~kickFreq   = msg[1].asFloat.clip(30, 120)  }, '/instrument/drums/kick/freq');
    OSCdef(\drumKickDecay,  { |msg| ~kickDecay  = msg[1].asFloat.clip(0.1, 1.0) }, '/instrument/drums/kick/decay');
    OSCdef(\drumKickTone,   { |msg| ~kickTone   = msg[1].asFloat.clip(0, 1)     }, '/instrument/drums/kick/tone');
    OSCdef(\drumSnareSnap,  { |msg| ~snareSnap  = msg[1].asFloat.clip(0, 1)     }, '/instrument/drums/snare/snap');
    OSCdef(\drumSnareDecay, { |msg| ~snareDecay = msg[1].asFloat.clip(0.05, 0.5)}, '/instrument/drums/snare/decay');
    OSCdef(\drumHihatOpen,  { |msg| ~hihatOpen  = msg[1].asFloat.clip(0, 1)     }, '/instrument/drums/hihat/open');

    // ── Pad ───────────────────────────────────────────────────────────────
    OSCdef(\padToggle, { |msg|
        ~padOn = ~padOn.not;
        ~padSynth.set(\gate, if(~padOn, 1, 0));
        ("Pad: " ++ if(~padOn, "ON", "OFF")).postln;
    }, '/instrument/pad/toggle');

    // ── Transitions ───────────────────────────────────────────────────────
    // Queue a riser — it fires on the next bar downbeat, impact lands right after.
    OSCdef(\transRiser, { |msg|
        ~riserPending = true;
        "Riser queued for next bar boundary".postln;
    }, '/instrument/transition/riser');

    // ── Master parameters ──────────────────────────────────────────────────

    OSCdef(\masterVol, { |msg|
        ~masterVol = msg[1].asFloat.clip(0.0, 1.0);
        ~masterSynth.set(\vol, ~masterVol);
    }, '/sc/master/volume');

    OSCdef(\masterRev, { |msg|
        ~masterRev = msg[1].asFloat.clip(0.0, 1.0);
        ~masterSynth.set(\revMix, ~masterRev);
    }, '/sc/master/reverb');

    OSCdef(\masterBpm, { |msg|
        ~bpm = msg[1].asFloat.clip(60.0, 200.0);
    }, '/sc/rhythm/bpm');

    OSCdef(\masterTuning, { |msg|
        ~tuningHz = msg[1].asFloat.clip(430.0, 460.0);
        ~updateTonnetz.value;
    }, '/sc/master/tuning');

    // ── Tell Rust we're ready ──────────────────────────────────────────────

    s.sync;
    "neural_synth ready".postln;
    ~rustAddr.sendMsg("/sc/status", "booted");

}); // end s.waitForBoot
