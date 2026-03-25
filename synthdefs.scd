// mind-daw SuperCollider SynthDefs
//
// How to use:
//   1. Open this file in the SuperCollider IDE (scide)
//   2. Select all (Ctrl+A) and evaluate (Ctrl+Enter)
//   3. Wait for "mind-daw SynthDefs loaded." in the post window
//   4. In mind-daw, click "Connect SC" in the Tonnetz tab

// Launch the IDE with:  pw-jack scide
// This routes JACK audio through PipeWire (required on most modern Linux).
s.boot;
s.waitForBoot {

// ── Warm ambient pad: 7 detuned sines + lowpass filter ──────────────────
SynthDef(\mpad, { |out=0, freq=220, amp=0.15, gate=1,
                   ffreq=1200, rq=0.6, detune=0.008,
                   atk=0.8, rel=2.0, pan=0|
    var sig, env, n=7;
    env = EnvGen.kr(Env.asr(atk, 1, rel), gate, doneAction: 2);
    sig = Mix.fill(n, { |i|
        var dt = detune * (i - (n-1)/2) / (n-1);
        SinOsc.ar(freq * (1 + dt), {2pi.rand}!2)
    }) / n;
    sig = BLowPass4.ar(sig, ffreq.clip(80, 18000), rq.clip(0.1, 2));
    sig = sig * env * amp;
    sig = Balance2.ar(sig[0], sig[1], pan);
    Out.ar(out, sig);
}).add;

// ── Rival-Consoles-inspired pluck ───────────────────────────────────────
// Sawtooth + pitch vibrato (3.1 Hz, ~14 cents) + 12dB filter envelope
// + soft tanh saturation.  Velocity controls brightness and decay.
SynthDef(\mrcpluck, { |out=0, freq=440, amp=0.25, vel=0.8,
                       fbase=112, fdecay=0.3, vdecay=0.85,
                       vibrate=3.1, vibdepth=0.0081, pan=0|
    var sig, fenv, aenv, vib;
    vib = SinOsc.kr(vibrate, {2pi.rand}) * vibdepth * freq;
    sig = LFSaw.ar(freq + vib, 0, 0.7) + WhiteNoise.ar(0.03);
    fenv = EnvGen.kr(Env.perc(0, fdecay * (0.5 + vel)), doneAction: 0);
    sig = BLowPass.ar(sig, (fbase + (4000 * fenv * vel)).clip(20, 18000), 0.7);
    aenv = EnvGen.kr(Env.perc(0, vdecay * (0.5 + vel * 0.5)), doneAction: 2);
    sig = Pan2.ar(sig * aenv * amp * vel, pan);
    sig = (sig * 1.5).tanh * 0.7;
    Out.ar(out, sig);
}).add;

// ── Plucked string (Karplus-Strong) ─────────────────────────────────────
SynthDef(\mpluck, { |out=0, freq=440, amp=0.3, dur=2, coef=0.1, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.linen(0.001, dur, 0.1), doneAction: 2);
    sig = Pluck.ar(PinkNoise.ar(amp), Impulse.kr(0), 0.2,
                   freq.reciprocal, dur, coef);
    sig = Pan2.ar(sig * env, pan);
    Out.ar(out, sig);
}).add;

// ── Bell / chime: fundamental + inharmonic partials ─────────────────────
SynthDef(\mbell, { |out=0, freq=880, amp=0.2, dur=3, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.perc(0.005, dur), doneAction: 2);
    sig = SinOsc.ar(freq) * env * amp;
    sig = sig + (SinOsc.ar(freq * 2.01, 0, amp * 0.3) * env);
    sig = sig + (SinOsc.ar(freq * 3.03, 0, amp * 0.1) * env);
    sig = Pan2.ar(sig, pan);
    Out.ar(out, sig);
}).add;

// ── Bass drone: sub saw + sine through lowpass ──────────────────────────
SynthDef(\mdrone, { |out=0, freq=55, amp=0.12, gate=1, ffreq=400|
    var sig, env;
    env = EnvGen.kr(Env.asr(1.5, 1, 2), gate, doneAction: 2);
    sig = LFSaw.ar([freq, freq*1.002]) * 0.5;
    sig = sig + (SinOsc.ar(freq * 0.5) * 0.5);
    sig = BLowPass4.ar(sig, ffreq.clip(60, 5000), 0.5);
    sig = sig * env * amp;
    Out.ar(out, sig);
}).add;

// ── Percussive pulse: sine + noise transient ────────────────────────────
SynthDef(\mpulse, { |out=0, freq=440, amp=0.15, dur=0.1, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.perc(0.002, dur), doneAction: 2);
    sig = SinOsc.ar(freq) * env * amp;
    sig = sig + (WhiteNoise.ar(amp * 0.1) * EnvGen.kr(Env.perc(0.001, 0.02)));
    sig = Pan2.ar(sig, pan);
    Out.ar(out, sig);
}).add;

// ── Soft kick drum: pitched sine body + click transient ─────────────────
SynthDef(\mkick, { |out=0, amp=0.4, dur=0.4|
    var body, click, env, clickEnv, freqEnv;
    freqEnv = EnvGen.kr(Env.perc(0.001, 0.07), doneAction: 0) * 200 + 45;
    env = EnvGen.kr(Env.perc(0.005, dur), doneAction: 2);
    clickEnv = EnvGen.kr(Env.perc(0.001, 0.01));
    body = SinOsc.ar(freqEnv) * env;
    click = WhiteNoise.ar(0.3) * clickEnv;
    Out.ar(out, (body + click) * amp ! 2);
}).add;

// ── Closed hi-hat: filtered noise burst ─────────────────────────────────
SynthDef(\mhat, { |out=0, amp=0.12, dur=0.06, pan=0|
    var sig, env;
    env = EnvGen.kr(Env.perc(0.001, dur), doneAction: 2);
    sig = BHiPass4.ar(WhiteNoise.ar(1), 7000, 0.5) * env * amp;
    sig = Pan2.ar(sig, pan);
    Out.ar(out, sig);
}).add;

// ── Ethereal shimmer pad: detuned sines + sub octave + slow LFO ────────
SynthDef(\mshimmer, { |out=0, freq=220, amp=0.08, gate=1,
                       ffreq=900, detune=0.015, atk=3, rel=5, pan=0|
    var sig, env, sub, n=9;
    env = EnvGen.kr(Env.asr(atk, 1, rel), gate, doneAction: 2);
    sig = Mix.fill(n, { |i|
        var dt = detune * (i - (n-1)/2) / (n-1);
        var lfo = SinOsc.kr(0.1 + (i * 0.04), {2pi.rand}) * 0.003;
        SinOsc.ar(freq * (1 + dt + lfo), {2pi.rand}!2)
    }) / n;
    sub = SinOsc.ar(freq * 0.5, 0, 0.3);
    sig = sig + sub;
    sig = BLowPass4.ar(sig, ffreq.clip(80, 12000), 0.4);
    sig = sig * env * amp;
    sig = Balance2.ar(sig[0], sig[1], pan);
    Out.ar(out, sig);
}).add;

// ── Lo-fi / bitcrushed lead: saw + noise + decimator + resonant filter ──
// Crystal Castles inspired: harsh digital texture, resonant filter sweep
SynthDef(\mlofi, { |out=0, freq=440, amp=0.2, vel=0.8,
                    fbase=200, fdecay=0.4, vdecay=0.6,
                    crush=8, pan=0|
    var sig, fenv, aenv;
    sig = LFSaw.ar(freq, 0, 0.6) + LFPulse.ar(freq * 1.003, 0, 0.5, 0.3);
    sig = sig + WhiteNoise.ar(0.05);
    // Bitcrush: reduce bit depth
    sig = sig.round(2.pow(crush.neg));
    // Resonant filter with fast envelope
    fenv = EnvGen.kr(Env.perc(0.001, fdecay * (0.5 + vel)), doneAction: 0);
    sig = RLPF.ar(sig, (fbase + (6000 * fenv * vel)).clip(60, 16000), 0.3);
    aenv = EnvGen.kr(Env.perc(0.001, vdecay * (0.5 + vel * 0.5)), doneAction: 2);
    sig = Pan2.ar(sig * aenv * amp * vel, pan);
    sig = (sig * 2.0).tanh * 0.5; // harder saturation
    Out.ar(out, sig);
}).add;

// ── Glass pad: pure tones + 5th + octave harmonics, very slow ───────────
// Ethereal, crystalline, choir-like
SynthDef(\mglass, { |out=0, freq=220, amp=0.06, gate=1,
                     ffreq=2000, atk=4, rel=6, pan=0|
    var sig, env, n=5;
    env = EnvGen.kr(Env.asr(atk, 1, rel), gate, doneAction: 2);
    sig = SinOsc.ar(freq, 0, 0.4);
    sig = sig + SinOsc.ar(freq * 1.498, 0, 0.2);  // slightly detuned 5th
    sig = sig + SinOsc.ar(freq * 2.003, 0, 0.15); // octave + slight detune
    sig = sig + SinOsc.ar(freq * 3.01, 0, 0.05);  // 12th harmonic
    // Slow tremolo for movement
    sig = sig * (1 + SinOsc.kr(0.15 + {0.1.rand}, {2pi.rand}) * 0.15);
    sig = BLowPass4.ar(sig, ffreq.clip(100, 16000), 0.3);
    sig = sig * env * amp;
    sig = Pan2.ar(sig, pan + SinOsc.kr(0.07, {2pi.rand}, 0.3));
    Out.ar(out, sig);
}).add;

// ── Master reverb + delay (runs on FX bus) ──────────────────────────────
SynthDef(\mreverb, { |out=0, in=0, mix=0.35, room=0.7, damp=0.5, amp=1,
                      deltime=0.375, delmix=0.2, delfb=0.4|
    var dry, wet, delayed;
    dry = In.ar(in, 2);
    // Tempo-synced ping-pong delay
    delayed = dry;
    delayed = [
        DelayC.ar(delayed[0] + (LocalIn.ar(1) * delfb), 2, deltime),
        DelayC.ar(delayed[1] + (LocalIn.ar(1) * delfb), 2, deltime * 0.75)
    ];
    LocalOut.ar(delayed[0] * delfb);
    dry = dry + (delayed * delmix);
    // Reverb
    wet = FreeVerb2.ar(dry[0], dry[1], mix, room, damp);
    ReplaceOut.ar(out, wet * amp);
}).add;

"mind-daw SynthDefs loaded.".postln;

}; // end s.waitForBoot
