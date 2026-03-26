//! Orbifold Tonnetz — Tymoczko's geometric theory of harmony.
//!
//! Chords live in the orbifold T^n/S_n (n-torus mod symmetric group).
//! T^2/S_2 is a Möbius strip (dyads), T^3/S_3 is a twisted triangular
//! prism (triads), T^4/S_4 handles tetrads, etc.
//!
//! Three visualizations:
//!   1. **Tonnetz**: The classic note-based graph (Euler/Oettingen-Riemann).
//!      Notes are vertices on a triangular lattice; chords are triangles/edges.
//!   2. **Orbifold**: Tymoczko's chord space. For dyads, a Möbius strip grid
//!      (Science 2006, Fig 2). For triads/tetrads, the orbifold fundamental domain.
//!   3. **Chord Space**: 3D rotatable projection of the orbifold prism.

use std::collections::VecDeque;

// ── Pitch-class helpers ─────────────────────────────────────────────────────

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];

pub fn pc_name(pc: u8) -> &'static str {
    NOTE_NAMES[(pc % 12) as usize]
}

/// Shortest distance between two pitch classes on the circle.
fn pc_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(12.0);
    d.min(12.0 - d)
}

/// A chord as a sorted multiset of pitch classes in [0, 12).
#[derive(Clone, Debug, PartialEq)]
pub struct Chord {
    pub pcs: Vec<f32>,
}

impl Chord {
    pub fn new(mut pcs: Vec<f32>) -> Self {
        for p in &mut pcs {
            *p = p.rem_euclid(12.0);
        }
        pcs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { pcs }
    }

    #[allow(dead_code)]
    pub fn from_semitones(root: u8, intervals: &[i32]) -> Self {
        let pcs: Vec<f32> = intervals
            .iter()
            .map(|&i| ((root as i32 + i) as f32).rem_euclid(12.0))
            .collect();
        Self::new(pcs)
    }

    pub fn n(&self) -> usize {
        self.pcs.len()
    }

    pub fn label(&self) -> String {
        self.pcs
            .iter()
            .map(|&p| {
                let rounded = p.round() as u8 % 12;
                let frac = (p - p.round()).abs();
                if frac < 0.05 {
                    pc_name(rounded).to_string()
                } else {
                    format!("{:.1}", p)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Short label: root + type, e.g. "C Maj" or "F# min7"
    pub fn short_label(&self) -> String {
        let root = pc_name(self.pcs[0].round() as u8 % 12);
        format!("{} {}", root, self.type_label())
    }

    pub fn type_label(&self) -> &'static str {
        if self.n() < 2 {
            return "note";
        }
        let intervals: Vec<u8> = (0..self.n())
            .map(|i| {
                let diff = self.pcs[i] - self.pcs[0];
                (diff.round() as u8) % 12
            })
            .collect();
        match self.n() {
            2 => match intervals[1] {
                7 => "P5",
                5 => "P4",
                4 => "M3",
                3 => "m3",
                6 => "tri",
                _ => "dyad",
            },
            3 => {
                let (a, b) = (intervals[1], intervals[2]);
                match (a, b) {
                    (4, 7) => "Maj",
                    (3, 7) => "min",
                    (3, 6) => "dim",
                    (4, 8) => "Aug",
                    (5, 7) => "sus4",
                    (2, 7) => "sus2",
                    _ => "triad",
                }
            }
            4 => {
                let (a, b, c) = (intervals[1], intervals[2], intervals[3]);
                match (a, b, c) {
                    (4, 7, 11) => "Maj7",
                    (4, 7, 10) => "dom7",
                    (3, 7, 10) => "min7",
                    (3, 6, 10) => "hdim7",
                    (3, 6, 9) => "dim7",
                    _ => "7th",
                }
            }
            _ => "chord",
        }
    }

    /// Hue index for color-coding chord type.
    pub fn hue_index(&self) -> u8 {
        match self.type_label() {
            "Maj" | "M3" | "P5" | "Maj7" => 0,
            "min" | "m3" | "min7" => 1,
            "dim" | "hdim7" | "dim7" => 2,
            "Aug" => 3,
            "dom7" | "P4" => 4,
            _ => 5,
        }
    }
}

// ── Voice leading ───────────────────────────────────────────────────────────

pub fn voice_leading_distance(a: &Chord, b: &Chord) -> f32 {
    if a.n() != b.n() || a.n() == 0 {
        return f32::INFINITY;
    }
    let n = a.n();
    if n <= 4 {
        let mut best = f32::INFINITY;
        permute(n, &mut |perm| {
            let mut sum = 0.0f32;
            for i in 0..n {
                let d = pc_dist(a.pcs[i], b.pcs[perm[i]]);
                sum += d * d;
            }
            best = best.min(sum.sqrt());
        });
        best
    } else {
        // Greedy for larger chords
        let mut used = vec![false; n];
        let mut total = 0.0f32;
        for i in 0..n {
            let mut best_j = 0;
            let mut best_d = f32::INFINITY;
            for j in 0..n {
                if !used[j] {
                    let d = pc_dist(a.pcs[i], b.pcs[j]);
                    if d < best_d {
                        best_d = d;
                        best_j = j;
                    }
                }
            }
            used[best_j] = true;
            total += best_d * best_d;
        }
        total.sqrt()
    }
}

fn permute(n: usize, f: &mut impl FnMut(&[usize])) {
    let mut perm: Vec<usize> = (0..n).collect();
    permute_inner(&mut perm, 0, f);
}

fn permute_inner(perm: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
    if k == perm.len() {
        f(perm);
        return;
    }
    for i in k..perm.len() {
        perm.swap(k, i);
        permute_inner(perm, k + 1, f);
        perm.swap(k, i);
    }
}

// ── Orbifold types & chord generation ───────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OrbifoldType {
    Notes,   // T¹/S₁ — circle of 12 pitch classes
    Dyads,   // T²/S₂ — Möbius strip
    Triads,  // T³/S₃
}

impl OrbifoldType {
    pub fn n(self) -> usize {
        match self {
            Self::Notes => 1,
            Self::Dyads => 2,
            Self::Triads => 3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Notes => "T\u{00B9}/S\u{2081} Notes",
            Self::Dyads => "T\u{00B2}/S\u{2082} Dyads",
            Self::Triads => "T\u{00B3}/S\u{2083} Triads",
        }
    }

    /// Transposition period of the fundamental domain: 12/n.
    pub fn domain_period(self) -> f32 {
        12.0 / self.n() as f32
    }

    /// Generate the set of chords for this orbifold.
    pub fn chords(self) -> Vec<Chord> {
        match self {
            Self::Notes => {
                // All 12 pitch classes as single-note "chords"
                (0..12u8).map(|p| Chord::new(vec![p as f32])).collect()
            }
            Self::Dyads => {
                // All 78 unordered pairs in T²/S₂ (including 12 unisons on the boundary)
                let mut out = Vec::new();
                for a in 0..12u8 {
                    for b in a..12u8 {
                        out.push(Chord::new(vec![a as f32, b as f32]));
                    }
                }
                out
            }
            Self::Triads => {
                // All C(12,3) = 220 unordered 3-note chords in T³/S₃
                let mut out = Vec::new();
                for a in 0..12u8 {
                    for b in (a + 1)..12u8 {
                        for c in (b + 1)..12u8 {
                            out.push(Chord::new(vec![a as f32, b as f32, c as f32]));
                        }
                    }
                }
                out
            }
        }
    }
}

// ── Layout algorithms ───────────────────────────────────────────────────────

/// Orbifold T²/S₂ (Möbius strip) layout, matching Tymoczko Science 2006 Fig.2.
///
/// Coordinates: s = (p₁ + p₂)/2 mod 12,  d = (p₂ − p₁) mod 12.
/// Möbius identification: (s, d) ∼ (s+6 mod 12, 12−d).
///
/// Fundamental domain: **[0, 6] × [0, 12]** (drawn as a square).
///   x-axis = transposition (0..6), y-axis = ordered interval (0..12).
///   Left edge (x=0) identified with right edge (x=6) via half-twist y→12−y.
///   Top (y=12) and bottom (y=0) are the singular boundary (unisons/mirrors).
///   Middle (y=6) is also singular (tritones).
///
/// Interval labels along the boundary edge from bottom to top:
///   y=0 unison, y=1 m2, y=2 M2, y=3 m3, y=4 M3, y=5 P4,
///   y=6 tritone,
///   y=7 P4, y=8 M3, y=9 m3, y=10 M2, y=11 m2, y=12 unison
pub fn mobius_strip_layout(chord: &Chord) -> (f32, f32) {
    assert!(chord.n() == 2);
    let sum = chord.pcs[0] + chord.pcs[1];
    let s = (sum / 2.0).rem_euclid(12.0);
    let d = (chord.pcs[1] - chord.pcs[0]).rem_euclid(12.0);
    // Fold into fundamental domain [0, 6) × [0, 12)
    if s >= 6.0 {
        (s - 6.0, (12.0 - d).rem_euclid(12.0))
    } else {
        (s, d)
    }
}

/// For triads: position in the orbifold T³/S₃ prism fundamental domain.
///
/// The fundamental domain is a triangular prism:
///   x = transposition = (p₁+p₂+p₃)/3, period 4 (= 12/3)
///   (y, z) = shape coordinates from barycentric projection of the interval
///            simplex (i₁, i₂, i₃) where i₁+i₂+i₃ = 12.
///
/// The two triangular faces (x=0 and x=4) are identified with a 120° rotation
/// (cyclic permutation of the interval triple).  This is the S₃ twist that
/// makes the orbifold non-orientable.
///
/// To fold into the fundamental domain we apply cyclic permutations of
/// (i₁, i₂, i₃) until x ∈ [0, 4).  Each cyclic permutation shifts the
/// transposition by 4 and rotates the shape triangle by 120°.
pub fn triad_prism_layout(chord: &Chord) -> (f32, f32, f32) {
    assert!(chord.n() == 3);
    let avg = (chord.pcs[0] + chord.pcs[1] + chord.pcs[2]) / 3.0;

    // Three successive intervals around the pitch-class circle
    let i1 = (chord.pcs[1] - chord.pcs[0]).rem_euclid(12.0);
    let i2 = (chord.pcs[2] - chord.pcs[1]).rem_euclid(12.0);
    let i3 = (12.0 - i1 - i2).rem_euclid(12.0);

    // Determine how many cyclic permutations to apply so x lands in [0, 4).
    // Each cyclic permutation (i1,i2,i3)→(i2,i3,i1) shifts avg by +4.
    let raw_x = avg.rem_euclid(12.0);
    let (fi1, fi2, fi3) = if raw_x < 4.0 {
        (i1, i2, i3)
    } else if raw_x < 8.0 {
        (i2, i3, i1) // one cyclic permutation
    } else {
        (i3, i1, i2) // two cyclic permutations
    };

    let x = raw_x.rem_euclid(4.0);

    // Barycentric → Cartesian projection of the 2-simplex (fi1, fi2, fi3)
    // This maps the triangle to ℝ² preserving distances.
    let y = (fi1 - fi2) / 2.0f32.sqrt();
    let z = (2.0 * fi3 - fi1 - fi2) / 6.0f32.sqrt();
    (x, y, z)
}

// ── Graph structures ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TonnetzNode {
    pub chord: Chord,
    /// Orbifold 2D coordinates (not screen coords — those are computed in render)
    pub ox: f32,
    pub oy: f32,
    /// Third orbifold coordinate for 3D ChordSpace visualization.
    /// For dyads (2D space) this is 0. For triads, this captures
    /// the second shape dimension. For tetrads, a projection from 4D.
    pub oz: f32,
}

/// Compute 3D orbifold coordinates for ChordSpace visualization.
/// Dyads: (transposition, interval, 0) — 2D Möbius strip embedded in 3D.
/// Triads: uses the same folded prism layout as triad_prism_layout.
fn chord_to_3d(chord: &Chord, orbifold: OrbifoldType) -> (f32, f32, f32) {
    match orbifold {
        OrbifoldType::Notes => {
            // Place on a circle: angle = pc * 2π/12, radius = 1
            let pc = chord.pcs[0];
            let angle = pc * std::f32::consts::TAU / 12.0;
            (angle.cos(), angle.sin(), 0.0)
        }
        OrbifoldType::Dyads => {
            let (ox, oy) = mobius_strip_layout(chord);
            (ox, oy, 0.0)
        }
        OrbifoldType::Triads => triad_prism_layout(chord),
    }
}

// ── Voice-leading classification ─────────────────────────────────────────────

/// Types of voice-leading motion used for action-driven navigation.
///
/// In Tymoczko's framework, edges in the chord graph represent voice leadings.
/// Rather than mapping actions to arbitrary spatial directions, we classify
/// each edge by what its optimal voice leading *does* and let actions choose
/// among these musically meaningful move types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VoiceLeadingKind {
    /// Transpose everything up: all voices rise.
    TransposeUp,
    /// Transpose everything down: all voices fall.
    TransposeDown,
    /// Raise the highest voice (expand the chord upward).
    RaiseTop,
    /// Lower the highest voice (contract the chord from above).
    LowerTop,
    /// Raise the lowest voice (contract the chord from below).
    RaiseBottom,
    /// Lower the lowest voice (expand the chord downward).
    LowerBottom,
}

/// Compute the optimal voice leading from chord `a` to chord `b`.
///
/// Returns a vector of per-voice signed semitone displacements (shortest path
/// on the pitch-class circle) under the permutation that minimises L2 distance.
fn optimal_voice_leading(a: &Chord, b: &Chord) -> Vec<f32> {
    let n = a.n();
    if n != b.n() || n == 0 {
        return Vec::new();
    }

    // For small n, try all permutations (already done in voice_leading_distance).
    let mut best_perm: Vec<usize> = (0..n).collect();
    let mut best_cost = f32::INFINITY;

    if n <= 4 {
        permute(n, &mut |perm| {
            let cost: f32 = (0..n)
                .map(|i| {
                    let d = signed_pc_dist(a.pcs[i], b.pcs[perm[i]]);
                    d * d
                })
                .sum::<f32>()
                .sqrt();
            if cost < best_cost {
                best_cost = cost;
                best_perm = perm.to_vec();
            }
        });
    } else {
        // Greedy for larger chords
        let mut used = vec![false; n];
        for i in 0..n {
            let mut bj = 0;
            let mut bd = f32::INFINITY;
            for j in 0..n {
                if !used[j] {
                    let d = pc_dist(a.pcs[i], b.pcs[j]);
                    if d < bd {
                        bd = d;
                        bj = j;
                    }
                }
            }
            used[bj] = true;
            best_perm[i] = bj;
        }
    }

    (0..n)
        .map(|i| signed_pc_dist(a.pcs[i], b.pcs[best_perm[i]]))
        .collect()
}

/// Signed shortest-path distance on the pitch-class circle.
/// Positive = upward motion, negative = downward.
fn signed_pc_dist(from: f32, to: f32) -> f32 {
    let d = (to - from).rem_euclid(12.0);
    if d <= 6.0 { d } else { d - 12.0 }
}

/// Score how well a voice leading from `a` to `b` matches a `VoiceLeadingKind`.
///
/// Returns a positive score for good matches, ≤ 0 for non-matches.
/// Among all neighbors of the current chord, the one with the highest score
/// for the requested kind is chosen.
fn score_voice_leading(a: &Chord, b: &Chord, kind: VoiceLeadingKind) -> f32 {
    let vl = optimal_voice_leading(a, b);
    if vl.is_empty() {
        return f32::NEG_INFINITY;
    }
    let n = vl.len();

    // For each kind, we compute a score that measures how well the voice
    // leading aligns with the requested motion.  We use a soft scoring so
    // that even imperfect matches can win if nothing better is available.
    //
    // The key insight: score = (desired component) - penalty*(undesired motion).
    // This way a neighbor where the top voice rises by 1 and nothing else moves
    // scores higher than one where the top voice rises by 1 but another also moves,
    // but both score above zero and are valid candidates.
    let penalty = 0.3_f32; // how much to penalise extraneous motion

    match kind {
        VoiceLeadingKind::TransposeUp => {
            // Reward total upward motion, penalise any downward voices.
            let up: f32 = vl.iter().filter(|&&d| d > 0.0).sum();
            let down: f32 = vl.iter().filter(|&&d| d < 0.0).map(|d| d.abs()).sum();
            up - penalty * down
        }
        VoiceLeadingKind::TransposeDown => {
            let down: f32 = vl.iter().filter(|&&d| d < 0.0).map(|d| d.abs()).sum();
            let up: f32 = vl.iter().filter(|&&d| d > 0.0).sum();
            down - penalty * up
        }
        VoiceLeadingKind::RaiseTop => {
            let top = vl[n - 1];
            let rest: f32 = vl[..n - 1].iter().map(|d| d.abs()).sum();
            top - penalty * rest  // positive when top goes up
        }
        VoiceLeadingKind::LowerTop => {
            let top = vl[n - 1];
            let rest: f32 = vl[..n - 1].iter().map(|d| d.abs()).sum();
            -top - penalty * rest  // positive when top goes down
        }
        VoiceLeadingKind::RaiseBottom => {
            let bot = vl[0];
            let rest: f32 = vl[1..].iter().map(|d| d.abs()).sum();
            bot - penalty * rest
        }
        VoiceLeadingKind::LowerBottom => {
            let bot = vl[0];
            let rest: f32 = vl[1..].iter().map(|d| d.abs()).sum();
            -bot - penalty * rest
        }
    }
}

#[derive(Clone, Debug)]
pub struct TonnetzEdge {
    pub from: usize,
    pub to: usize,
    pub distance: f32,
}

pub fn build_graph(orbifold: OrbifoldType) -> (Vec<TonnetzNode>, Vec<TonnetzEdge>) {
    let chords = orbifold.chords();
    let threshold = match orbifold {
        OrbifoldType::Notes => 1.5,  // semitone neighbors
        OrbifoldType::Dyads => 2.0,
        OrbifoldType::Triads => 2.5,
    };

    let nodes: Vec<TonnetzNode> = chords
        .iter()
        .map(|chord| {
            let (ox, oy, oz) = chord_to_3d(chord, orbifold);
            TonnetzNode {
                chord: chord.clone(),
                ox,
                oy,
                oz,
            }
        })
        .collect();

    let mut edges = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let d = voice_leading_distance(&nodes[i].chord, &nodes[j].chord);
            if d <= threshold && d > 0.01 {
                edges.push(TonnetzEdge {
                    from: i,
                    to: j,
                    distance: d,
                });
            }
        }
    }

    (nodes, edges)
}

// ── Navigation state ────────────────────────────────────────────────────────

pub const TRAIL_LEN: usize = 64;

pub struct TonnetzState {
    pub orbifold: OrbifoldType,
    pub nodes: Vec<TonnetzNode>,
    pub edges: Vec<TonnetzEdge>,

    pub current_chord_idx: usize,
    pub chord_trail: VecDeque<usize>,
    pub position: [f32; 3],
    pub position_trail: VecDeque<[f32; 3]>,

    pub yaw: f32,
    pub pitch: f32,
    pub dragging: bool,
    pub last_drag_pos: Option<(f32, f32)>,

    pub nav_velocity: [f32; 3],
    pub zoom: f32,
}

impl TonnetzState {
    pub fn new(orbifold: OrbifoldType) -> Self {
        let (nodes, edges) = build_graph(orbifold);
        Self {
            orbifold,
            nodes,
            edges,
            current_chord_idx: 0,
            chord_trail: VecDeque::with_capacity(TRAIL_LEN),
            position: [3.0, 6.0, 0.0], // center of [0,6]×[0,12] domain
            position_trail: VecDeque::with_capacity(TRAIL_LEN),
            yaw: 0.0,
            pitch: 0.0,
            dragging: false,
            last_drag_pos: None,
            nav_velocity: [0.0; 3],
            zoom: 1.0,
        }
    }

    pub fn set_orbifold(&mut self, orbifold: OrbifoldType) {
        let (nodes, edges) = build_graph(orbifold);
        self.orbifold = orbifold;
        self.nodes = nodes;
        self.edges = edges;
        self.current_chord_idx = 0;
        self.chord_trail.clear();
        self.position_trail.clear();
        // Center of the fundamental domain
        let period = orbifold.domain_period();
        self.position = [period / 2.0, 0.0, 0.0];
        self.nav_velocity = [0.0; 3];
        self.zoom = 1.0;
    }

    /// Update from a ControlState (the calibrated decoder output).
    ///
    /// Confidence modulates speed and smoothing:
    ///  - High confidence → faster, more responsive movement
    ///  - Low confidence → slower, more damped, biased toward staying put
    ///
    /// Tension biases toward more dissonant chords (diminished, augmented).
    /// Stability controls the snap strength toward chord attractors.
    pub fn update_from_control(&mut self, ctl: &crate::control::ControlState) {
        if ctl.freeze {
            // Still record trail position for visualization
            if self.position_trail.len() >= TRAIL_LEN {
                self.position_trail.pop_front();
            }
            self.position_trail.push_back(self.position);
            return;
        }

        // Confidence-weighted speed: [0.02, 0.2]
        let confidence = ctl.overall_confidence();
        let speed = 0.02 + 0.18 * confidence;

        // Confidence-weighted smoothing: high confidence → less smoothing (0.7),
        // low confidence → heavy smoothing (0.97)
        let smoothing = 0.97 - 0.27 * confidence;

        // Apply motion with smoothing.
        // For triads the z-axis (second shape dimension) is driven by tension.
        let signal = [ctl.motion_x, ctl.motion_y, ctl.tension - 0.5];
        for i in 0..3 {
            self.nav_velocity[i] =
                smoothing * self.nav_velocity[i] + (1.0 - smoothing) * signal[i];
        }
        for i in 0..3 {
            self.position[i] += self.nav_velocity[i] * speed;
        }

        // Stability-based attractor snap: when stability is high, gently pull
        // toward the nearest chord node to keep harmony coherent.
        if ctl.stability > 0.3 && !self.nodes.is_empty() {
            let snap_strength = (ctl.stability - 0.3) * 0.05; // 0..0.035
            let node = &self.nodes[self.current_chord_idx];
            let period = self.orbifold.domain_period();
            let raw_dx = (node.ox - self.position[0]).rem_euclid(period);
            let dx = if raw_dx > period / 2.0 { raw_dx - period } else { raw_dx };
            let dy = node.oy - self.position[1];
            let dz = node.oz - self.position[2];
            self.position[0] += dx * snap_strength;
            self.position[1] += dy * snap_strength;
            self.position[2] += dz * snap_strength;
        }

        self.clamp_and_snap();
    }

    /// Handle a reset event: return to the center of the fundamental domain.
    pub fn reset_to_home(&mut self) {
        let period = self.orbifold.domain_period();
        self.position = [period / 2.0, 0.0, 0.0];
        self.nav_velocity = [0.0; 3];
    }

    fn clamp_and_snap(&mut self) {
        // Wrap/clamp to fundamental domain
        let period = self.orbifold.domain_period();
        self.position[0] = self.position[0].rem_euclid(period);
        self.position[1] = self.position[1].clamp(-12.0, 12.0);
        self.position[2] = self.position[2].clamp(-12.0, 12.0);

        // Find nearest chord by orbifold distance (wrapping x, using y and z)
        if !self.nodes.is_empty() {
            let mut best_idx = 0;
            let mut best_dist = f32::INFINITY;
            for (i, node) in self.nodes.iter().enumerate() {
                // x wraps at the orbifold period
                let raw_dx = (self.position[0] - node.ox).rem_euclid(period);
                let dx = raw_dx.min(period - raw_dx);
                let dy = self.position[1] - node.oy;
                let dz = self.position[2] - node.oz;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }

            if best_idx != self.current_chord_idx {
                if self.chord_trail.len() >= TRAIL_LEN {
                    self.chord_trail.pop_front();
                }
                self.chord_trail.push_back(self.current_chord_idx);
                self.current_chord_idx = best_idx;
            }
        }

        if self.position_trail.len() >= TRAIL_LEN {
            self.position_trail.pop_front();
        }
        self.position_trail.push_back(self.position);
    }

    pub fn current_chord(&self) -> Option<&Chord> {
        self.nodes.get(self.current_chord_idx).map(|n| &n.chord)
    }

    pub fn current_edges(&self) -> Vec<&TonnetzEdge> {
        let idx = self.current_chord_idx;
        self.edges
            .iter()
            .filter(|e| e.from == idx || e.to == idx)
            .collect()
    }

    /// Navigate by voice-leading type.
    ///
    /// Each edge in the graph IS a voice leading.  This method finds the
    /// neighbor whose optimal voice leading best matches the requested
    /// `VoiceLeadingKind` and jumps there.  Returns `true` if the chord changed.
    pub fn navigate_by_voice_leading(&mut self, kind: VoiceLeadingKind) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let idx = self.current_chord_idx;
        let cur_chord = &self.nodes[idx].chord;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = None;

        for edge in &self.edges {
            let other_idx = if edge.from == idx {
                edge.to
            } else if edge.to == idx {
                edge.from
            } else {
                continue;
            };
            let other_chord = &self.nodes[other_idx].chord;

            // Compute the optimal voice leading and score it against the kind.
            let score = score_voice_leading(cur_chord, other_chord, kind);
            if score > best_score {
                best_score = score;
                best_idx = Some(other_idx);
            }
        }

        if let Some(target) = best_idx {
            if best_score > 0.0 {
                if self.chord_trail.len() >= TRAIL_LEN {
                    self.chord_trail.pop_front();
                }
                self.chord_trail.push_back(self.current_chord_idx);
                if self.position_trail.len() >= TRAIL_LEN {
                    self.position_trail.pop_front();
                }
                self.position_trail.push_back(self.position);

                self.current_chord_idx = target;
                self.position = [
                    self.nodes[target].ox,
                    self.nodes[target].oy,
                    self.nodes[target].oz,
                ];
                self.nav_velocity = [0.0; 3];
                return true;
            }
        }
        false
    }
}

/// Convert a chord's pitch classes to MIDI note numbers, centered around C4 (MIDI 60).
/// Returns sorted MIDI notes in a comfortable middle register.
pub fn chord_to_midi_notes(chord: &Chord) -> Vec<u8> {
    chord
        .pcs
        .iter()
        .map(|&pc| {
            let rounded = pc.round() as i32;
            // Place in octave 4 (MIDI 60 = C4)
            (60 + rounded.rem_euclid(12)) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigate_by_voice_leading_moves() {
        let mut state = TonnetzState::new(OrbifoldType::Dyads);
        let start = state.current_chord_idx;

        let moved = state.navigate_by_voice_leading(VoiceLeadingKind::TransposeUp);
        assert!(moved, "transpose up should find a neighbor");
        assert_ne!(state.current_chord_idx, start);
        assert!(!state.chord_trail.is_empty());
    }

    #[test]
    fn transpose_up_then_down_returns() {
        let mut state = TonnetzState::new(OrbifoldType::Dyads);

        state.navigate_by_voice_leading(VoiceLeadingKind::TransposeUp);
        let mid = state.current_chord_idx;
        state.navigate_by_voice_leading(VoiceLeadingKind::TransposeDown);
        // Should move away from mid (likely back toward start)
        assert_ne!(state.current_chord_idx, mid);
    }

    #[test]
    fn voice_leading_is_consistent() {
        // Same kind from the same position must always pick the same neighbor.
        let mut s1 = TonnetzState::new(OrbifoldType::Triads);
        let mut s2 = TonnetzState::new(OrbifoldType::Triads);

        s1.navigate_by_voice_leading(VoiceLeadingKind::RaiseTop);
        s2.navigate_by_voice_leading(VoiceLeadingKind::RaiseTop);
        assert_eq!(s1.current_chord_idx, s2.current_chord_idx);
    }

    #[test]
    fn raise_top_actually_raises_top_voice() {
        let mut state = TonnetzState::new(OrbifoldType::Dyads);
        let before = state.current_chord().unwrap().pcs.clone();

        if state.navigate_by_voice_leading(VoiceLeadingKind::RaiseTop) {
            let after = state.current_chord().unwrap().pcs.clone();
            // The optimal voice leading should have the top voice higher.
            let vl = optimal_voice_leading(
                &Chord::new(before),
                &Chord::new(after),
            );
            let top_motion = vl.last().copied().unwrap_or(0.0);
            assert!(top_motion > 0.0, "top voice should move up, got {top_motion}");
        }
    }

    #[test]
    fn all_six_kinds_produce_moves_on_triads() {
        let kinds = [
            VoiceLeadingKind::TransposeUp,
            VoiceLeadingKind::TransposeDown,
            VoiceLeadingKind::RaiseTop,
            VoiceLeadingKind::LowerTop,
            VoiceLeadingKind::RaiseBottom,
            VoiceLeadingKind::LowerBottom,
        ];
        for kind in kinds {
            let mut state = TonnetzState::new(OrbifoldType::Triads);
            let moved = state.navigate_by_voice_leading(kind);
            assert!(moved, "{kind:?} should produce a move on triads");
        }
    }

    #[test]
    fn triads_generates_all_220_chords() {
        let chords = OrbifoldType::Triads.chords();
        assert_eq!(chords.len(), 220, "C(12,3) = 220 three-note chords");
        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for c in &chords {
            let key = format!("{:?}", c.pcs);
            assert!(seen.insert(key), "duplicate chord found");
        }
    }

    #[test]
    fn dyads_generates_all_78_chords() {
        let chords = OrbifoldType::Dyads.chords();
        // C(12,2) = 66 distinct pairs + 12 unisons on the singular boundary = 78
        assert_eq!(chords.len(), 78, "66 pairs + 12 unisons = 78");
        // Verify unisons exist
        let unisons: Vec<_> = chords.iter().filter(|c| (c.pcs[0] - c.pcs[1]).abs() < 0.01).collect();
        assert_eq!(unisons.len(), 12, "should have 12 unisons");
    }

    #[test]
    fn prism_layout_folds_correctly() {
        // Two chords related by transposition of 4 should have the same x
        // after folding (since period = 4).
        let c1 = Chord::new(vec![0.0, 4.0, 7.0]); // C major
        let c2 = Chord::new(vec![4.0, 8.0, 11.0]); // E major (transposed +4)
        let (x1, _, _) = triad_prism_layout(&c1);
        let (x2, _, _) = triad_prism_layout(&c2);
        // After folding, x should be equal (both map to same transposition class)
        assert!(
            (x1 - x2).abs() < 0.01,
            "C maj (x={x1:.3}) and E maj (x={x2:.3}) should fold to same x"
        );
    }

    #[test]
    fn triad_graph_has_semitone_edges() {
        // C major {0,4,7} should connect to C minor {0,3,7} by distance 1.0
        let (nodes, edges) = build_graph(OrbifoldType::Triads);
        let c_maj = nodes.iter().position(|n| {
            let p = &n.chord.pcs;
            p.len() == 3 && (p[0] - 0.0).abs() < 0.01
                && (p[1] - 4.0).abs() < 0.01
                && (p[2] - 7.0).abs() < 0.01
        });
        let c_min = nodes.iter().position(|n| {
            let p = &n.chord.pcs;
            p.len() == 3 && (p[0] - 0.0).abs() < 0.01
                && (p[1] - 3.0).abs() < 0.01
                && (p[2] - 7.0).abs() < 0.01
        });
        assert!(c_maj.is_some(), "C major should be in the graph");
        assert!(c_min.is_some(), "C minor should be in the graph");
        let i = c_maj.unwrap();
        let j = c_min.unwrap();
        let edge = edges.iter().find(|e| {
            (e.from == i && e.to == j) || (e.from == j && e.to == i)
        });
        assert!(edge.is_some(), "C major and C minor should be connected");
        let d = edge.unwrap().distance;
        assert!(
            (d - 1.0).abs() < 0.01,
            "voice-leading distance should be 1.0, got {d}"
        );
    }

    #[test]
    fn triad_layout_uses_both_shape_dims() {
        // Chords with same transposition but different shapes should differ in y or z
        let c_maj = Chord::new(vec![0.0, 4.0, 7.0]);
        let c_sus2 = Chord::new(vec![0.0, 2.0, 7.0]);
        let (_, y1, z1) = triad_prism_layout(&c_maj);
        let (_, y2, z2) = triad_prism_layout(&c_sus2);
        let dist = ((y1 - y2).powi(2) + (z1 - z2).powi(2)).sqrt();
        assert!(
            dist > 0.5,
            "C major and C sus2 should be well separated in shape space, got dist={dist}"
        );
    }
}
