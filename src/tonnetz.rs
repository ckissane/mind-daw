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
    Dyads,   // T²/S₂ — Möbius strip
    Triads,  // T³/S₃
    Tetrads, // T⁴/S₄
}

impl OrbifoldType {
    pub fn n(self) -> usize {
        match self {
            Self::Dyads => 2,
            Self::Triads => 3,
            Self::Tetrads => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Dyads => "T\u{00B2}/S\u{2082} Dyads",
            Self::Triads => "T\u{00B3}/S\u{2083} Triads",
            Self::Tetrads => "T\u{2074}/S\u{2084} Tetrads",
        }
    }

    /// Transposition period of the fundamental domain: 12/n.
    pub fn domain_period(self) -> f32 {
        12.0 / self.n() as f32
    }

    /// Generate the set of chords for this orbifold.
    pub fn chords(self) -> Vec<Chord> {
        match self {
            Self::Dyads => {
                // All 66 unordered pairs in T²/S₂
                let mut out = Vec::new();
                for a in 0..12u8 {
                    for b in a..12u8 {
                        if a == b { continue; } // skip unisons
                        out.push(Chord::new(vec![a as f32, b as f32]));
                    }
                }
                out
            }
            Self::Triads => {
                // Major and minor triads (24 total) — the core of tonal harmony
                let mut out = Vec::new();
                for root in 0..12u8 {
                    out.push(Chord::from_semitones(root, &[0, 4, 7])); // major
                    out.push(Chord::from_semitones(root, &[0, 3, 7])); // minor
                }
                // Diminished (12) and augmented (4)
                for root in 0..12u8 {
                    out.push(Chord::from_semitones(root, &[0, 3, 6]));
                }
                for root in (0..12u8).step_by(4) {
                    out.push(Chord::from_semitones(root, &[0, 4, 8]));
                }
                out
            }
            Self::Tetrads => {
                let mut out = Vec::new();
                for root in 0..12u8 {
                    out.push(Chord::from_semitones(root, &[0, 4, 7, 11])); // Maj7
                    out.push(Chord::from_semitones(root, &[0, 4, 7, 10])); // dom7
                    out.push(Chord::from_semitones(root, &[0, 3, 7, 10])); // min7
                    out.push(Chord::from_semitones(root, &[0, 3, 6, 10])); // hdim7
                }
                for root in (0..12u8).step_by(3) {
                    out.push(Chord::from_semitones(root, &[0, 3, 6, 9])); // dim7
                }
                out
            }
        }
    }
}

// ── Layout algorithms ───────────────────────────────────────────────────────

/// Tonnetz note layout: place the 12 pitch classes on a triangular lattice.
/// Horizontal axis = major thirds (+4), NE diagonal = minor thirds (+3),
/// NW diagonal = perfect fifths (+7). Returns (x, y) for each of 12 PCs.
///
/// We tile a few periods so chords don't wrap weirdly.
pub fn tonnetz_note_positions() -> Vec<(u8, f32, f32)> {
    // The classic Tonnetz is a planar tiling. We generate a grid where:
    //   moving right = +4 semitones (major third)
    //   moving up-right = +3 semitones (minor third)
    //   moving up-left = +7 semitones (perfect fifth)
    //
    // We lay out a hex-ish grid and label each position with its pitch class.
    let mut notes = Vec::new();
    let dx = 60.0f32; // horizontal spacing (major third)
    let dy = 52.0f32; // vertical spacing

    // Generate a grid of pitch classes
    for row in -2i32..=3 {
        for col in -3i32..=4 {
            // Each row shifts by +3 semitones (minor third up from previous row)
            // Each col shifts by +4 semitones (major third right)
            let pc = ((row * 3 + col * 4) as i32).rem_euclid(12) as u8;
            let x = col as f32 * dx + row as f32 * dx * 0.5;
            let y = -(row as f32) * dy; // negative so up is positive
            notes.push((pc, x, y));
        }
    }
    notes
}

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
/// The fundamental domain is a triangular prism with:
///   x = transposition = (p₁+p₂+p₃)/3, period 4 (= 12/3)
///   y = shape coordinate from barycentric projection of interval simplex
///
/// The two triangular faces (x=0 and x=4) are identified with a 120° rotation
/// (cyclic permutation of voices). The rectangular faces are identified with
/// orientation reversal (transposition of two voices).
///
/// We fold into the fundamental domain [0, 4) × [y_min, y_max] analogously
/// to how mobius_strip_layout folds dyads into [0, 6) × [0, 12).
pub fn triad_prism_layout(chord: &Chord) -> (f32, f32) {
    assert!(chord.n() == 3);
    let avg = (chord.pcs[0] + chord.pcs[1] + chord.pcs[2]) / 3.0;
    let x = avg.rem_euclid(4.0); // period 4 for triads

    // Three successive intervals around the chord
    let i1 = (chord.pcs[1] - chord.pcs[0]).rem_euclid(12.0);
    let i2 = (chord.pcs[2] - chord.pcs[1]).rem_euclid(12.0);
    // Barycentric projection of (i1, i2, i3) simplex onto 1D shape axis
    let y = (i1 - i2) / 2.0f32.sqrt();
    (x, y)
}

/// Legacy alias — kept for compatibility.
pub fn triad_orbifold_layout(chord: &Chord) -> (f32, f32) {
    triad_prism_layout(chord)
}

/// For tetrads: position in T⁴/S₄.
pub fn tetrad_orbifold_layout(chord: &Chord) -> (f32, f32) {
    assert!(chord.n() == 4);
    let avg = chord.pcs.iter().sum::<f32>() / 4.0;
    let x = avg.rem_euclid(12.0);

    let even = 3.0; // 12/4
    let mut dev = 0.0f32;
    for i in 0..4 {
        let next = (i + 1) % 4;
        let interval = (chord.pcs[next] - chord.pcs[i]).rem_euclid(12.0);
        dev += (interval - even).powi(2);
    }
    let y = dev.sqrt();
    (x, y)
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
/// Triads: (transposition, shape_y, shape_z) — 3D twisted prism.
///   Shape coordinates come from barycentric projection of the interval simplex.
/// Tetrads: (transposition, shape_y, shape_z) — projected from 4D.
fn chord_to_3d(chord: &Chord, orbifold: OrbifoldType) -> (f32, f32, f32) {
    match orbifold {
        OrbifoldType::Dyads => {
            let (ox, oy) = mobius_strip_layout(chord);
            (ox, oy, 0.0)
        }
        OrbifoldType::Triads => {
            let avg = (chord.pcs[0] + chord.pcs[1] + chord.pcs[2]) / 3.0;
            let x = avg.rem_euclid(12.0);
            // Three intervals around the chord
            let i1 = (chord.pcs[1] - chord.pcs[0]).rem_euclid(12.0);
            let i2 = (chord.pcs[2] - chord.pcs[1]).rem_euclid(12.0);
            let i3 = (12.0 - i1 - i2).rem_euclid(12.0);
            // Barycentric → 2D projection of the interval simplex
            // (i1, i2, i3) sum to 12, live on a 2-simplex
            let y = (i1 - i2) / 2.0f32.sqrt();
            let z = (2.0 * i3 - i1 - i2) / 6.0f32.sqrt();
            (x, y, z)
        }
        OrbifoldType::Tetrads => {
            let avg = chord.pcs.iter().sum::<f32>() / 4.0;
            let x = avg.rem_euclid(12.0);
            // Four intervals
            let mut intervals = [0.0f32; 4];
            for k in 0..4 {
                intervals[k] = (chord.pcs[(k + 1) % 4] - chord.pcs[k]).rem_euclid(12.0);
            }
            // Project 3-simplex (4 intervals summing to 12) to 3D
            // Using standard simplex basis vectors
            let y = (intervals[0] - intervals[1]) / 2.0f32.sqrt();
            let z = (intervals[0] + intervals[1] - 2.0 * intervals[2]) / 6.0f32.sqrt();
            // Third shape dimension (we project to 2D for display, using z)
            (x, y, z)
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
        OrbifoldType::Dyads => 2.0,
        OrbifoldType::Triads => 2.5,
        OrbifoldType::Tetrads => 3.0,
    };

    let nodes: Vec<TonnetzNode> = chords
        .iter()
        .map(|chord| {
            let (ox, oy) = match orbifold {
                OrbifoldType::Dyads => mobius_strip_layout(chord),
                OrbifoldType::Triads => triad_prism_layout(chord),
                OrbifoldType::Tetrads => tetrad_orbifold_layout(chord),
            };
            let (_, _, oz) = chord_to_3d(chord, orbifold);
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
    pub note_positions: Vec<(u8, f32, f32)>,

    pub current_chord_idx: usize,
    pub chord_trail: VecDeque<usize>,
    pub position: [f32; 3],
    pub position_trail: VecDeque<[f32; 3]>,

    pub viz_mode: VizMode,
    pub yaw: f32,
    pub pitch: f32,
    pub dragging: bool,
    pub last_drag_pos: Option<(f32, f32)>,

    pub nav_velocity: [f32; 3],
    pub nav_smoothing: f32,
    pub zoom: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VizMode {
    Tonnetz,    // Classic note-based tonnetz lattice
    Orbifold,   // Tymoczko orbifold fundamental domain
    ChordSpace, // 3D rotatable projection
}

impl VizMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tonnetz => "Tonnetz",
            Self::Orbifold => "Orbifold",
            Self::ChordSpace => "Chord Space",
        }
    }
}

impl TonnetzState {
    pub fn new(orbifold: OrbifoldType) -> Self {
        let (nodes, edges) = build_graph(orbifold);
        Self {
            orbifold,
            nodes,
            edges,
            note_positions: tonnetz_note_positions(),
            current_chord_idx: 0,
            chord_trail: VecDeque::with_capacity(TRAIL_LEN),
            position: [3.0, 6.0, 0.0], // center of [0,6]×[0,12] domain
            position_trail: VecDeque::with_capacity(TRAIL_LEN),
            viz_mode: VizMode::Tonnetz,
            yaw: 0.0,
            pitch: 0.0,
            dragging: false,
            last_drag_pos: None,
            nav_velocity: [0.0; 3],
            nav_smoothing: 0.85,
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

    pub fn update_from_brain(&mut self, brain_signal: [f32; 3]) {
        for i in 0..3 {
            self.nav_velocity[i] =
                self.nav_smoothing * self.nav_velocity[i]
                    + (1.0 - self.nav_smoothing) * brain_signal[i];
        }
        let speed = 0.15;
        for i in 0..3 {
            self.position[i] += self.nav_velocity[i] * speed;
        }
        // Wrap/clamp to fundamental domain
        let period = self.orbifold.domain_period();
        self.position[0] = self.position[0].rem_euclid(period);
        self.position[1] = self.position[1].clamp(-12.0, 12.0);
        self.position[2] = self.position[2].clamp(-12.0, 12.0);

        // Find nearest chord by orbifold distance
        if !self.nodes.is_empty() {
            let mut best_idx = 0;
            let mut best_dist = f32::INFINITY;
            for (i, node) in self.nodes.iter().enumerate() {
                // x wraps at the orbifold period
                let raw_dx = (self.position[0] - node.ox).rem_euclid(period);
                let dx = raw_dx.min(period - raw_dx);
                let dy = self.position[1] - node.oy;
                let d = (dx * dx + dy * dy).sqrt();
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
}

/// Prepare rendering data for the tonnetz note-based view.
/// Returns: note_dots (pc, x, y), chord_triangles (indices into note_dots, is_current, hue_idx)
pub fn prepare_tonnetz_render(
    state: &TonnetzState,
) -> (
    Vec<(u8, f32, f32)>,                  // note positions
    Vec<(Vec<usize>, bool, u8, String)>,  // chord shapes: (note indices, is_current, hue_idx, label)
) {
    let notes = &state.note_positions;

    // For each chord, find which note positions match its pitch classes
    let mut chord_shapes = Vec::new();
    for (ci, node) in state.nodes.iter().enumerate() {
        let is_current = ci == state.current_chord_idx;

        // Find a connected cluster of note positions that form this chord.
        // We need one note position per pitch class, and they should be close together.
        let pcs: Vec<u8> = node.chord.pcs.iter().map(|&p| p.round() as u8 % 12).collect();

        // For each pc, collect candidate note indices
        let candidates: Vec<Vec<usize>> = pcs
            .iter()
            .map(|&pc| {
                notes
                    .iter()
                    .enumerate()
                    .filter(|(_, (n, _, _))| *n == pc)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();

        // Find the combination of candidates that minimizes total distance
        if candidates.iter().any(|c| c.is_empty()) {
            continue;
        }

        let best = find_closest_cluster(notes, &candidates);
        chord_shapes.push((
            best,
            is_current,
            node.chord.hue_index(),
            node.chord.short_label(),
        ));
    }

    (notes.clone(), chord_shapes)
}

/// Find the combination of note positions (one per voice) that minimizes spread.
fn find_closest_cluster(
    notes: &[(u8, f32, f32)],
    candidates: &[Vec<usize>],
) -> Vec<usize> {
    if candidates.len() == 1 {
        return vec![candidates[0][0]];
    }

    // For triads/dyads (small n), brute-force the combinations
    // For efficiency, anchor on the first voice's candidates and find nearest others
    let mut best_combo = vec![0usize; candidates.len()];
    let mut best_spread = f32::INFINITY;

    for &anchor in &candidates[0] {
        let (ax, ay) = (notes[anchor].1, notes[anchor].2);
        let mut combo = vec![anchor];
        let mut total_dist = 0.0f32;

        for voice in 1..candidates.len() {
            let mut nearest = candidates[voice][0];
            let mut nearest_d = f32::INFINITY;
            for &idx in &candidates[voice] {
                let dx = notes[idx].1 - ax;
                let dy = notes[idx].2 - ay;
                let d = dx * dx + dy * dy;
                if d < nearest_d {
                    nearest_d = d;
                    nearest = idx;
                }
            }
            combo.push(nearest);
            total_dist += nearest_d;
        }

        if total_dist < best_spread {
            best_spread = total_dist;
            best_combo = combo;
        }
    }
    best_combo
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

/// Extract a 3-component brain navigation signal from EEG features.
pub fn eeg_to_nav_signal(features: &[f32]) -> [f32; 3] {
    if features.is_empty() {
        return [0.0; 3];
    }

    let bins_per_ch = 32.min(features.len());
    let n_ch = (features.len() / bins_per_ch).min(64);
    if n_ch == 0 || bins_per_ch < 16 {
        return [0.0; 3];
    }

    let ch_limit = n_ch.min(8);
    let mut theta_power = 0.0f32;
    let mut alpha_power = 0.0f32;
    let mut beta_power = 0.0f32;
    let mut gamma_power = 0.0f32;

    for ch in 0..ch_limit {
        let offset = ch * bins_per_ch;
        for b in 1..=2 {
            theta_power += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        for b in 2..=3 {
            alpha_power += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        for b in 3..=6 {
            beta_power += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
        for b in 7..=17.min(bins_per_ch - 1) {
            gamma_power += features.get(offset + b).copied().unwrap_or(0.0).abs();
        }
    }

    let norm = (theta_power + alpha_power + beta_power + gamma_power).max(1e-6);
    let alpha_beta = (alpha_power - beta_power) / norm;
    let theta_norm = (theta_power / norm - 0.25) * 4.0;
    let gamma_norm = (gamma_power / norm - 0.1) * 5.0;

    [
        alpha_beta.clamp(-1.0, 1.0),
        theta_norm.clamp(-1.0, 1.0),
        gamma_norm.clamp(-1.0, 1.0),
    ]
}
