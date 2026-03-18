//! Dual-mode instrument interface.
//!
//! **Arrange** mode — cue synthesis layers in/out (drums, bass, arp).
//! Continuous EEG signals navigate the Tonnetz and shape global energy.
//!
//! **Edit** mode — focused on one module, continuous signals remap to
//! that module's sonic parameters (filter, speed, waveform, etc.).
//!
//! **Mode toggle**: jaw double-clench switches Arrange ↔ Edit.
//! **Module cycle**: jaw single-clench in Edit mode cycles the focused module.

use super::mapping::Module;

// ── InstrumentMode ────────────────────────────────────────────────────────────

/// The current operating mode of the neural instrument interface.
#[derive(Debug, Clone, PartialEq)]
pub enum InstrumentMode {
    /// Cue layers in/out. EEG navigates Tonnetz chord space.
    Arrange,
    /// Sculpt the sonic quality of the focused synthesis module.
    Edit { focused: Module },
}

impl Default for InstrumentMode {
    fn default() -> Self {
        Self::Arrange
    }
}

impl InstrumentMode {
    /// Short label shown in the mode bar.
    pub fn label(&self) -> String {
        match self {
            Self::Arrange            => "ARRANGE".to_string(),
            Self::Edit { focused }   => format!("EDIT  {}", focused.label()),
        }
    }

    /// True when in Arrange mode.
    pub fn is_arrange(&self) -> bool {
        matches!(self, Self::Arrange)
    }

    /// True when in Edit mode.
    pub fn is_edit(&self) -> bool {
        matches!(self, Self::Edit { .. })
    }

    /// Returns the focused module, if in Edit mode.
    pub fn focused_module(&self) -> Option<&Module> {
        match self {
            Self::Edit { focused } => Some(focused),
            Self::Arrange => None,
        }
    }

    /// Jaw-double: toggle Arrange ↔ Edit{Tonnetz}.
    pub fn toggle(&self) -> Self {
        match self {
            Self::Arrange => Self::Edit { focused: Module::Tonnetz },
            Self::Edit { .. } => Self::Arrange,
        }
    }

    /// Jaw-single in Edit mode: cycle the focused module forward.
    /// No-op in Arrange mode.
    pub fn cycle_focus(&self) -> Self {
        match self {
            Self::Arrange => self.clone(),
            Self::Edit { focused } => {
                let all = Module::all();
                let i = all.iter().position(|m| m == focused).unwrap_or(0);
                Self::Edit { focused: all[(i + 1) % all.len()].clone() }
            }
        }
    }

    /// Colour hue for the mode indicator (0–1 HSL hue).
    /// Arrange = teal (0.50), Edit = amber (0.11).
    pub fn hue(&self) -> f32 {
        match self {
            Self::Arrange  => 0.50,
            Self::Edit { .. } => 0.11,
        }
    }
}

// ── LayerState ────────────────────────────────────────────────────────────────

/// Which synthesis layers are currently active (cued in).
#[derive(Debug, Clone, Default)]
pub struct LayerState {
    pub drums_on: bool,
    pub bass_on:  bool,
    pub arp_on:   bool,
    pub pad_on:   bool,
}

impl LayerState {
    /// Toggle drums; returns new state for OSC send decision.
    pub fn toggle_drums(&mut self) -> bool {
        self.drums_on = !self.drums_on;
        self.drums_on
    }
    pub fn toggle_bass(&mut self) -> bool {
        self.bass_on = !self.bass_on;
        self.bass_on
    }
    pub fn toggle_arp(&mut self) -> bool {
        self.arp_on = !self.arp_on;
        self.arp_on
    }
    pub fn toggle_pad(&mut self) -> bool {
        self.pad_on = !self.pad_on;
        self.pad_on
    }
}
