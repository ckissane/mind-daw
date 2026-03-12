//! Structured session logging for EEG-to-music sessions.
//!
//! Logs decoder outputs, orbifold coordinates, sounding chords, and events
//! with timestamps for post-session analysis and mapping refinement.

use crate::calibration::BandPowers;
use crate::control::ControlState;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

/// A single timestamped log entry.
#[derive(Serialize)]
struct LogEntry {
    /// Seconds since session start.
    t: f32,
    /// Entry type.
    kind: &'static str,
    /// JSON payload.
    data: serde_json::Value,
}

/// Session logger — writes JSONL (one JSON object per line).
pub struct SessionLog {
    writer: Option<BufWriter<File>>,
    start: Instant,
}

impl SessionLog {
    /// Create a new session log. Logs are stored in ~/.mind-daw/sessions/.
    pub fn new(user_name: &str) -> Self {
        let mut dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        dir.push(".mind-daw");
        dir.push("sessions");
        let _ = std::fs::create_dir_all(&dir);

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let safe_name: String = user_name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let filename = format!("{}_{}.jsonl", safe_name, ts);
        let path = dir.join(&filename);

        let writer = File::create(&path)
            .map(BufWriter::new)
            .map_err(|e| eprintln!("Failed to create session log {:?}: {}", path, e))
            .ok();

        SessionLog {
            writer,
            start: Instant::now(),
        }
    }

    fn write_entry(&mut self, kind: &'static str, data: serde_json::Value) {
        let t = self.start.elapsed().as_secs_f32();
        let Some(ref mut w) = self.writer else { return };
        let entry = LogEntry {
            t,
            kind,
            data,
        };
        if let Ok(json) = serde_json::to_string(&entry) {
            let _ = writeln!(w, "{}", json);
        }
    }

    /// Log band power features.
    pub fn log_features(&mut self, bands: &BandPowers) {
        self.write_entry(
            "features",
            serde_json::json!({
                "delta": bands.delta,
                "theta": bands.theta,
                "alpha": bands.alpha,
                "beta": bands.beta,
                "gamma": bands.gamma,
            }),
        );
    }

    /// Log decoder control state output.
    pub fn log_control(&mut self, state: &ControlState) {
        self.write_entry(
            "control",
            serde_json::json!({
                "motion_x": state.motion_x,
                "motion_y": state.motion_y,
                "tension": state.tension,
                "stability": state.stability,
                "freeze": state.freeze,
                "confidence_c": state.confidence_continuous,
                "confidence_d": state.confidence_discrete,
            }),
        );
    }

    /// Log orbifold position.
    pub fn log_position(&mut self, pos: [f32; 3]) {
        self.write_entry(
            "position",
            serde_json::json!({
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
            }),
        );
    }

    /// Log chord change.
    pub fn log_chord(&mut self, chord_label: &str, midi_notes: &[u8]) {
        self.write_entry(
            "chord",
            serde_json::json!({
                "label": chord_label,
                "midi": midi_notes,
            }),
        );
    }

    /// Log calibration metadata.
    pub fn log_calibration(&mut self, profile_name: &str) {
        self.write_entry(
            "calibration",
            serde_json::json!({ "profile": profile_name }),
        );
    }

    /// Flush buffered writes.
    pub fn flush(&mut self) {
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
        }
    }
}

impl Drop for SessionLog {
    fn drop(&mut self) {
        self.flush();
    }
}
