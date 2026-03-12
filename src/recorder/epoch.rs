use serde::{Deserialize, Serialize};

/// One labelled EEG epoch captured around a stimulus event.
/// 300 samples = 1 second at 300 Hz (60 pre + 240 post).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StimulusEpoch {
    /// Unique epoch identifier (UUID v4 as string).
    pub id: String,
    /// Stimulus class label, e.g. "blink_left".
    pub label: String,
    /// Unix timestamp (seconds) of the stimulus onset (ring-buffer capture moment).
    pub timestamp: f64,
    /// Raw EEG samples: `samples[sample_index][channel_index]`.
    /// Typically 300 samples × 64 channels.
    /// Stored as Vec<Vec<f32>> for serde compatibility (serde arrays only up to size 32).
    pub samples: Vec<Vec<f32>>,
    /// Sample rate of the source stream (Hz).
    pub sample_rate: f32,
    /// Number of pre-stimulus samples included (default 60 = 200 ms).
    pub pre_samples: usize,
    /// Optional free-text note.
    pub notes: Option<String>,
}

impl StimulusEpoch {
    /// Duration of this epoch in milliseconds.
    pub fn duration_ms(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate * 1000.0
    }

    /// Extract a single channel as a flat `Vec<f32>`.
    pub fn channel(&self, ch: usize) -> Vec<f32> {
        self.samples.iter().map(|s| s.get(ch).copied().unwrap_or(0.0)).collect()
    }
}

/// An ongoing recording session containing all captured epochs.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct RecordingSession {
    /// Session identifier (UUID v4 as string).
    pub session_id: String,
    /// Unix timestamp when the session was created.
    pub created_at: f64,
    /// Source device name, e.g. "Cognionics HD-72" or "LSL".
    pub device: String,
    /// All accepted epochs in capture order.
    pub epochs: Vec<StimulusEpoch>,
}

impl RecordingSession {
    pub fn new(device: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            created_at: now,
            device,
            epochs: Vec::new(),
        }
    }

    /// Number of epochs for a given label.
    pub fn count_for(&self, label: &str) -> usize {
        self.epochs.iter().filter(|e| e.label == label).count()
    }

    /// Minimum epochs-per-class across all classes present.
    pub fn min_class_count(&self) -> usize {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for e in &self.epochs {
            *counts.entry(e.label.as_str()).or_default() += 1;
        }
        counts.values().copied().min().unwrap_or(0)
    }

    /// All distinct labels in the session.
    pub fn labels(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.epochs.iter().map(|e| e.label.clone()).filter(|l| seen.insert(l.clone())).collect()
    }
}
