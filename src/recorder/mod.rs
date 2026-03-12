pub mod auto_detect;
pub mod baseline;
pub mod classifier;
pub mod epoch;
pub mod features;
pub mod storage;

pub use auto_detect::AutoDetectThresholds;
pub use classifier::ClassifierPrediction;
pub use epoch::{RecordingSession, StimulusEpoch};
