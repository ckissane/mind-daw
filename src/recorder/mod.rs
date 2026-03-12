pub mod auto_detect;
pub mod baseline;
pub mod classifier;
pub mod epoch;
pub mod features;
pub mod storage;

pub use auto_detect::AutoDetectThresholds;
pub use baseline::{BaselineProfile, BaselineRecorder};
pub use classifier::{ClassifierPrediction, TrainedClassifier, MIN_EPOCHS_PER_CLASS, RETRAIN_EVERY};
pub use epoch::{RecordingSession, StimulusEpoch};
