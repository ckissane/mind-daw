use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::recorder::{epoch::StimulusEpoch, features::extract_features};

/// Minimum epochs per class required before the classifier activates.
pub const MIN_EPOCHS_PER_CLASS: usize = 5;

/// Retrain after this many new accepted epochs.
pub const RETRAIN_EVERY: usize = 3;

// ── Trained classifier ────────────────────────────────────────────────────────

/// Serialisable classifier weights — one mean feature vector per class.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainedClassifier {
    /// label → mean-normalised feature vector (cosine template).
    pub templates: HashMap<String, Vec<f32>>,
    pub feature_dim: usize,
}

impl TrainedClassifier {
    /// Build templates by averaging feature vectors per class.
    ///
    /// Only includes classes that have ≥ `MIN_EPOCHS_PER_CLASS` epochs.
    /// Returns `None` if no class meets the threshold.
    pub fn train(epochs: &[StimulusEpoch]) -> Option<Self> {
        // Group epochs by label
        let mut groups: HashMap<&str, Vec<Vec<f32>>> = HashMap::new();
        for ep in epochs {
            groups.entry(ep.label.as_str()).or_default().push(extract_features(ep));
        }

        // Keep only classes with enough data
        let mut templates: HashMap<String, Vec<f32>> = HashMap::new();
        let mut feature_dim = 0usize;
        for (label, feats) in &groups {
            if feats.len() < MIN_EPOCHS_PER_CLASS {
                continue;
            }
            let dim = feats[0].len();
            feature_dim = dim;
            let mut mean = vec![0.0f32; dim];
            for f in feats {
                for (m, v) in mean.iter_mut().zip(f.iter()) {
                    *m += v;
                }
            }
            let n = feats.len() as f32;
            for m in &mut mean {
                *m /= n;
            }
            // L2-normalise the template so dot-product == cosine similarity
            let norm: f32 = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for m in &mut mean {
                    *m /= norm;
                }
            }
            templates.insert(label.to_string(), mean);
        }

        if templates.is_empty() {
            None
        } else {
            Some(Self { templates, feature_dim })
        }
    }
}

// ── Prediction ────────────────────────────────────────────────────────────────

/// Live classification result with continuous similarity scores.
#[derive(Clone, Debug)]
pub struct ClassifierPrediction {
    /// The class with the highest similarity.
    pub predicted_label: String,
    /// Normalised confidence: max_similarity / sum(similarities).
    pub confidence: f32,
    /// Continuous 0–1 similarity to every class template.
    /// `similarity[class] = 1 / (1 + cosine_distance)` — gives "half-blink" values.
    pub similarities: HashMap<String, f32>,
    /// True if the best match similarity is below 0.3 (signal doesn't match any class).
    pub is_novel: bool,
}

/// Classify a pre-computed feature vector (used in live-prediction mode).
pub fn predict_features(features: &[f32], classifier: &TrainedClassifier) -> ClassifierPrediction {
    let mut similarities: HashMap<String, f32> = HashMap::new();

    for (label, template) in &classifier.templates {
        let cos_sim = cosine_similarity(features, template);
        // Map cosine similarity (−1…1) → continuous score (0…1) via 1/(1+distance)
        let distance = 1.0 - cos_sim; // 0 = perfect match
        let score = 1.0 / (1.0 + distance);
        similarities.insert(label.clone(), score);
    }

    // Best class
    let empty_label = String::new();
    let (best_label, &best_score) = similarities
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(l, s)| (l, s))
        .unwrap_or((&empty_label, &0.0));

    let total: f32 = similarities.values().sum();
    let confidence = if total > 1e-10 { best_score / total } else { 0.0 };
    let is_novel = best_score < 0.3;

    ClassifierPrediction {
        predicted_label: best_label.clone(),
        confidence,
        similarities,
        is_novel,
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}
