//! Filter-Bank Common Spatial Patterns (FBCSP) + Shrinkage LDA classifier.
//!
//! Pipeline:
//!   1. Filter bank: 9 sub-bands (4-8, 8-12, ..., 36-40 Hz) via FFT bandpass
//!   2. CSP: per sub-band spatial filters maximizing class variance separation
//!   3. Features: log-variance of CSP-projected signals
//!   4. Classifier: shrinkage LDA (Ledoit-Wolf regularized covariance)
//!
//! Reference: Ang et al. 2012, "Filter Bank Common Spatial Pattern Algorithm
//! on BCI Competition IV Datasets 2a and 2b" (Frontiers in Neuroscience).

use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};

// ── Filter bank ─────────────────────────────────────────────────────────────

/// Sub-band frequency ranges (Hz). 9 bands spanning 4-40 Hz in 4 Hz steps.
const FILTER_BANK: [(f32, f32); 9] = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
];

/// Number of CSP filter pairs per sub-band.
const CSP_PAIRS: usize = 2;

/// Minimum training epochs per class.
pub const MIN_EPOCHS: usize = 15;

/// FFT-based bandpass filter with smooth cosine-tapered edges.
fn bandpass_fft(signal: &[f32], sample_rate: f32, lo_hz: f32, hi_hz: f32) -> Vec<f32> {
    let n = signal.len();
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_fwd.process(&mut buf);

    let bin_hz = sample_rate / n as f32;
    let trans = 1.0_f32; // transition width in Hz (raised-cosine taper)
    for (i, c) in buf.iter_mut().enumerate() {
        let freq = if i <= n / 2 { i as f32 * bin_hz } else { (n - i) as f32 * bin_hz };
        let gain = if freq < lo_hz - trans || freq > hi_hz + trans {
            0.0
        } else if freq < lo_hz {
            // Ramp up: 0 at (lo - trans), 1 at lo
            0.5 * (1.0 - (std::f32::consts::PI * (freq - lo_hz) / trans).cos())
        } else if freq > hi_hz {
            // Ramp down: 1 at hi, 0 at (hi + trans)
            0.5 * (1.0 + (std::f32::consts::PI * (freq - hi_hz) / trans).cos())
        } else {
            1.0
        };
        *c = Complex::new(c.re * gain, c.im * gain);
    }

    fft_inv.process(&mut buf);
    let scale = 1.0 / n as f32;
    buf.iter().map(|c| c.re * scale).collect()
}

/// Bandpass filter a multi-channel trial. Returns (channels x time) as flat Vec.
fn bandpass_trial(
    trial: &[Vec<f32>],
    sample_rate: f32,
    lo_hz: f32,
    hi_hz: f32,
) -> Vec<Vec<f32>> {
    trial
        .iter()
        .map(|ch| bandpass_fft(ch, sample_rate, lo_hz, hi_hz))
        .collect()
}

// ── Matrix helpers (row-major, using faer for eigendecomp) ──────────────────

/// Compute the mean-centered covariance matrix of a (channels x time) trial.
/// Returns a (C x C) symmetric matrix as flat row-major Vec.
fn covariance(trial: &[Vec<f32>]) -> Vec<f32> {
    let c = trial.len();
    let t = trial[0].len();
    // Per-channel means
    let means: Vec<f32> = trial
        .iter()
        .map(|ch| ch.iter().sum::<f32>() / t as f32)
        .collect();
    let mut cov = vec![0.0_f32; c * c];
    for i in 0..c {
        for j in i..c {
            let mut sum = 0.0_f32;
            for k in 0..t {
                sum += (trial[i][k] - means[i]) * (trial[j][k] - means[j]);
            }
            let val = sum / (t - 1).max(1) as f32;
            cov[i * c + j] = val;
            cov[j * c + i] = val;
        }
    }
    cov
}

/// Add alpha * I to a symmetric matrix (in-place).
fn regularize(mat: &mut [f32], n: usize, alpha: f32) {
    for i in 0..n {
        mat[i * n + i] += alpha;
    }
}

/// Solve generalized eigenvalue problem: A w = lambda (A + B) w
/// via Cholesky reduction. Returns (eigenvalues_desc, eigenvectors_as_columns).
///
/// Uses `faer` for Cholesky and symmetric eigendecomposition.
fn generalized_eigen(a: &[f32], b: &[f32], n: usize) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
    use faer::prelude::*;

    // Composite: C = A + B
    let mut c_flat: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) as f64).collect();

    // Regularize composite for numerical stability
    let trace: f64 = (0..n).map(|i| c_flat[i * n + i]).sum();
    let eps = 1e-6 * trace / n as f64;
    for i in 0..n {
        c_flat[i * n + i] += eps;
    }

    // Build faer matrices
    let c_mat = faer::Mat::from_fn(n, n, |i, j| c_flat[i * n + j]);
    let a_mat = faer::Mat::from_fn(n, n, |i, j| a[i * n + j] as f64);

    // Cholesky of composite: C = L L^T
    let chol = c_mat.llt(faer::Side::Lower).ok()?;

    // Compute L_inv by solving C * X = I
    let eye = faer::Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
    let l_inv = chol.solve(&eye);

    // Z = L_inv * A * L_inv^T (symmetric reduced form)
    let z_tmp = &l_inv * &a_mat;
    let l_inv_t = l_inv.transpose().to_owned();
    let z = &z_tmp * &l_inv_t;

    // Symmetrize Z (numerical cleanup)
    let z_sym = faer::Mat::from_fn(n, n, |i, j| (z[(i, j)] + z[(j, i)]) / 2.0);

    // Symmetric eigendecomposition
    let eigen = z_sym.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let eig_vals = eigen.S().column_vector();
    let eig_vecs = eigen.U();

    // Back-transform eigenvectors: W = L^{-T} * U
    let w = &l_inv_t * &eig_vecs;

    // Sort by eigenvalue descending
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|i| (eig_vals[i], i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f32> = indexed.iter().map(|(v, _)| *v as f32).collect();
    let eigenvectors: Vec<Vec<f32>> = indexed
        .iter()
        .map(|(_, idx)| (0..n).map(|row| w[(row, *idx)] as f32).collect())
        .collect();

    Some((eigenvalues, eigenvectors))
}

// ── CSP ─────────────────────────────────────────────────────────────────────

/// Spatial filters for one sub-band. Columns of W are the 2*CSP_PAIRS selected filters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CspFilters {
    /// Shape: (2*CSP_PAIRS) x n_channels, stored as Vec of filter vectors.
    pub filters: Vec<Vec<f32>>,
    pub n_channels: usize,
}

impl CspFilters {
    /// Train CSP filters from two sets of bandpass-filtered trials.
    /// Each trial is (channels x time).
    fn train(class1: &[Vec<Vec<f32>>], class2: &[Vec<Vec<f32>>], n_channels: usize) -> Option<Self> {
        if class1.is_empty() || class2.is_empty() {
            return None;
        }

        // Average covariance per class
        let mut cov1 = vec![0.0_f32; n_channels * n_channels];
        for trial in class1 {
            let c = covariance(trial);
            for (x, y) in cov1.iter_mut().zip(c.iter()) {
                *x += y;
            }
        }
        let n1 = class1.len() as f32;
        for x in &mut cov1 {
            *x /= n1;
        }

        let mut cov2 = vec![0.0_f32; n_channels * n_channels];
        for trial in class2 {
            let c = covariance(trial);
            for (x, y) in cov2.iter_mut().zip(c.iter()) {
                *x += y;
            }
        }
        let n2 = class2.len() as f32;
        for x in &mut cov2 {
            *x /= n2;
        }

        // Regularize class covariances
        let gamma = 0.05_f32;
        let trace1: f32 = (0..n_channels).map(|i| cov1[i * n_channels + i]).sum();
        let trace2: f32 = (0..n_channels).map(|i| cov2[i * n_channels + i]).sum();
        regularize(&mut cov1, n_channels, gamma * trace1 / n_channels as f32);
        regularize(&mut cov2, n_channels, gamma * trace2 / n_channels as f32);

        // Solve generalized eigenvalue problem: cov1 * w = lambda * (cov1 + cov2) * w
        let (_eigenvalues, eigenvectors) = generalized_eigen(&cov1, &cov2, n_channels)?;

        if eigenvectors.len() < 2 * CSP_PAIRS {
            return None;
        }

        // Select top-m (best for class 1) and bottom-m (best for class 2)
        let m = CSP_PAIRS.min(eigenvectors.len() / 2);
        let mut filters = Vec::with_capacity(2 * m);
        for i in 0..m {
            filters.push(eigenvectors[i].clone());
        }
        for i in (eigenvectors.len() - m)..eigenvectors.len() {
            filters.push(eigenvectors[i].clone());
        }

        Some(CspFilters { filters, n_channels })
    }

    /// Project a trial through the spatial filters. Returns log-variance features.
    fn extract_features(&self, trial: &[Vec<f32>]) -> Vec<f32> {
        let t = trial[0].len();
        let n_filters = self.filters.len();
        let mut variances = vec![0.0_f32; n_filters];

        for (fi, filter) in self.filters.iter().enumerate() {
            // Project: z[t] = sum_ch filter[ch] * trial[ch][t]
            let mut sum_sq = 0.0_f32;
            let mut sum = 0.0_f32;
            for s in 0..t {
                let mut val = 0.0_f32;
                for (ch, &w) in filter.iter().enumerate() {
                    if ch < trial.len() {
                        val += w * trial[ch][s];
                    }
                }
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / t as f32;
            variances[fi] = (sum_sq / t as f32 - mean * mean).max(1e-12);
        }

        // Log-variance, normalized
        let total: f32 = variances.iter().sum();
        variances
            .iter()
            .map(|&v| (v / total.max(1e-12)).ln())
            .collect()
    }
}

// ── Shrinkage LDA ───────────────────────────────────────────────────────────

/// Trained LDA classifier with Ledoit-Wolf shrinkage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShrinkageLda {
    /// Per-class weights: w_k = Sigma_shrunk^{-1} * mu_k
    pub weights: Vec<Vec<f32>>,
    /// Per-class bias: c_k = -0.5 * mu_k^T * w_k + ln(pi_k)
    pub biases: Vec<f32>,
    /// Class labels.
    pub labels: Vec<String>,
}

impl ShrinkageLda {
    /// Train LDA with Ledoit-Wolf shrinkage on the within-class covariance.
    ///
    /// `features`: list of (feature_vector, class_index) pairs.
    /// `labels`: class label strings.
    fn train(features: &[(Vec<f32>, usize)], labels: &[String]) -> Option<Self> {
        let n_classes = labels.len();
        let n_features = features.first()?.0.len();
        let n_samples = features.len();

        if n_features == 0 || n_classes < 2 {
            return None;
        }

        // Compute class means
        let mut class_counts = vec![0usize; n_classes];
        let mut class_sums = vec![vec![0.0_f32; n_features]; n_classes];
        for (feat, class) in features {
            class_counts[*class] += 1;
            for (s, &v) in class_sums[*class].iter_mut().zip(feat.iter()) {
                *s += v;
            }
        }
        let class_means: Vec<Vec<f32>> = (0..n_classes)
            .map(|k| {
                let n = class_counts[k].max(1) as f32;
                class_sums[k].iter().map(|&s| s / n).collect()
            })
            .collect();

        // Compute pooled within-class scatter matrix
        let p = n_features;
        let mut s_w = vec![0.0_f32; p * p];
        let mut centered_data: Vec<Vec<f32>> = Vec::with_capacity(n_samples);

        for (feat, class) in features {
            let mean = &class_means[*class];
            let centered: Vec<f32> = feat.iter().zip(mean.iter()).map(|(&f, &m)| f - m).collect();
            // Outer product accumulation
            for i in 0..p {
                for j in i..p {
                    let val = centered[i] * centered[j];
                    s_w[i * p + j] += val;
                    if i != j {
                        s_w[j * p + i] += val;
                    }
                }
            }
            centered_data.push(centered);
        }

        // Use biased covariance (1/n) for Ledoit-Wolf shrinkage estimation
        // (the derivation assumes this normalization), then the unbiased (1/(n-K))
        // for the actual LDA covariance.
        let n_f = n_samples as f32;
        let mut s_w_biased = s_w.clone();
        for v in &mut s_w_biased {
            *v /= n_f;
        }
        let alpha = ledoit_wolf_alpha(&centered_data, &s_w_biased, p);

        let denom = (n_samples - n_classes).max(1) as f32;
        for v in &mut s_w {
            *v /= denom;
        }
        let mu_target: f32 = (0..p).map(|i| s_w[i * p + i]).sum::<f32>() / p as f32;

        // Sigma_shrunk = (1 - alpha) * S_w + alpha * mu * I
        let mut sigma = s_w.clone();
        for i in 0..p {
            for j in 0..p {
                sigma[i * p + j] *= 1.0 - alpha;
            }
            sigma[i * p + i] += alpha * mu_target;
        }

        // Invert Sigma_shrunk via Cholesky (using faer)
        let sigma_inv = invert_symmetric(&sigma, p)?;

        // Compute per-class weights and biases
        let mut weights = Vec::with_capacity(n_classes);
        let mut biases = Vec::with_capacity(n_classes);
        let total_samples = n_samples as f32;

        for k in 0..n_classes {
            // w_k = Sigma_inv * mu_k
            let mu_k = &class_means[k];
            let w_k: Vec<f32> = (0..p)
                .map(|i| {
                    (0..p).map(|j| sigma_inv[i * p + j] * mu_k[j]).sum::<f32>()
                })
                .collect();

            // c_k = -0.5 * mu_k^T * w_k + ln(pi_k)
            let dot: f32 = mu_k.iter().zip(w_k.iter()).map(|(&m, &w)| m * w).sum();
            let pi_k = class_counts[k] as f32 / total_samples;
            let c_k = -0.5 * dot + pi_k.max(1e-10).ln();

            weights.push(w_k);
            biases.push(c_k);
        }

        Some(ShrinkageLda {
            weights,
            biases,
            labels: labels.to_vec(),
        })
    }

    /// Classify a feature vector. Returns (predicted_label, class_scores).
    fn predict(&self, features: &[f32]) -> (String, Vec<(String, f32)>) {
        let scores: Vec<f32> = self
            .weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w, &b)| {
                let dot: f32 = w.iter().zip(features.iter()).map(|(&wi, &fi)| wi * fi).sum();
                dot + b
            })
            .collect();

        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let label_scores: Vec<(String, f32)> = self
            .labels
            .iter()
            .zip(scores.iter())
            .map(|(l, &s)| (l.clone(), s))
            .collect();

        (self.labels[best].clone(), label_scores)
    }
}

/// Ledoit-Wolf optimal shrinkage intensity (Oracle Approximating Shrinkage).
fn ledoit_wolf_alpha(centered_data: &[Vec<f32>], s_w: &[f32], p: usize) -> f32 {
    let n = centered_data.len() as f32;
    if n < 2.0 || p == 0 {
        return 1.0;
    }

    let mu: f32 = (0..p).map(|i| s_w[i * p + i]).sum::<f32>() / p as f32;

    // delta = ||S_w - mu*I||_F^2 / p
    let mut delta = 0.0_f32;
    for i in 0..p {
        for j in 0..p {
            let target = if i == j { mu } else { 0.0 };
            let diff = s_w[i * p + j] - target;
            delta += diff * diff;
        }
    }
    delta /= p as f32;

    // beta = (1/n^2) * sum_k ||x_k x_k^T - S_w||_F^2 / p
    let mut beta_sum = 0.0_f32;
    for x in centered_data {
        // Compute ||x x^T - S_w||_F^2
        let mut norm_sq = 0.0_f32;
        for i in 0..p {
            for j in 0..p {
                let outer = x[i] * x[j];
                let diff = outer - s_w[i * p + j];
                norm_sq += diff * diff;
            }
        }
        beta_sum += norm_sq;
    }
    let beta = beta_sum / (n * n * p as f32);

    // alpha = min(beta / delta, 1.0)
    if delta > 1e-10 {
        (beta / delta).clamp(0.0, 1.0)
    } else {
        1.0 // Degenerate case: shrink fully to identity
    }
}

/// Invert a symmetric positive-definite matrix using faer Cholesky.
fn invert_symmetric(mat: &[f32], n: usize) -> Option<Vec<f32>> {
    use faer::prelude::*;

    let m = faer::Mat::from_fn(n, n, |i, j| mat[i * n + j] as f64);
    let chol = m.llt(faer::Side::Lower).ok()?;
    let eye = faer::Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
    let inv = chol.solve(&eye);

    let mut result = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = inv[(i, j)] as f32;
        }
    }
    Some(result)
}

// ── FBCSP Trained Model ─────────────────────────────────────────────────────

/// Complete trained FBCSP + shrinkage LDA model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FbcspModel {
    /// CSP filters per sub-band.  Each entry is a Vec of CspFilters,
    /// one per One-vs-Rest binary problem (K filters for K classes).
    pub csp_per_band: Vec<Option<Vec<CspFilters>>>,
    /// LDA classifier operating on concatenated CSP features.
    pub lda: ShrinkageLda,
    /// Number of channels the model was trained on.
    pub n_channels: usize,
    /// Sample rate used during training.
    pub sample_rate: f32,
    /// Class labels.
    pub labels: Vec<String>,
}

/// A training trial: multi-channel EEG segment with a class label.
#[derive(Clone)]
pub struct FbcspTrial {
    /// Per-channel time series. channels[ch][sample].
    pub channels: Vec<Vec<f32>>,
    /// Class label string.
    pub label: String,
}

impl FbcspModel {
    /// Train a new FBCSP model from labeled trials.
    ///
    /// Requires at least `MIN_EPOCHS` trials per class, and at least 2 classes.
    pub fn train(trials: &[FbcspTrial], sample_rate: f32) -> Option<Self> {
        if trials.is_empty() {
            return None;
        }

        let n_channels = trials[0].channels.len();

        // Group trials by label
        let mut label_set: Vec<String> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new(); // groups[class_idx] = [trial_indices]
        for (i, trial) in trials.iter().enumerate() {
            if let Some(pos) = label_set.iter().position(|l| l == &trial.label) {
                groups[pos].push(i);
            } else {
                label_set.push(trial.label.clone());
                groups.push(vec![i]);
            }
        }

        // Require at least 2 classes with enough data
        let valid_classes: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.len() >= MIN_EPOCHS)
            .map(|(i, _)| i)
            .collect();

        if valid_classes.len() < 2 {
            return None;
        }

        // Bandpass-filter all trials for each sub-band
        let mut filtered_trials: Vec<Vec<Vec<Vec<f32>>>> =
            Vec::with_capacity(FILTER_BANK.len());
        // filtered_trials[band][trial_idx] = channels x time

        for &(lo, hi) in &FILTER_BANK {
            let band_trials: Vec<Vec<Vec<f32>>> = trials
                .iter()
                .map(|t| bandpass_trial(&t.channels, sample_rate, lo, hi))
                .collect();
            filtered_trials.push(band_trials);
        }

        // One-vs-Rest CSP: for each valid class, train CSP(this_class vs all_others)
        let mut csp_per_band: Vec<Option<Vec<CspFilters>>> =
            Vec::with_capacity(FILTER_BANK.len());

        for band_idx in 0..FILTER_BANK.len() {
            let mut band_csps: Vec<CspFilters> = Vec::new();
            for &vc in &valid_classes {
                let this_trials: Vec<Vec<Vec<f32>>> = groups[vc]
                    .iter()
                    .map(|&ti| filtered_trials[band_idx][ti].clone())
                    .collect();
                let rest_trials: Vec<Vec<Vec<f32>>> = valid_classes
                    .iter()
                    .filter(|&&c| c != vc)
                    .flat_map(|&c| groups[c].iter())
                    .map(|&ti| filtered_trials[band_idx][ti].clone())
                    .collect();
                if let Some(csp) = CspFilters::train(&this_trials, &rest_trials, n_channels) {
                    band_csps.push(csp);
                }
            }
            csp_per_band.push(if band_csps.is_empty() {
                None
            } else {
                Some(band_csps)
            });
        }

        // Active labels: only those classes with enough data
        let active_labels: Vec<String> = valid_classes
            .iter()
            .map(|&i| label_set[i].clone())
            .collect();

        // Extract features for all trials across all sub-bands
        let mut all_features: Vec<(Vec<f32>, usize)> = Vec::new();

        for (ti, trial) in trials.iter().enumerate() {
            // Map to active class index (position within valid_classes)
            let orig_idx = match label_set.iter().position(|l| l == &trial.label) {
                Some(idx) if valid_classes.contains(&idx) => idx,
                _ => continue,
            };
            let class_idx = valid_classes
                .iter()
                .position(|&v| v == orig_idx)
                .unwrap();

            let mut feat = Vec::new();
            for (band_idx, csp_set_opt) in csp_per_band.iter().enumerate() {
                if let Some(csp_set) = csp_set_opt {
                    for csp in csp_set {
                        let band_feat =
                            csp.extract_features(&filtered_trials[band_idx][ti]);
                        feat.extend_from_slice(&band_feat);
                    }
                }
            }

            if !feat.is_empty() {
                all_features.push((feat, class_idx));
            }
        }

        if all_features.len() < 2 * valid_classes.len() {
            return None;
        }

        // Train LDA on the concatenated features
        let lda = ShrinkageLda::train(&all_features, &active_labels)?;

        Some(FbcspModel {
            csp_per_band,
            lda,
            n_channels,
            sample_rate,
            labels: active_labels,
        })
    }

    /// Classify a single trial. Returns (predicted_label, class_scores).
    pub fn predict(&self, trial_channels: &[Vec<f32>]) -> (String, Vec<(String, f32)>) {
        let mut feat = Vec::new();

        for (band_idx, &(lo, hi)) in FILTER_BANK.iter().enumerate() {
            if let Some(ref csp_set) = self.csp_per_band[band_idx] {
                let filtered = bandpass_trial(trial_channels, self.sample_rate, lo, hi);
                for csp in csp_set {
                    let band_feat = csp.extract_features(&filtered);
                    feat.extend_from_slice(&band_feat);
                }
            }
        }

        if feat.is_empty() {
            return (String::new(), Vec::new());
        }

        self.lda.predict(&feat)
    }

    /// Total number of features (for diagnostics).
    pub fn n_features(&self) -> usize {
        self.csp_per_band
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|set| set.len() * 2 * CSP_PAIRS)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandpass_preserves_length() {
        let signal: Vec<f32> = (0..300).map(|i| (i as f32 * 0.1).sin()).collect();
        let filtered = bandpass_fft(&signal, 300.0, 8.0, 12.0);
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_covariance_symmetric() {
        let trial = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        let cov = covariance(&trial);
        assert!((cov[0 * 2 + 1] - cov[1 * 2 + 0]).abs() < 1e-6);
    }

    #[test]
    fn test_ledoit_wolf_alpha_range() {
        let data = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        let cov = vec![0.25, 0.0, 0.0, 0.25];
        let alpha = ledoit_wolf_alpha(&data, &cov, 2);
        assert!(alpha >= 0.0 && alpha <= 1.0);
    }

    /// Generate a synthetic EEG trial with a dominant frequency band.
    fn synthetic_trial(n_channels: usize, n_samples: usize, freq_hz: f32, sr: f32) -> Vec<Vec<f32>> {
        (0..n_channels)
            .map(|ch| {
                (0..n_samples)
                    .map(|t| {
                        let phase = ch as f32 * 0.5;
                        let noise = ((t as f32 * 7.3 + ch as f32 * 13.7).sin()) * 2.0;
                        (2.0 * std::f32::consts::PI * freq_hz * t as f32 / sr + phase).sin() * 20.0
                            + noise
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_csp_train_and_extract() {
        let n_ch = 8;
        let n_samples = 300;
        let sr = 300.0;
        // Class 1: strong 10 Hz (alpha), Class 2: strong 22 Hz (beta)
        let class1: Vec<Vec<Vec<f32>>> = (0..20)
            .map(|_| synthetic_trial(n_ch, n_samples, 10.0, sr))
            .collect();
        let class2: Vec<Vec<Vec<f32>>> = (0..20)
            .map(|_| synthetic_trial(n_ch, n_samples, 22.0, sr))
            .collect();

        let csp = CspFilters::train(&class1, &class2, n_ch);
        assert!(csp.is_some(), "CSP training should succeed");
        let csp = csp.unwrap();
        assert_eq!(csp.filters.len(), 2 * CSP_PAIRS);

        // Extract features from a class 1 trial
        let feat = csp.extract_features(&class1[0]);
        assert_eq!(feat.len(), 2 * CSP_PAIRS);
        // Features should be finite
        assert!(feat.iter().all(|f| f.is_finite()), "features must be finite");
    }

    #[test]
    fn test_shrinkage_lda_train_and_predict() {
        // Simple 2-class problem with separable features
        let mut features = Vec::new();
        for _ in 0..30 {
            features.push((vec![1.0, 0.2, -0.5], 0));
            features.push((vec![-1.0, -0.3, 0.5], 1));
        }
        let labels = vec!["alpha".to_string(), "beta".to_string()];
        let lda = ShrinkageLda::train(&features, &labels);
        assert!(lda.is_some(), "LDA training should succeed");
        let lda = lda.unwrap();

        let (pred, _scores) = lda.predict(&[1.0, 0.2, -0.5]);
        assert_eq!(pred, "alpha");
        let (pred, _scores) = lda.predict(&[-1.0, -0.3, 0.5]);
        assert_eq!(pred, "beta");
    }

    #[test]
    fn test_fbcsp_full_pipeline() {
        let n_ch = 8;
        let n_samples = 300;
        let sr = 300.0;

        // Create synthetic trials: class A = 10 Hz dominant, class B = 25 Hz dominant
        let mut trials = Vec::new();
        for _ in 0..20 {
            trials.push(FbcspTrial {
                channels: synthetic_trial(n_ch, n_samples, 10.0, sr),
                label: "alpha_state".to_string(),
            });
            trials.push(FbcspTrial {
                channels: synthetic_trial(n_ch, n_samples, 25.0, sr),
                label: "beta_state".to_string(),
            });
        }

        let model = FbcspModel::train(&trials, sr);
        assert!(model.is_some(), "FBCSP model training should succeed");
        let model = model.unwrap();
        assert_eq!(model.labels.len(), 2);
        assert!(model.n_features() > 0);

        // Predict on a new alpha trial
        let test_trial = synthetic_trial(n_ch, n_samples, 10.0, sr);
        let (label, scores) = model.predict(&test_trial);
        assert!(!label.is_empty(), "prediction should return a label");
        assert!(!scores.is_empty(), "prediction should return scores");
        eprintln!("[test] predicted: {label}, scores: {scores:?}");
    }
}
