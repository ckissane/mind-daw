use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::recorder::{baseline::BaselineProfile, epoch::RecordingSession, features::extract_features};

// ── Save / load ───────────────────────────────────────────────────────────────

/// Persist a session to `./sessions/{session_id}/`.
///
/// Creates:
/// - `metadata.json`  — session without epoch raw samples
/// - `epochs/{uuid}_{label}.json` — one file per epoch (full raw data)
pub fn save_session(session: &RecordingSession) -> Result<PathBuf> {
    let dir = session_dir(&session.session_id);
    let epoch_dir = dir.join("epochs");
    std::fs::create_dir_all(&epoch_dir)
        .with_context(|| format!("create {}", epoch_dir.display()))?;

    // Metadata (no raw samples — keep file small)
    let meta = serde_json::json!({
        "session_id": session.session_id,
        "created_at": session.created_at,
        "device": session.device,
        "epoch_count": session.epochs.len(),
        "labels": session.labels(),
    });
    let meta_path = dir.join("metadata.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("write {}", meta_path.display()))?;

    // Individual epoch files
    for ep in &session.epochs {
        let fname = format!("{}_{}.json", ep.id, sanitise_label(&ep.label));
        let path = epoch_dir.join(&fname);
        std::fs::write(&path, serde_json::to_string(ep)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    Ok(dir)
}

/// Load a previously saved session (full raw data) from its directory.
pub fn load_session(session_id: &str) -> Result<RecordingSession> {
    let epoch_dir = session_dir(session_id).join("epochs");
    let mut epochs = Vec::new();

    if epoch_dir.exists() {
        for entry in std::fs::read_dir(&epoch_dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?;
                match serde_json::from_str(&text) {
                    Ok(ep) => epochs.push(ep),
                    Err(e) => eprintln!("[recorder] skip {}: {e}", path.display()),
                }
            }
        }
    }

    // Rebuild session metadata from metadata.json
    let meta_path = session_dir(session_id).join("metadata.json");
    let meta: serde_json::Value = if meta_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?
    } else {
        serde_json::json!({})
    };

    Ok(RecordingSession {
        session_id: session_id.to_string(),
        created_at: meta["created_at"].as_f64().unwrap_or(0.0),
        device: meta["device"].as_str().unwrap_or("unknown").to_string(),
        epochs,
    })
}

// ── CSV export ────────────────────────────────────────────────────────────────

/// Export summary features to CSV.
///
/// Columns: epoch_id, label, timestamp, duration_ms, peak_fp1_uv, peak_fp2_uv,
///          fp1_fp2_asymmetry, peak_jaw_uv, peak_motor_uv,
///          delta, theta, alpha, beta, gamma (averaged across all channels),
///          temporal_hf_power, notes
pub fn export_csv(session: &RecordingSession) -> Result<PathBuf> {
    let export_dir = session_dir(&session.session_id).join("exports");
    std::fs::create_dir_all(&export_dir)?;
    let path = export_dir.join(format!("{}.csv", session.session_id));

    let mut rows = Vec::new();
    rows.push(
        "epoch_id,label,timestamp,duration_ms,\
peak_fp1_uv,peak_fp2_uv,fp1_fp2_asymmetry,\
peak_jaw_uv,peak_motor_uv,\
delta_power,theta_power,alpha_power,beta_power,gamma_power,\
temporal_hf_power,notes"
            .to_string(),
    );

    for ep in &session.epochs {
        let feats = extract_features(ep);
        // Feature layout (from features.rs):
        //   0: peak fp1, 1: peak fp2 (BLINK_CHANNELS)
        //   2–5: peak jaw (JAW_CHANNELS)
        //   6–9: peak motor (MOTOR_CHANNELS)
        //   10..(10 + 5*64): band powers [ch0_delta, ch0_theta, ..., ch63_gamma]
        //   10+320: asymmetry
        //   10+321: temporal hf
        let peak_fp1 = feats.first().copied().unwrap_or(0.0);
        let peak_fp2 = feats.get(1).copied().unwrap_or(0.0);
        let peak_jaw = feats[2..6].iter().copied().fold(0.0f32, f32::max);
        let peak_motor = feats[6..10].iter().copied().fold(0.0f32, f32::max);
        let asym = feats.get(330).copied().unwrap_or(0.0);
        let hf = feats.get(331).copied().unwrap_or(0.0);

        // Average each band across all 64 channels
        let band_start = 10usize;
        let mut band_avgs = [0.0f32; 5];
        for ch in 0..64 {
            for b in 0..5 {
                let idx = band_start + ch * 5 + b;
                band_avgs[b] += feats.get(idx).copied().unwrap_or(0.0);
            }
        }
        for v in &mut band_avgs {
            *v /= 64.0;
        }

        rows.push(format!(
            "{},{},{:.3},{:.1},{:.2},{:.2},{:.4},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
            ep.id,
            ep.label,
            ep.timestamp,
            ep.duration_ms(),
            peak_fp1,
            peak_fp2,
            asym,
            peak_jaw,
            peak_motor,
            band_avgs[0],
            band_avgs[1],
            band_avgs[2],
            band_avgs[3],
            band_avgs[4],
            hf,
            ep.notes.as_deref().unwrap_or(""),
        ));
    }

    std::fs::write(&path, rows.join("\n"))?;
    Ok(path)
}

// ── Baseline profiles ─────────────────────────────────────────────────────────

/// Save raw baseline EEG frames for MNE post-processing.
///
/// Writes two files under `./profiles/{name}/`:
/// - `raw_baseline.bin`  — flat little-endian float32, shape (n_samples, 64)
/// - `raw_baseline.json` — metadata: sample_rate, n_channels, n_samples
///
/// Python reads it as:
/// ```python
/// import numpy as np, json
/// meta = json.load(open("raw_baseline.json"))
/// raw  = np.fromfile("raw_baseline.bin", dtype="<f4")
///         .reshape(meta["n_samples"], meta["n_channels"])
/// ```
pub fn save_raw_baseline(
    name: &str,
    frames: &[[f32; 64]],
    sample_rate: f32,
) -> Result<PathBuf> {
    let slug = sanitise_label(name);
    let dir = profiles_dir().join(&slug);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;

    // Binary data
    let bin_path = dir.join("raw_baseline.bin");
    let mut bytes: Vec<u8> = Vec::with_capacity(frames.len() * 64 * 4);
    for frame in frames {
        for &v in frame {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&bin_path, &bytes)
        .with_context(|| format!("write {}", bin_path.display()))?;

    // Metadata sidecar
    let meta = serde_json::json!({
        "sample_rate": sample_rate,
        "n_channels": 64,
        "n_samples": frames.len(),
    });
    let meta_path = dir.join("raw_baseline.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("write {}", meta_path.display()))?;

    Ok(bin_path)
}

/// Save a named baseline profile to `./profiles/{name}/baseline.json`.
/// `name` is sanitised (spaces → underscores, special chars removed).
pub fn save_baseline_profile(name: &str, profile: &BaselineProfile) -> Result<PathBuf> {
    let slug = sanitise_label(name);
    let dir = profiles_dir().join(&slug);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;
    let path = dir.join("baseline.json");
    std::fs::write(&path, serde_json::to_string_pretty(profile)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Load a named baseline profile from `./profiles/{name}/baseline.json`.
pub fn load_baseline_profile(name: &str) -> Result<BaselineProfile> {
    let slug = sanitise_label(name);
    let path = profiles_dir().join(&slug).join("baseline.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// Return all saved profile names in alphabetical order.
pub fn list_baseline_profiles() -> Vec<String> {
    let dir = profiles_dir();
    if !dir.exists() {
        return Vec::new();
    }
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("baseline.json").exists())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn profiles_dir() -> PathBuf {
    PathBuf::from("profiles")
}

fn session_dir(session_id: &str) -> PathBuf {
    PathBuf::from("sessions").join(session_id)
}

fn sanitise_label(label: &str) -> String {
    label.chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}
