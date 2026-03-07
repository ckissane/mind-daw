use std::time::Instant;

pub const FEATURE_DIM: usize = 2048;
pub const WORD_DIM: usize = 128;
pub const NUM_WORDS: usize = 20;
const TRAINING_WORDS: usize = 20;
const WORD_DISPLAY_SECS: f32 = 2.0;
const LEARNING_RATE: f32 = 0.01;
const TOP_K: usize = 5;

pub struct WordVocab {
    pub words: Vec<String>,
    pub vectors: Vec<[f32; WORD_DIM]>,
}

impl WordVocab {
    pub fn load_from_file(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 8 {
            return None;
        }

        let num_words = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let vec_dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if vec_dim != WORD_DIM || num_words == 0 {
            return None;
        }

        let mut words = Vec::with_capacity(num_words);
        let mut vectors = Vec::with_capacity(num_words);
        let mut offset = 8;

        for _ in 0..num_words {
            if offset + 2 > data.len() {
                return None;
            }
            let word_len =
                u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + word_len > data.len() {
                return None;
            }
            let word = String::from_utf8_lossy(&data[offset..offset + word_len]).into_owned();
            offset += word_len;

            let float_bytes = WORD_DIM * 4;
            if offset + float_bytes > data.len() {
                return None;
            }
            let mut vec = [0.0f32; WORD_DIM];
            for i in 0..WORD_DIM {
                let start = offset + i * 4;
                vec[i] = f32::from_le_bytes([
                    data[start],
                    data[start + 1],
                    data[start + 2],
                    data[start + 3],
                ]);
            }
            offset += float_bytes;

            words.push(word);
            vectors.push(vec);
        }

        Some(Self { words, vectors })
    }

    pub fn random_fallback() -> Self {
        // Load real words from data/words.txt
        let word_list = std::fs::read_to_string("data/words.txt").unwrap_or_default();
        let mut words: Vec<String> = word_list
            .lines()
            .filter(|l| !l.is_empty())
            .take(NUM_WORDS)
            .map(|s| s.to_string())
            .collect();

        // Pad if file was short or missing
        while words.len() < NUM_WORDS {
            words.push(format!("word{}", words.len()));
        }

        let mut vectors = Vec::with_capacity(NUM_WORDS);

        // Deterministic LCG random vectors
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let lcg_next = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        for _ in 0..words.len() {
            let mut vec = [0.0f32; WORD_DIM];
            for v in &mut vec {
                *v = lcg_next(&mut rng_state);
            }
            // L2 normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            vectors.push(vec);
        }

        Self { words, vectors }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum TrainingPhase {
    Idle,
    ShowingWord,
}

pub struct WordReadState {
    pub vocab: WordVocab,
    pub projection_w: Vec<[f32; FEATURE_DIM]>,
    pub phase: TrainingPhase,
    pub current_word_idx: usize,
    pub words_trained: usize,
    pub word_shown_at: Option<Instant>,
    pub latest_features: Vec<f32>,
    pub top_predictions: Vec<(String, f32)>,
}

impl WordReadState {
    pub fn new() -> Self {
        let vocab = match WordVocab::load_from_file("data/word_vectors.bin") {
            Some(v) => {
                eprintln!("[word_read] Loaded {} words from data/word_vectors.bin", v.words.len());
                v
            }
            None => {
                eprintln!("[word_read] data/word_vectors.bin not found, using random fallback");
                WordVocab::random_fallback()
            }
        };

        // Init projection W (WORD_DIM rows x FEATURE_DIM cols) with small deterministic random values
        let mut projection_w = Vec::with_capacity(WORD_DIM);
        let mut rng_state: u64 = 0xBAAD_F00D_1337_C0DE;
        for _ in 0..WORD_DIM {
            let mut row = [0.0f32; FEATURE_DIM];
            for v in &mut row {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = ((rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * 0.01;
            }
            projection_w.push(row);
        }

        Self {
            vocab,
            projection_w,
            phase: TrainingPhase::Idle,
            current_word_idx: 0,
            words_trained: 0,
            word_shown_at: None,
            latest_features: vec![0.0; FEATURE_DIM],
            top_predictions: Vec::new(),
        }
    }

    pub fn start_training(&mut self) {
        self.current_word_idx = 0;
        self.words_trained = 0;
        self.phase = TrainingPhase::ShowingWord;
        self.word_shown_at = Some(Instant::now());
    }

    pub fn tick(&mut self, features: &[f32]) {
        // Store latest features
        if features.len() == FEATURE_DIM {
            self.latest_features.copy_from_slice(features);
        }

        // Training logic
        if self.phase == TrainingPhase::ShowingWord {
            if let Some(shown_at) = self.word_shown_at {
                if shown_at.elapsed().as_secs_f32() >= WORD_DISPLAY_SECS {
                    // SGD update: W += lr * outer(target - W@x, x)
                    let x = &self.latest_features;
                    let target = &self.vocab.vectors[self.current_word_idx];

                    // Compute W @ x -> predicted (WORD_DIM)
                    let mut predicted = [0.0f32; WORD_DIM];
                    for (j, row) in self.projection_w.iter().enumerate() {
                        let mut sum = 0.0f32;
                        for i in 0..FEATURE_DIM {
                            sum += row[i] * x[i];
                        }
                        predicted[j] = sum;
                    }

                    // error = target - predicted
                    let mut error = [0.0f32; WORD_DIM];
                    for j in 0..WORD_DIM {
                        error[j] = target[j] - predicted[j];
                    }

                    // W += lr * outer(error, x)
                    for j in 0..WORD_DIM {
                        let lr_error = LEARNING_RATE * error[j];
                        let row = &mut self.projection_w[j];
                        for i in 0..FEATURE_DIM {
                            row[i] += lr_error * x[i];
                        }
                    }

                    // Advance to next word, looping over first TRAINING_WORDS
                    self.words_trained += 1;
                    self.current_word_idx += 1;
                    let limit = TRAINING_WORDS.min(self.vocab.words.len());
                    if self.current_word_idx >= limit {
                        self.current_word_idx = 0;
                    }
                    self.word_shown_at = Some(Instant::now());
                }
            }
        }

        // Always update predictions
        self.update_predictions();
    }

    fn update_predictions(&mut self) {
        let x = &self.latest_features;

        // Compute W @ x -> projected (WORD_DIM)
        let mut projected = [0.0f32; WORD_DIM];
        for (j, row) in self.projection_w.iter().enumerate() {
            let mut sum = 0.0f32;
            for i in 0..FEATURE_DIM {
                sum += row[i] * x[i];
            }
            projected[j] = sum;
        }

        // L2 norm of projected
        let proj_norm = projected.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);

        // Cosine similarity vs all vocab vectors
        let mut scores: Vec<(usize, f32)> = self
            .vocab
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dot: f32 = (0..WORD_DIM).map(|j| projected[j] * vec[j]).sum();
                let vec_norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                (idx, dot / (proj_norm * vec_norm))
            })
            .collect();

        // Partial sort for top K
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        self.top_predictions = scores
            .iter()
            .take(TOP_K)
            .map(|(idx, score)| (self.vocab.words[*idx].clone(), *score))
            .collect();
    }

    pub fn current_word(&self) -> Option<&str> {
        if self.phase == TrainingPhase::ShowingWord {
            self.vocab.words.get(self.current_word_idx).map(|s| s.as_str())
        } else {
            None
        }
    }

    pub fn progress(&self) -> f32 {
        let limit = TRAINING_WORDS.min(self.vocab.words.len()).max(1);
        (self.current_word_idx as f32) / limit as f32
    }
}
