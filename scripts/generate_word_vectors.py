#!/usr/bin/env python3
"""Generate compressed word vectors from GloVe embeddings.

Usage:
    python scripts/generate_word_vectors.py /path/to/glove.6B.300d.txt

Outputs data/word_vectors.bin in the format:
    [u32 num_words][u32 vec_dim]
    per word: [u16 word_len][utf8 bytes][f32 x vec_dim]
"""

import sys
import struct
import numpy as np
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-glove.6B.300d.txt>", file=sys.stderr)
        sys.exit(1)

    glove_path = Path(sys.argv[1])
    if not glove_path.exists():
        print(f"File not found: {glove_path}", file=sys.stderr)
        sys.exit(1)

    target_dim = 128
    max_words = 1000

    # Load first 1000 alphabetic words
    words = []
    vectors = []
    print(f"Loading GloVe vectors from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(words) >= max_words:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            if not word.isalpha():
                continue
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vec.shape[0] != 300:
                continue
            words.append(word)
            vectors.append(vec)

    print(f"Loaded {len(words)} words")

    # Stack into matrix (num_words x 300)
    mat = np.stack(vectors, axis=0)

    # Center
    mean = mat.mean(axis=0)
    mat_centered = mat - mean

    # PCA via truncated SVD -> project 300d to 128d
    print(f"Computing SVD for PCA projection to {target_dim}d...")
    U, S, Vt = np.linalg.svd(mat_centered, full_matrices=False)
    # Project: take first target_dim components
    projected = mat_centered @ Vt[:target_dim].T  # (num_words x target_dim)

    # L2 normalize each vector
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    projected = projected / norms
    projected = projected.astype(np.float32)

    # Write binary file
    out_path = Path(__file__).parent.parent / "data" / "word_vectors.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_words = len(words)
    vec_dim = target_dim

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", num_words, vec_dim))
        for i, word in enumerate(words):
            word_bytes = word.encode("utf-8")
            f.write(struct.pack("<H", len(word_bytes)))
            f.write(word_bytes)
            f.write(projected[i].tobytes())

    print(f"Wrote {num_words} words x {vec_dim}d to {out_path}")
    print(f"File size: {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
