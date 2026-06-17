"""
run_example.py
==============
Minimal end-to-end demonstration:
  1. build a few synthetic 3D nanoporous structures
  2. extract features from each
  3. assemble a feature table
  4. (optionally) train the surrogate if TensorFlow is installed

This runs with NO external data files so anyone can verify the pipeline.
Replace the synthetic-structure generator with your own loader that reads
the 3D voxel geometry exported from FDTD (e.g., a .npy / .mat voxel array).
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_extraction.feature_extraction_3d import extract_features


def make_synthetic_structure(seed, nx=40, ny=40, nz=40):
    """Create one synthetic 3D structure with random pores in the top half."""
    rng = np.random.default_rng(seed)
    solid = np.ones((nx, ny, nz), dtype=bool)
    n_pores = rng.integers(15, 30)
    for _ in range(n_pores):
        cx, cy, cz = rng.integers(0, nx), rng.integers(0, ny), rng.integers(nz//2, nz)
        rad = rng.integers(2, 4)
        xx, yy, zz = np.ogrid[:nx, :ny, :nz]
        solid[(xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2 <= rad**2] = False
    return solid


def main():
    print("Generating synthetic structures and extracting features...")
    rows = []
    for seed in range(10):
        im = make_synthetic_structure(seed)
        feats = extract_features(im, voxel_size=1.0)
        feats["structure_id"] = seed
        rows.append(feats)

    df = pd.DataFrame(rows).set_index("structure_id")
    print(f"\nFeature table: {df.shape[0]} structures x {df.shape[1]} features")
    print(df.round(3).to_string())

    out = os.path.join(os.path.dirname(__file__), "example_features.csv")
    df.to_csv(out)
    print(f"\nSaved feature table to {out}")

    # Optional: train the surrogate if TF is present (uses dummy targets here)
    try:
        from model.surrogate_model import train_surrogate, normalize_features, r2_score
        print("\nTensorFlow found - running a short training demo on dummy targets.")
        X = df.values
        y = np.random.default_rng(0).random(len(df))   # placeholder targets
        Xn, _ = normalize_features(X, [])
        model, hist = train_surrogate(Xn, y, epochs=5)
        print("Demo training complete (5 epochs).")
    except ImportError:
        print("\nTensorFlow not installed - skipping model demo (feature "
              "extraction shown above is the core pipeline).")


if __name__ == "__main__":
    main()
