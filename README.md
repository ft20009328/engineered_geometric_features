# Nanoporous Surface Recombination — Feature Extraction & Deep-Learning Surrogate

Code accompanying the paper:

> **Modeling Surface Recombination in Nanoporous Thin-Film Solar Cells Using
> Optical Transfer Matrix and Deep Learning**
> F. Tabassum, S. Hajimirza, *Surfaces and Interfaces* (under review).

This repository provides the automated **3D geometric/statistical feature
extraction** pipeline and the **supervised deep-learning surrogate model**
used to predict the surface recombination rate of stochastic nanoporous
a-Si thin-film solar cells with randomly distributed nanopores and surface
nanodots.

---

## What this code does

1. **Feature extraction** (`src/feature_extraction/feature_extraction_3d.py`)
   Takes a 3D voxelized geometry of the structure (solid a-Si vs. pore
   phase) and computes a single feature vector combining:
   - **Morphological descriptors** (per pore, then aggregated): volume,
     extent, solidity, equivalent diameter, major/minor axis lengths,
     aspect ratio — computed with `scikit-image` `regionprops` in 3D.
   - **Statistical / multi-scale descriptors** (whole domain): porosity,
     two-point correlation function, chord-length distribution,
     pore-size distribution, nearest-neighbor distance — computed with
     `porespy` (natively 3D) and `scipy`.

   These features correspond to those listed in **Appendix II** of the
   manuscript.

2. **Surrogate model** (`src/model/surrogate_model.py`)
   A 4-layer dense network (128–96–64–32, then a single output) trained
   with Adam + MSE and a held-out 10% validation split, matching Table 1
   of the manuscript. Includes training-set-only feature normalization
   (to prevent leakage), an R² metric, and a training/validation loss
   curve plotter.

3. **Example** (`examples/run_example.py`)
   A self-contained demo that builds synthetic 3D structures, extracts
   features, and writes a feature table — runs with **no external data**.

---

## Installation

```bash
git clone https://github.com/<your-username>/nanoporous-surface-recombination.git
cd nanoporous-surface-recombination
pip install -r requirements.txt
```

Python 3.9+ recommended.

## Quick start

```bash
python examples/run_example.py
```

This prints a feature table for 10 synthetic structures and saves
`examples/example_features.csv`.

## Using your own geometry

Replace the synthetic generator in `run_example.py` with a loader for your
own 3D voxel array exported from FDTD (e.g. a `.npy` or `.mat` file), then:

```python
import numpy as np
from src.feature_extraction.feature_extraction_3d import extract_features

im_solid = np.load("my_structure.npy")   # bool array: True=solid, False=pore
features  = extract_features(im_solid, voxel_size=1.0)  # voxel edge in nm
```

For a batch of structures, use `extract_dataset(list_of_arrays)` and convert
the result to a `pandas.DataFrame`.

## Training the surrogate

```python
from src.model.surrogate_model import (
    normalize_features, train_surrogate, r2_score, plot_loss_curve)

X_train_n, [X_test_n] = normalize_features(X_train, [X_test])  # train stats only
model, history = train_surrogate(X_train_n, y_train, epochs=200)
plot_loss_curve(history, "loss_curve.png")
print("Test R^2:", r2_score(y_test, model.predict(X_test_n).ravel()))
```

---

## Notes on reproducibility

- A fixed random seed (`RANDOM_SEED = 1`) is used for data splitting and
  weight initialization.
- Feature normalization uses **training-partition statistics only**.
- The out-of-sample evaluation set should be generated from random seeds
  disjoint from the training/validation/test sets.

## Repository structure

```
nanoporous-surface-recombination/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── src/
│   ├── feature_extraction/feature_extraction_3d.py
│   └── model/surrogate_model.py
└── examples/run_example.py
```

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT — see `LICENSE`.
