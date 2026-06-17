"""
surrogate_model.py
==================
Supervised deep-learning surrogate that predicts the surface recombination
rate of a stochastic nanoporous thin-film solar cell from its extracted
geometric/statistical features.

Architecture and hyperparameters follow Table 1 of the manuscript:
  - 4 dense layers (128, 96, 64, 32 neurons)
  - activations: relu, tanh, tanh, tanh
  - output: 1 neuron, tanh
  - optimizer: Adam, lr = 1e-4
  - loss: mean squared error (MSE)
  - batch size: 10, validation split: 0.1

Dependencies: numpy, pandas, scikit-learn, tensorflow (see requirements.txt)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


RANDOM_SEED = 1   # fixed seed for reproducible splits and initialization


def normalize_features(X_train, X_other_list):
    """
    Min-max normalize features using TRAINING statistics only, then apply
    the same scaling to the other partitions. This prevents information
    leakage from validation/test/out-of-sample into training.

    Returns scaled X_train and a list of scaled "other" arrays.
    """
    X_train = np.asarray(X_train, dtype=float)
    xmin = X_train.min(axis=0)
    xmax = X_train.max(axis=0)
    rng = (xmax - xmin)
    rng[rng == 0] = 1.0  # avoid divide-by-zero for constant features

    def scale(X):
        return (np.asarray(X, dtype=float) - xmin) / rng

    return scale(X_train), [scale(X) for X in X_other_list]


def build_model(n_features):
    """Build the 4-layer surrogate per Table 1."""
    if not _HAS_TF:
        raise ImportError("TensorFlow is required to build the model.")
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(96, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(1, activation="tanh"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="mse",
                  metrics=["mae"])
    return model


def train_surrogate(X, y, epochs=200, batch_size=10, val_split=0.1):
    """
    Train the surrogate with a held-out validation split. Returns the
    trained model and the Keras History object (for the loss curve).
    """
    if not _HAS_TF:
        raise ImportError("TensorFlow is required to train the model.")
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = build_model(X.shape[1])
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=2,
    )
    return model, history


def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def plot_loss_curve(history, savepath="loss_curve.png"):
    """Plot training vs validation loss (the learning curve in Fig. 5c)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = history.history
    plt.figure(figsize=(5, 4))
    plt.plot(h["loss"], label="training")
    plt.plot(h["val_loss"], "--", label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
    plt.yscale("log"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(savepath, dpi=300)
    return savepath
