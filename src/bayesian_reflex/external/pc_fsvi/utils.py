"""SHIM written for this repo: your original utils.py was not uploaded. Assumed standard RBF kernel
rbf_kernel(X1, X2, lengthscale, variance) = variance * exp(-0.5 * ||x1-x2||^2 / lengthscale^2), inputs (N, d).
Replace with your own utils.py to remove the assumption."""
import numpy as np


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    X1, X2 = np.asarray(X1, float), np.asarray(X2, float)
    X1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
    X2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2
    d2 = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2.0 * X1 @ X2.T
    return variance * np.exp(-0.5 * np.maximum(d2, 0.0) / lengthscale ** 2)
