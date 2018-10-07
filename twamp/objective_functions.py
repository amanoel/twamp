"""
Compute objective functions associated to different problems
"""
import numpy as np

from .utils import create_tv_matrix


def l1(w, X, y, alpha=0.1):
    r"""Squared loss + :math`\ell_1` penalty"""
    loss = .5 * np.sum((y - X.dot(w)) ** 2)
    penalty = alpha * np.sum(np.abs(w))
    return loss, penalty


def tv(w, X, y, mask, alpha=0.1):
    r"""Squared loss + TV penalty"""

    # Assume problem is 1-dimensional if mask is not given
    if mask is None:
        mask = np.bool_(np.ones(len(w)))

    # Create gradient matrix explicitly (not very efficient)
    A = create_tv_matrix(mask.shape)
    if len(w) != np.product(mask.shape):
        w_full = np.zeros(mask.shape, dtype=w.dtype)
        w_full[mask] = w
        w_full = w_full.ravel()
        w_masked = w
    else:
        w_full = w
        w_masked = w.reshape(mask.shape)[mask].ravel()

    loss = .5 * np.sum((y - X.dot(w_masked)) ** 2)
    penalty = alpha * np.sum(np.linalg.norm(A.dot(w_full).reshape(-1,
            mask.ndim), axis=1))
    return loss, penalty
