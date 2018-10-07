"""
Compute proximal operators for different penalties
"""
import numpy as np


def threshold(x, alpha, tv=False, n_dims=1, hard=False):
    """Soft-thresholding or group soft-thresholding"""
    if tv:  # group soft-thresholding
        x_ = x.reshape(-1, n_dims)
        norm = np.sqrt(np.sum(x_ ** 2, 1))
        if hard:
            x_[norm < alpha] = 0.
        else:
            x_ *= np.maximum(0., 1. - alpha / norm[:, np.newaxis])
        return np.ravel(x_)
    else:  # soft-thresholding
        x_ = np.copy(x)
        if hard:
            x_[np.abs(x_) < alpha] = 0.
        else:
            nnz = (x != 0)
            x_[nnz] *= np.maximum(0., 1. - alpha / np.abs(x_[nnz]))
        return x_
