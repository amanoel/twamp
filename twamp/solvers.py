from time import time

import numpy as np
import scipy.sparse as ssp

from .vamp import iterate
from .proximal_operators import threshold
from .utils import create_tv_matrix


def tvvamp(X, y, alpha=0.1,
           max_iter=100, tol=1e-4, damp=0.05,
           verbose=1, adaptive=True, A=1.0, weights_label=None,
           tv=True, mask=None, woodbury=False):
    r"""Linear system solver wrapping the VAMP solver

    Determines the :math:`w` that minimizes :math:`l(y - Xw) + \alpha * r(w)`,
    for squared loss :math:`l` and different regularizers :math:`r`.

    Parameters
    ----------
    X: np.ndarray, ndim=2
        Feature matrix appearing in :math:`l(y - Xw)`

    y: np.ndarray, ndim=1
        Target vector appearing in :math:`l(y - Xw)`

    alpha: float, optional
        Regularization parameter, defaults to 0.1.

    Returns
    -------
    max_iter: int, optional
        Number of iterations, defaults to :code:`100`.

    tol: float, optional
        Tolerance used in stopping criterion; iteration halts when norm
        of difference between current and previous estimate is less than
        :code:`tol`. Defaults to :math:`10^{-4}`.

    damp: float, optional
        Damping factor (alternatively, one minus relaxation factor).
        Defaults to :code:`0.05`.

    verbose: int, optional
        Prints iteration status on screen if larger than :code:`0`, debugging
        information if larger than :code:`1`. Defaults to :code:`1`.

    adaptive: bool, optional
        When set to :code:`False`, executes instead Peaceman-Rachford splitting
        with stepsize :code:`A`. Defaults to :code:`True`.

    A: float, optional
        Initial stepsize, defaults to :code:`1.0`.

    weights_label: str, optional
        If not :code:`None`, the weights will be stored in the
        :code:`weights.h5` file, under the directory specified by
        :code:`weights_label`.

    tv: bool
        Uses TV penalty if set to :code:`True`, otherwise uses :code:`\ell_1`.

    mask: np.ndarray
        
    woodbury: bool
        Whether to use Woodbury trick, recommended when number of rows in
        :code:`X` is much smaller than the number of columns.

    Returns
    -------
    w: np.ndarray, ndim=1
        Value obtained for the minimizer of :math:`l(y - Xw) + \alpha * r(w)`.

    iter_info: dict
        Dictionary containing values stored at each iteration; six keys
        are present: :code:`norm`, :code:`x_rho`, :code:`z_rho`, :code:`x_var`,
        :code:`z_var` and :code:`metric`>
    """
    # Pre-compute some useful quantities
    n_samples, n_features = X.shape
    n_nonzeros = np.sum(np.sum(X, 0) != 0)

    if tv:
        if n_features != np.product(mask.shape):
            raise ValueError("n_features != np.product(mask.shape)")

    # For TV, explicitly creates sparse gradient/Laplacian; otherwise, identity
    # FIXME: can be a bit slow if n_features is large
    if tv:
        if verbose > 0:
            print("Creating TV matrix...")
        K = create_tv_matrix(mask.shape)
    else:
        K = ssp.eye(n_features).tocsc()
    KtK = K.T.dot(K)

    if verbose > 0:
        print("Pre-computing quantities (FFT, SVD, etc.)")

    # Define proximal of f, changes depending on `woodbury`/`tv`
    if woodbury:
        if tv:
            # Some pre-computations
            # \tilde{X} (2D FFT of each row of X)
            if verbose > 0:
                print("  - 2D FFT of the rows of X")
            Xt = np.fft.fftn(X.reshape((-1, *mask.shape)), s=mask.shape).reshape(-1, n_features).T
            iXt = Xt.conj().T / n_features

            # \Lambda (eigenvalues of KTK)
            if verbose > 0:
                print("  - eigenvalues of K^T K")
            first_col = np.asarray(KtK[:, 0].todense()).reshape(mask.shape)
            lamb = np.r_[1e7, 1. / np.real(np.fft.fftn(first_col)).ravel()[1:]]

            # \tilde{X}^T \Lambda^{-1} \tilde{X} and its eigendecomposition
            if verbose > 0:
                print(r"  - X^T \Lambda X = U diag(s) U^T")
            lXt = lamb[:, np.newaxis] * Xt
            s, U = np.linalg.eigh(np.real(iXt.dot(lXt)))
            print("  - X^T U")
            XtU = lXt.dot(U)

            def f_prox(A, B, compute_var=True):
                v_t = lamb * np.fft.fftn(K.T.dot(B / A).reshape(mask.shape)).ravel()
                y_r = U.T.dot(y - iXt.dot(v_t))
                w_t = v_t + XtU.dot(y_r / (s + A))

                mean = np.real(np.fft.ifftn(w_t.reshape(mask.shape))).ravel()
                var = (1 / A) * (1 - np.sum(s / (s + A)) / n_features) / np.ndim(mask)
                return mean, var

            print("Done!")
        else:
            # Some pre-computations
            s, U = np.linalg.eigh(X.dot(X.T))
            XtU = X.T.dot(U)

            def f_prox(A, B, compute_var=True):
                y_r = U.T.dot(y - X.dot(B / A))
                mean = (B / A) + XtU.dot(y_r / (s + A))
                var = (1 / A) * (1 - np.sum(s / (s + A)) / n_features)
                return mean, var
    else:
        # Some pre-computations
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        I = np.eye(n_features)

        def f_prox(A, B, compute_var=True):
            if compute_var:
                cov = np.asarray(np.linalg.inv(XtX + A * KtK))
                mean = cov.dot(Xty + K.T.dot(B))
                var = np.mean(np.diag(KtK.dot(cov)))
                if tv:
                    var /= np.ndim(mask)
            else:
                mean = np.linalg.solve(XtX + A * KtK, Xty + K.T.dot(B))
                var = None
            return mean, var

    # Define proximal of g, changes depending on whether or not `tv` is True
    if tv:
        def g_prox(A, B, compute_var=True):
            B_ = B.reshape(-1, np.ndim(mask))
            norm = np.sqrt(np.sum(B_ ** 2, 1))
            mean = np.maximum(0, 1. - alpha / norm[:, np.newaxis]) * B_ / A
            if compute_var:
                div = (1. - ((np.ndim(mask) - 1) / np.ndim(mask)) * alpha / norm) * (norm > alpha)
                var = np.maximum(1. / np.sum(mask), np.mean(div[mask.ravel()])) / A
            else:
                var = None
            return np.ravel(mean), var
    else:
        def g_prox(A, B, compute_var=True):
            mean = np.sign(B) * np.maximum(0, np.abs(B) - alpha) / A
            if compute_var:
                var = np.maximum(1. / n_features, np.mean(np.abs(B) > alpha)) / A
            else:
                var = None
            return mean, var

    # Define callback, changes depending on whether or not `tv` is True
    if tv:
        # NOTE: computing TV objective on-the-fly is quite inefficient
        def metric(w):
            return w, time()
    else:
        def metric(w):
            objective = .5 * np.sum((y - X.dot(w)) ** 2) + alpha * \
                np.sum(np.abs(w))
            return objective, time()

    # Call VAMP
    if verbose > 0:
        print("Running iteration...")
    w, iter_info = iterate(f_prox, g_prox, K,
                           max_iter=max_iter, tol=tol, damp=damp,
                           verbose=verbose, adaptive=adaptive, rho=A,
                           metric=metric, weights_label=weights_label)

    return w, iter_info
