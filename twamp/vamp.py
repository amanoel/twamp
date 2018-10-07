from time import time

import numpy as np
import h5py


def iterate(f_prox, g_prox, K,
            max_iter=100, tol=1e-4, damp=0.05,
            verbose=1, adaptive=True, rho=1.0,
            metric=None, weights_label=None):
    r"""Somewhat generic TV-VAMP iteration.

    Implement iteration (24)-(26) from arXiv:1809.06304 for minimizing
    a sum of functions :math:`f(x) + g(Kx)`.

    Parameters
    ----------
    f_prox: callback
        Callback that returns proximal of :math:`f` and its divergence.

    g_prox: callback
        Callback that returns proximal of :math:`g` and its divergence.

    K: np.ndarray, ndim=2
        Matrix multiplying :math:`x` in :math:`g(Kx)`.

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
        with stepsize :code:`rho`. Defaults to :code:`True`.

    rho: float, optional
        Initial stepsize :math:r`\rho`, defaults to :code:`1.0`.

    metric: callback, optional
        Additional callback to be evaluated and stored at each iteration.
        Defaults to :code:`None`.

    weights_label: str, optional
        If not :code:`None`, the weights will be stored in the
        :code:`weights.h5` file, under the directory specified by
        :code:`weights_label`.

    Returns
    -------
    x: np.ndarray, ndim=1
        Value obtained for the minimizer of :math:`f(x) + g(Kx)`

    iter_info: dict
        Dictionary containing values stored at each iteration; five keys are
        present: :code:`norm`, :code:`x_rho`, :code:`z_rho`, :code:`x_var`,
        :code:`z_var`. Additionally, if a :code:`metric` callback has been
        specified, its value at each iteration is stored under the key
        :code:`metric`.
            
    """
    # Precompute some useful quantities
    z_size, x_size = K.shape

    # Initialize variables
    x = np.zeros(x_size)
    z = np.zeros(z_size)
    B = np.zeros(z_size)

    x_var = z_var = 1. if adaptive else .5 / rho
    A = 1. if adaptive else rho

    # Set up weight storing
    if weights_label is not None:
        weights = h5py.File("weights.h5", "a")
        if weights_label in weights:
            del weights["/" + weights_label]
        weights.create_dataset("/" + weights_label + "/time",
                shape=(max_iter + 1,), dtype=np.float)
        weights["/" + weights_label + "/time"][0] = time()

    iter_vars = ["norm", "x_rho", "z_rho", "x_var", "z_var", "metric"]
    iter_info = dict([(var, []) for var in iter_vars])
    for t in range(max_iter):
        # Perform iteration
        x_old = np.copy(x)

        x, x_var_t = f_prox(A, B, compute_var=adaptive)
        if adaptive:
            x_var = x_var_t

        z, z_var_t = g_prox(1 / x_var - A, K.dot(x) / x_var - B,
                compute_var=adaptive)
        if adaptive:
            z_var = z_var_t

        A = A + (1. - damp) * (1 / z_var - 1 / x_var)
        B = B + (1. - damp) * (z / z_var - K.dot(x) / x_var)

        if verbose > 1:
            print("      x_var: %g, z_var: %g, A: %g" % (x_var, z_var, A))

        # Compute and store metrics
        norm = np.linalg.norm(x - x_old)

        iter_info["norm"].append(norm)
        iter_info["x_rho"].append(A)
        iter_info["z_rho"].append(1 / x_var - A)
        iter_info["x_var"].append(x_var)
        iter_info["z_var"].append(z_var)
        if metric is not None:  # compute and store additional callback
            iter_info["metric"].append(metric(x))

        # Print iteration status
        if verbose > 0:
            print("TV-VAMP: iter #%04d -- norm = %.2e" % (t + 1, norm))

        # Save weights to disk
        if weights_label is not None:
            weights.create_dataset("/" + weights_label + "/%d" % t, data=x)
            weights["/" + weights_label + "/time"][t + 1] = time()

        if norm < tol:
            if verbose > 0:
                print("Converged w/ norm = %.2e" % norm)
            break

    if weights_label is not None:
        weights.close()

    return x, iter_info
