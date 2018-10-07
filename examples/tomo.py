import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import data_dir
from skimage.io import imread
from skimage.transform import rescale

from twamp import objective_functions
from twamp.solvers import tvvamp
from twamp.utils import create_radon_matrix


# matplotlib settings
plt.switch_backend("Agg")


def generate_problem(n_angles, snr=0.01, seed=42):
    """Load image and generate noisy projections"""

    # Load Shepp-Logan phantom
    im = imread(data_dir + "/phantom.png", as_gray=True)
    im = rescale(im, scale=0.25, mode="reflect", multichannel=False)

    # Generate projections
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    R = create_radon_matrix(im.shape[0], angles)
    y = R.dot(im.ravel())

    # Make projections noisy
    np.random.seed(seed)
    var_noise = snr * np.mean(y ** 2)
    noise = np.sqrt(var_noise) * np.random.randn(len(y))
    y = y + noise

    return im, R, y


def run_solver(n_angles, max_iter=1000, alpha=1.0, plot_interval=5):
    """Run TV-VAMP solver on reconstruction problem"""

    # Load data and perform some preprocessing
    im, R, y = generate_problem(n_angles)

    Rd = np.asarray(R.todense())
    mask = np.bool_(np.ones_like(im))

    # Run comparison
    print("\nRunning TV-VAMP...")
    start_time = time()
    x, hist = tvvamp(Rd, y, alpha=alpha, tol=0, max_iter=max_iter,
            damp=0.4, verbose=True, tv=True, mask=mask, woodbury=True,
            weights_label="vamp")
    obj_final = sum(objective_functions.tv(x, Rd, y, mask, alpha))
    print("TV-VAMP finished in %.2f seconds with objective %.8g" % \
            (time() - start_time, obj_final))

    lossreg = np.array([objective_functions.tv(e[0], Rd, y, mask, alpha) \
            for e in hist["metric"][::plot_interval]])
    objs_value = np.sum(lossreg, 1)
    objs_time = np.array([e[1] for e in hist["metric"]]) - start_time

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.semilogy(objs_time[::plot_interval], objs_value - objs_value[-1],
            lw=3, alpha=0.7)
    ax.set_ylabel("objective rel. to min.")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig("tomo.pdf")

    # Save results to disk
    os.rename("weights.h5", "tomo.h5")


def plot_reconstruction(n_angles, iter_vamp):
    # Load data
    weights = h5py.File("tomo.h5", "r")
    w_vamp = weights["vamp/%d" % iter_vamp].value
    weights.close()

    orig = imread(data_dir + "/phantom.png", as_gray=True)
    orig = rescale(orig, scale=0.25, mode='reflect', multichannel=False)

    # Plot images
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"hspace" : 0.5})
    axs[0].imshow(orig, interpolation="nearest",
            cmap="Greys_r", vmin=0, vmax=1)
    axs[1].imshow(w_vamp.reshape(orig.shape), interpolation="nearest",
            cmap="Greys_r", vmin=0, vmax=1)

    axs[0].set_title("original")
    axs[1].set_title("TV-VAMP")

    for ax in axs:
        ax.axis("off")

    fig.savefig("tomo-rec.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    n_angles = 20
    max_iter = 250

    run_solver(n_angles=n_angles, max_iter=max_iter)
    plot_reconstruction(n_angles=n_angles, iter_vamp=max_iter - 1)


if __name__ == "__main__":
    main()
