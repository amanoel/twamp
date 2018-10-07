import os

import numpy as np
import scipy.sparse as ssp


def create_tv_matrix(dims, mode="circular", cache_dir="./.cache/"):
    """Explicitly creates sparse gradient (i.e. TV) matrix"""
    n_dims = len(dims)
    n_features = np.product(dims)

    if n_dims > 3:
        raise ValueError("n_dims must be <= 3.")

    # Check if matrix was previously created
    suffix = "%d" % dims[0]
    if len(dims) > 1:
        suffix += "-%d" % dims[1]
    if len(dims) > 2:
        suffix += "-%d" % dims[2]
    filename = cache_dir + "operator_" + suffix + ".npz"

    rows = []
    cols = []
    data = []
    if os.path.exists(filename):
        mat = ssp.load_npz(filename)
    else:
        for k in range(n_features):
            rows += [n_dims * k + i for i in range(n_dims)]
            cols += n_dims * [k]
            if mode == "circular":
                data += n_dims * [-1]
            else:
                cond1 = ((k + 1) % dims[-1] > 0)
                data += [-1] if cond1 else [0]
                if n_dims > 1:
                    cond2 = ((np.floor(k / dims[-1] % dims[-2]) + 1) % dims[-2] > 0)
                    data += [-1] if cond2 else [0]
                    if n_dims > 2:
                        cond3 = (k + len_block < n_features)
                        data += [-1] if cond3 else [0]

            # 1st dimension (careful w/ last column)
            if (k + 1) % dims[-1] > 0:
                rows += [n_dims * k]
                cols += [k + 1]
                data += [1]
            elif mode == "circular":
                rows += [n_dims * k]
                cols += [k + 1 - dims[-1]]
                data += [1]

            # 2nd dimension (careful w/ last row)
            if n_dims > 1:
                len_block = dims[-1] * dims[-2]

                if (np.floor(k / dims[-1] % dims[-2]) + 1) % dims[-2] > 0:
                    rows += [n_dims * k + 1]
                    cols += [k + dims[-1]]
                    data += [1]
                elif mode == "circular":
                    rows += [n_dims * k + 1]
                    cols += [int(np.floor(k / len_block)) * len_block + k % dims[-1]]
                    data += [1]

                # 3rd dimension (careful w/ last block)
                if n_dims > 2:
                    if k + len_block < n_features:
                        rows += [n_dims * k + 2]
                        cols += [k + len_block]
                        data += [1]
                    elif mode == "circular":
                        rows += [n_dims * k + 2]
                        cols += [k % len_block]
                        data += [1]

        # NOTE: should we use CSC here or something else?
        mat = ssp.csr_matrix((data, (rows, cols)), (n_dims * n_features, n_features))
        if os.path.exists(cache_dir):
            ssp.save_npz(filename, mat)

    return mat


def create_radon_matrix(width, angles):
    """Explicitly creates sparse Radon matrix"""

    # Do it for one angle
    if np.isscalar(angles):
        radius = (width - 1.) / 2.
        theta = np.deg2rad(angles)
        y, x = np.ogrid[-radius:(radius + 1), -radius:(radius + 1)]

        # Perform rotation and change reference frame
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        idxs = np.round(radius + x_rot)

        # Construct Radon operator
        mask = (0 <= idxs) & (idxs < width)
        lines, cols = idxs[mask].flatten(), np.where(mask.ravel())[0]
        data = np.ones(np.sum(mask))
        op = ssp.coo_matrix( (data, (lines, cols)),
            shape = (width, width ** 2), dtype = np.int ).tocsr()

        return op

    # Do it for many angles
    else:
        ops = []
        for mu, angle in enumerate(angles):
            ops.append(create_radon_matrix(width, angle))
        return ssp.vstack(ops)
