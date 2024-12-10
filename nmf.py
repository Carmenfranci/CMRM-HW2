import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from tqdm import tnrange
from typing import Tuple

EPS = np.finfo(float).eps


def nmf(V: np.ndarray,
         cost_func: str = 'KLDiv',
         num_iter: int = 30,
         init_W: np.ndarray = None,
         init_H: np.ndarray = None,
         fix_W: bool = False,
         cont_polyphony: int = 5,
         cont_length: int = 10,
         cont_grid: int = 5,
         cont_sparsen: Tuple = (1, 1),
         vis: bool = False)-> Tuple[np.ndarray, np.ndarray]:

    """Given a non-negative matrix V, find non-negative matrix factors W and H
    such that V ~ WH. 


    Args:
        V: np.ndarray
            N x M matrix to be factorized

        cost_func: str, default=KLDiv
            Distance measure which is used for the optimization. Values are 'EucDist' for Euclidean, or 'KLDiv' for
            KL-divergence.

        num_iter: int, default=30
           Number of iterations the algorithm will run.

        init_W: np.ndarray, default=None
            Initialized W matrix

        init_H: np.ndarray, default=None
            Initialized H matrix

        fix_W: bool, default=False
            Set True if templates W should be constant during the update process.

        cont_polyphony: int, default=5
            Parameter to control continuity in terms of polyphony.

        cont_length: int, default=10
            Number of templates which should be activated successively for enforced continuity constraints.

        cont_grid: int, default=5
            Indicates in which iterations of the NMF update procedure the continuity constraints should be enforced.

        cont_sparsen: Tuple, default=(1, 1)
            Parameter to control sparsity in terms of polyphony.

    Returns:
        W: np.ndarray
            NxK non-negative matrix factor

        H: np.ndarray
            KxM non-negative matrix factor
    """

    N, M = V.shape  # V matrix dimensions

    num_of_simul_act = cont_polyphony

    # V matrix factorization
    #  initialization of W and H
    W = init_W.copy()
    H = init_H.copy()

    energy_in_W = np.sum(W**2, axis=0).reshape(-1, 1)
    energyScaler = np.tile(energy_in_W, (1, H.shape[1]))

    s = np.array(cont_sparsen)
    assert np.mod(s[0], 2) == 1 and np.mod(s[1], 2) == 1, 'Sparsity parameter needs to be odd!'

    max_filt_kernel = np.zeros(s)
    max_filt_kernel[:, np.ceil(s[0] / 2).astype(int) - 1] = 1
    max_filt_kernel[np.ceil(s[0] / 2).astype(int) - 1, :] = 1

    for k in tnrange(num_iter, desc='Processing'):

        if np.mod(k, cont_grid) == 0:
            if s.max() > 1:
                H_filt = maximum_filter(H, footprint=max_filt_kernel, mode='constant')  # find max values in neighborhood

                cond = np.array(H != np.array(H_filt))
                H = np.where(cond, H * (1 - (k + 1) / num_iter), H)

            if num_of_simul_act < H.shape[1]:
                sort_vec = np.argsort(np.multiply(-H, energyScaler), axis=0)

                for j in range(H.shape[1]):
                    H[sort_vec[num_of_simul_act:, j], j] *= (1 - (k + 1) / num_iter)

            filt = np.eye(cont_length)
            H = convolve2d(H, filt, 'same')

        if cost_func == 'EucDist':  # euclidean update rules
            H *= (W.T @ V) / (W.T @ W @ H + EPS)

            if not fix_W:
                W *= (V @ H.T / ((W @ H @ H.T) + EPS))

        elif cost_func == 'KLDiv':  # divergence update rules
            H *= (W.T @ (V / (W @ H + EPS))) / (np.sum(W, axis=0).T.reshape(-1, 1) @ np.ones((1, M)) + EPS)

            if not fix_W:
                W *= ((V / (W @ H + EPS)) @ H.T) / (np.ones((N, 1)) @ np.sum(H, axis=1).reshape(1, -1) + EPS)

        else:
            raise ValueError('Unknown distance measure')

    return W, H
