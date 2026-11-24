"""Noise generators.

alpha = 0  -> class-conditional noise (the assumption of confident learning, as in the paper)
alpha > 0  -> instance-dependent noise: flip probability grows with the example's hardness.

Flip probability of an example x with true class j and hardness percentile u_x in (0, 1):

    q_j(x) = min(1, tau_j * (1 + alpha) * u_x ** alpha)

Because u_x is uniform inside each class, E[(1 + alpha) u^alpha] = 1, so the class-level noise rate
is tau_j for every alpha (when nothing is clipped): only the *dependence of noise on x* changes.
When an example flips, its new label i != j is drawn from the column D[:, j] of a (possibly sparse)
destination matrix.
"""
from __future__ import annotations

import warnings

import numpy as np


def make_destination_matrix(m: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """Column-stochastic m x m matrix D with zero diagonal; D[i, j] = p(new label i | flip, y*=j).

    `sparsity` is the fraction of off-diagonal entries forced to zero. Every column keeps at least
    one non-zero entry so a flip is always possible.
    """
    W = rng.random((m, m))
    np.fill_diagonal(W, 0.0)
    off = [(i, j) for i in range(m) for j in range(m) if i != j]
    n_zero = int(round(sparsity * len(off)))
    zero_idx = rng.permutation(len(off))[:n_zero]
    for k in zero_idx:
        i, j = off[k]
        W[i, j] = 0.0
    for j in range(m):  # guarantee a non-empty column
        if W[:, j].sum() == 0:
            i = rng.choice([a for a in range(m) if a != j])
            W[i, j] = 1.0
    return W / W.sum(axis=0, keepdims=True)


def hardness_percentile(P_clean: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """u_x in (0,1): percentile rank of hardness 1 - p_hat(y*|x) within the example's true class.

    P_clean should be out-of-sample probabilities from a model trained on CLEAN labels.
    u close to 1 means a hard example.
    """
    n, m = P_clean.shape
    hardness = 1.0 - P_clean[np.arange(n), y_true]
    u = np.empty(n)
    for j in range(m):
        idx = np.flatnonzero(y_true == j)
        order = np.argsort(np.argsort(hardness[idx], kind="stable"), kind="stable")
        u[idx] = (order + 0.5) / len(idx)
    return u


def flip_probability(u: np.ndarray, tau: float | np.ndarray, alpha: float) -> np.ndarray:
    q = np.asarray(tau) * (1.0 + alpha) * u ** alpha
    if np.any(q > 1.0):
        warnings.warn("flip probability clipped at 1: realized noise rate will be below tau")
    return np.minimum(q, 1.0)


def apply_noise(
    y_true: np.ndarray,
    D: np.ndarray,
    tau: float,
    u: np.ndarray,
    alpha: float,
    v: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Return noisy labels.

    v, w are uniform(0,1) arrays shared across alpha values (common random numbers):
    example k flips iff v[k] < q_j(x_k); the destination is the inverse-CDF of D[:, j] at w[k].
    """
    tau_arr = np.full(len(y_true), float(tau))
    q = flip_probability(u, tau_arr, alpha)
    flip = v < q
    y_noisy = y_true.copy()
    cdf = np.cumsum(D, axis=0)  # (m, m): cdf[:, j] = CDF of destination for true class j
    for k in np.flatnonzero(flip):
        j = y_true[k]
        y_noisy[k] = int(np.searchsorted(cdf[:, j], w[k], side="right"))
    y_noisy = np.minimum(y_noisy, D.shape[0] - 1)
    return y_noisy
