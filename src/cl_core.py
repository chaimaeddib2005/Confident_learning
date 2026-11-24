"""Confident learning core: confident joint, joint estimation, error finding, metrics.

Conventions (same as the course notes):
  * P : (n, m) out-of-sample predicted probabilities,  P[k, i] = p_hat(y~ = i; x_k)
  * y : (n,)  given (noisy) labels in 0..m-1
  * matrices are indexed [noisy label i, true label j]
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Thresholds and confident joint
# ----------------------------------------------------------------------------
def class_thresholds(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """t_j = mean self-confidence of the examples whose given label is j (Eq. 2)."""
    m = P.shape[1]
    t = np.empty(m)
    for j in range(m):
        mask = y == j
        t[j] = P[mask, j].mean() if mask.any() else 1.0
    return t


def bucket_thresholds(P: np.ndarray, y: np.ndarray, b: np.ndarray, n_buckets: int) -> np.ndarray:
    """T[j, c] = mean of P[k, j] over examples with y_k = j and bucket b_k = c.

    Empty (class, bucket) cells fall back to the global class threshold t_j.
    """
    m = P.shape[1]
    t_global = class_thresholds(P, y)
    T = np.tile(t_global[:, None], (1, n_buckets))
    for j in range(m):
        for c in range(n_buckets):
            mask = (y == j) & (b == c)
            if mask.any():
                T[j, c] = P[mask, j].mean()
    return T


def entropy_buckets(P: np.ndarray, n_buckets: int) -> np.ndarray:
    """Label-free difficulty proxy: quantile bucket of the entropy of each row of P."""
    H = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1)
    if n_buckets <= 1:
        return np.zeros(len(P), dtype=int)
    edges = np.quantile(H, np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.digitize(H, edges)


def _count(P: np.ndarray, y: np.ndarray, thr: np.ndarray):
    """Shared counting step. thr is (n, m): the threshold each example uses for each class.

    Returns (C, y_star_hat, counted) where counted[k] is False if example k passes no threshold.
    Collisions are resolved by arg max over the classes that were passed (Eq. 1).
    """
    n, m = P.shape
    above = P >= thr
    counted = above.any(axis=1)
    masked = np.where(above, P, -np.inf)
    y_star = masked.argmax(axis=1)
    C = np.zeros((m, m), dtype=np.int64)
    np.add.at(C, (y[counted], y_star[counted]), 1)
    return C, y_star, counted


def confident_joint(P: np.ndarray, y: np.ndarray):
    """Standard confident joint (Algorithm 1). Returns (C, y_star_hat, counted, t)."""
    t = class_thresholds(P, y)
    thr = np.broadcast_to(t, P.shape)
    C, y_star, counted = _count(P, y, thr)
    return C, y_star, counted, t


def bucketed_confident_joint(P: np.ndarray, y: np.ndarray, n_buckets: int):
    """Confident joint with one threshold per (class, difficulty bucket). B=1 is standard CL."""
    b = entropy_buckets(P, n_buckets)
    T = bucket_thresholds(P, y, b, n_buckets)  # (m, B)
    thr = T[:, b].T  # (n, m): thr[k, j] = T[j, b_k]
    C, y_star, counted = _count(P, y, thr)
    return C, y_star, counted, T


def confusion_baseline(P: np.ndarray, y: np.ndarray):
    """C_confusion: assume y* = arg max p_hat. Returns (C, y_star_hat, counted)."""
    n, m = P.shape
    y_star = P.argmax(axis=1)
    C = np.zeros((m, m), dtype=np.int64)
    np.add.at(C, (y, y_star), 1)
    return C, y_star, np.ones(n, dtype=bool)


# ----------------------------------------------------------------------------
# Joint estimation (Algorithm 2) and derived statistics
# ----------------------------------------------------------------------------
def joint_estimate(C: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calibrate C so that row sums match |X_{y~=i}| and the total is 1 (Eq. 3).

    A row of C that is entirely zero (no confident example for that given label) is replaced by
    "all examples of that class are correct", i.e. all mass on the diagonal.
    """
    m = C.shape[0]
    counts = np.bincount(y, minlength=m).astype(float)
    C = C.astype(float).copy()
    for i in range(m):
        if C[i].sum() == 0:
            C[i, i] = 1.0
    C_tilde = C / C.sum(axis=1, keepdims=True) * counts[:, None]
    return C_tilde / C_tilde.sum()


def noise_statistics(Q: np.ndarray):
    """Latent prior, noise transition matrix p(y~=i|y*=j), mixing matrix p(y*=j|y~=i)."""
    prior = Q.sum(axis=0)
    transition = Q / np.where(prior == 0, 1, prior)
    row = Q.sum(axis=1, keepdims=True)
    mixing = Q / np.where(row == 0, 1, row)
    return prior, transition, mixing


def class_weights(Q: np.ndarray) -> np.ndarray:
    """w_i = Q_{y*}[i] / Q[i][i] (Eq. 5.1 of the course)."""
    prior = Q.sum(axis=0)
    diag = np.diag(Q)
    return prior / np.where(diag == 0, 1, diag)


# ----------------------------------------------------------------------------
# Flagging label errors
# ----------------------------------------------------------------------------
def flag_errors(y: np.ndarray, y_star: np.ndarray, counted: np.ndarray) -> np.ndarray:
    """Method 2 (off-diagonals of C): counted examples whose guessed true label differs from y~."""
    return counted & (y_star != y)


def normalized_margin(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p(y~=i;x) - max_{j != i} p(y~=j;x); most negative = most suspicious."""
    n = len(y)
    self_conf = P[np.arange(n), y]
    Q = P.copy()
    Q[np.arange(n), y] = -np.inf
    return self_conf - Q.max(axis=1)


# ----------------------------------------------------------------------------
# Metrics (need ground truth, so they are for experiments only)
# ----------------------------------------------------------------------------
def empirical_joint(y_noisy: np.ndarray, y_true: np.ndarray, m: int) -> np.ndarray:
    Q = np.zeros((m, m))
    np.add.at(Q, (y_noisy, y_true), 1)
    return Q / Q.sum()


def detection_metrics(flagged: np.ndarray, is_error: np.ndarray) -> dict:
    tp = int((flagged & is_error).sum())
    fp = int((flagged & ~is_error).sum())
    fn = int((~flagged & is_error).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_flagged": int(flagged.sum())}


def joint_rmse(Q_hat: np.ndarray, Q_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Q_hat - Q_true) ** 2)))
