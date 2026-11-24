"""Run with:  python -m pytest tests -q   (or  python tests/test_cl.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cl_core as cl  # noqa: E402
import noise as nz  # noqa: E402

# The 9-example dog/fox/cow table from the course notes (0=dog, 1=fox, 2=cow)
P_TOY = np.array([[.9, .05, .05], [.7, .2, .1], [.1, .8, .1], [.1, .7, .2], [.6, .3, .1],
                  [.2, .6, .2], [.1, .1, .8], [.3, .3, .4], [.05, .05, .9]])
Y_TOY = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


def test_toy_thresholds_and_confident_joint():
    C, y_star, counted, t = cl.confident_joint(P_TOY, Y_TOY)
    assert np.allclose(t, [0.9 / 3 + 0.7 / 3 + 0.1 / 3, (0.7 + 0.3 + 0.6) / 3, (0.8 + 0.4 + 0.9) / 3])
    assert (C == np.array([[2, 1, 0], [1, 2, 0], [0, 0, 2]])).all()
    assert not counted[7]  # example 8 (index 7) is counted nowhere


def test_toy_confusion_baseline():
    C, _, _ = cl.confusion_baseline(P_TOY, Y_TOY)
    assert (C == np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3]])).all()


def test_toy_joint_flags_and_weights():
    C, y_star, counted, _ = cl.confident_joint(P_TOY, Y_TOY)
    Q = cl.joint_estimate(C, Y_TOY)
    assert np.allclose(Q, np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3]]) / 9)
    assert np.allclose(cl.class_weights(Q), [1.5, 1.5, 1.0])
    flagged = cl.flag_errors(Y_TOY, y_star, counted)
    assert set(np.flatnonzero(flagged)) == {2, 4}
    # most suspicious first: example 3 (index 2) has margin -0.7, example 5 (index 4) has -0.3
    m = cl.normalized_margin(P_TOY, Y_TOY)
    assert np.isclose(m[2], -0.7) and np.isclose(m[4], -0.3)


def test_figure1_calibration():
    C = np.array([[100, 40, 20], [56, 60, 0], [32, 12, 80]])
    y = np.repeat([0, 1, 2], C.sum(1))
    Q = cl.joint_estimate(C, y)
    assert np.allclose(Q, C / 400)
    prior, trans, mixing = cl.noise_statistics(Q)
    assert np.allclose(prior, [0.47, 0.28, 0.25])
    assert np.allclose(trans.sum(axis=0), 1) and np.allclose(mixing.sum(axis=1), 1)


def test_bucketed_with_one_bucket_equals_standard_cl():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), size=500)
    y = rng.integers(0, 4, 500)
    C1, *_ = cl.confident_joint(P, y)
    C2, *_ = cl.bucketed_confident_joint(P, y, 1)
    assert (C1 == C2).all()


def test_theorem1_exact_recovery_with_ideal_probabilities():
    """Ideal probabilities p_hat = noise rates of the true class => confident joint exact (Thm 1)."""
    rng = np.random.default_rng(1)
    m, n = 4, 20000
    T = np.array([[.8, .1, .05, .02], [.1, .75, .1, .05], [.05, .1, .8, .08], [.05, .05, .05, .85]])
    T = T / T.sum(0)  # T[i, j] = p(y~=i | y*=j)
    y_true = rng.choice(m, n, p=[.4, .3, .2, .1])
    y_noisy = np.array([rng.choice(m, p=T[:, j]) for j in y_true])
    P = T[:, y_true].T
    C, *_ = cl.confident_joint(P, y_noisy)
    true = np.zeros((m, m), int)
    np.add.at(true, (y_noisy, y_true), 1)
    assert (C == true).all()


def test_noise_keeps_class_level_rate_for_every_alpha():
    rng = np.random.default_rng(2)
    n, m, tau = 60000, 5, 0.3
    y_true = rng.integers(0, m, n)
    u = np.empty(n)
    for j in range(m):  # uniform hardness percentile inside each class
        idx = np.flatnonzero(y_true == j)
        u[idx] = (rng.permutation(len(idx)) + 0.5) / len(idx)
    D = nz.make_destination_matrix(m, 0.4, rng)
    v, w = rng.random(n), rng.random(n)
    rates = []
    for alpha in [0.0, 0.5, 1.0]:
        y_noisy = nz.apply_noise(y_true, D, tau, u, alpha, v, w)
        rates.append((y_noisy != y_true).mean())
        assert (D[np.arange(m), np.arange(m)] == 0).all()
        # noisy labels only land on allowed destinations
        for j in range(m):
            dest = np.unique(y_noisy[(y_true == j) & (y_noisy != y_true)])
            assert all(D[i, j] > 0 for i in dest)
    assert np.allclose(rates, tau, atol=0.01), rates
    # for alpha > 0 the flipped examples are harder than average, for alpha = 0 they are not
    y0 = nz.apply_noise(y_true, D, tau, u, 0.0, v, w)
    y1 = nz.apply_noise(y_true, D, tau, u, 1.0, v, w)
    assert abs(u[y0 != y_true].mean() - 0.5) < 0.01
    assert u[y1 != y_true].mean() > 0.6


def test_destination_matrix_properties():
    rng = np.random.default_rng(3)
    D = nz.make_destination_matrix(10, 0.6, rng)
    assert np.allclose(D.sum(axis=0), 1) and (np.diag(D) == 0).all()
    off = D[~np.eye(10, dtype=bool)]
    assert abs((off == 0).mean() - 0.6) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
