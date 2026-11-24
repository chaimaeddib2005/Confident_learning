"""Experiments: confident learning under class-conditional (alpha=0) vs instance-dependent (alpha>0) noise.

Example (smoke test, offline):
    python src/run_experiments.py --dataset digits --noise 0.2 0.4 --alphas 0 0.5 1 --seeds 3 --eval-acc

Example (paper's dataset, fast backend: frozen ResNet-18 features + logistic regression):
    python src/run_experiments.py --dataset cifar10 --backend logreg --data-root data --eval-acc

Example (closer to the paper: small ResNet trained from scratch, 4-fold; needs a GPU):
    python src/run_experiments.py --dataset cifar10 --backend cnn --epochs 20 --seeds 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import cl_core as cl  # noqa: E402
import noise as nz  # noqa: E402
import probs as pr  # noqa: E402


def get_probs(ds, y_lab, args, seed):
    if args.backend == "logreg":
        return pr.oof_probs_logreg(ds["X_train"], y_lab, ds["m"], k=args.folds, seed=seed)
    return pr.oof_probs_cnn(ds["images_train"], y_lab, ds["m"], k=args.folds, seed=seed, epochs=args.epochs)


def train_eval_accuracy(ds, y_lab, keep, weights):
    """Retrain logistic regression on the kept examples (class-weighted) and score on the test set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = ds["X_train"][keep], y_lab[keep]
    sw = weights[ytr] if weights is not None else None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=200).fit(sc.transform(Xtr), ytr, sample_weight=sw)
    return float((clf.predict(sc.transform(ds["X_test"])) == ds["y_test"]).mean())


def run_method(name, P, y_noisy, args):
    if name == "confusion":
        C, y_star, counted = cl.confusion_baseline(P, y_noisy)
    elif name == "cl":
        C, y_star, counted, _ = cl.confident_joint(P, y_noisy)
    elif name.startswith("cl_bucket"):
        C, y_star, counted, _ = cl.bucketed_confident_joint(P, y_noisy, int(name[len("cl_bucket"):]))
    else:
        raise ValueError(name)
    Q_hat = cl.joint_estimate(C, y_noisy)
    flagged = cl.flag_errors(y_noisy, y_star, counted)
    return Q_hat, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "digits"], default="digits")
    ap.add_argument("--backend", choices=["logreg", "cnn"], default="logreg")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--sparsity", type=float, default=0.4)
    ap.add_argument("--buckets", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20, help="cnn backend only")
    ap.add_argument("--eval-acc", action="store_true", help="also retrain on cleaned data (logreg backend)")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    if args.dataset == "digits":
        from data import load_digits_dataset
        ds = load_digits_dataset()
    else:
        from data import load_cifar10_dataset
        ds = load_cifar10_dataset(args.data_root, args.backend)
    y_true, m = ds["y_train"], ds["m"]
    n = len(y_true)
    if args.eval_acc and args.backend != "logreg":
        print("--eval-acc is only implemented for the logreg backend; ignoring.")
        args.eval_acc = False

    # Hardness: out-of-sample probabilities of a model trained on CLEAN labels (computed once).
    t0 = time.time()
    P_clean = get_probs(ds, y_true, args, seed=0)
    u = nz.hardness_percentile(P_clean, y_true)
    print(f"clean model: OOF accuracy {(P_clean.argmax(1) == y_true).mean():.3f}  ({time.time() - t0:.0f}s)")

    methods = ["confusion", "cl"] + [f"cl_bucket{b}" for b in args.buckets]
    rows = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        D = nz.make_destination_matrix(m, args.sparsity, rng)  # same destinations for all noise / alpha
        v, w = rng.random(n), rng.random(n)  # common random numbers across alpha
        for tau in args.noise:
            for alpha in args.alphas:
                y_noisy = nz.apply_noise(y_true, D, tau, u, alpha, v, w)
                is_err = y_noisy != y_true
                Q_true = cl.empirical_joint(y_noisy, y_true, m)
                P = get_probs(ds, y_noisy, args, seed=seed)
                base = {"dataset": args.dataset, "backend": args.backend, "seed": seed, "noise": tau,
                        "alpha": alpha, "sparsity": args.sparsity, "true_error_rate": float(is_err.mean())}
                if args.eval_acc:
                    rows.append({**base, "method": "noisy_labels",
                                 "test_acc": train_eval_accuracy(ds, y_noisy, np.ones(n, bool), None)})
                for name in methods:
                    Q_hat, flagged = run_method(name, P, y_noisy, args)
                    met = cl.detection_metrics(flagged, is_err)
                    row = {**base, "method": name, **met,
                           "joint_rmse": cl.joint_rmse(Q_hat, Q_true),
                           "est_error_rate": float(1.0 - np.trace(Q_hat)),
                           "u_missed": float(u[is_err & ~flagged].mean()) if (is_err & ~flagged).any() else np.nan,
                           "u_false_pos": float(u[~is_err & flagged].mean()) if (~is_err & flagged).any() else np.nan}
                    if args.eval_acc:
                        row["test_acc"] = train_eval_accuracy(ds, y_noisy, ~flagged, cl.class_weights(Q_hat))
                    rows.append(row)
                    print(f"seed={seed} noise={tau} alpha={alpha} {name:12s} "
                          f"P={met['precision']:.3f} R={met['recall']:.3f} F1={met['f1']:.3f} "
                          f"rmse={row['joint_rmse']:.4f}")
                pd.DataFrame(rows).to_csv(args.out, index=False)  # save as we go
    print("saved", args.out)


if __name__ == "__main__":
    main()
