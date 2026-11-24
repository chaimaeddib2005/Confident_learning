"""Aggregate results.csv over seeds and make the figures used in the slides."""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"confusion": r"$C_{\mathrm{confusion}}$ (argmax)", "cl": "CL (standard thresholds)"}
STYLE = {"confusion": ("tab:gray", "s", "--"), "cl": ("tab:blue", "o", "-")}


def label(m):
    if m in LABELS:
        return LABELS[m]
    return f"CL bucketed (B={m[len('cl_bucket'):]})"


def style(m, i):
    if m in STYLE:
        return STYLE[m]
    return (["tab:red", "tab:green", "tab:orange"][i % 3], "^", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["method"] != "noisy_labels"] if "precision" in df else df
    metrics = [c for c in ["precision", "recall", "f1", "joint_rmse", "est_error_rate", "true_error_rate",
                           "u_missed", "u_false_pos", "test_acc"] if c in df]
    agg = df.groupby(["noise", "alpha", "method"])[metrics].agg(["mean", "std"])
    agg.to_csv(os.path.join(args.outdir, "summary.csv"))
    print(agg.round(3).to_string())

    noises = sorted(df["noise"].unique())
    for metric, title in [("f1", "Error-detection F1"), ("recall", "Recall"), ("precision", "Precision"),
                          ("joint_rmse", "RMSE of estimated joint")]:
        fig, axes = plt.subplots(1, len(noises), figsize=(4.6 * len(noises), 3.6), squeeze=False)
        for ax, tau in zip(axes[0], noises):
            sub = df[df["noise"] == tau]
            for i, meth in enumerate([m for m in sub["method"].unique()]):
                g = sub[sub["method"] == meth].groupby("alpha")[metric].agg(["mean", "std"]).fillna(0)
                c, mk, ls = style(meth, i - 2)
                ax.errorbar(g.index, g["mean"], yerr=g["std"], color=c, marker=mk, ls=ls, capsize=3,
                            label=label(meth))
            ax.set_title(f"noise = {int(tau * 100)}%")
            ax.set_xlabel(r"instance dependence $\alpha$  (0 = class-conditional)")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.outdir, f"{metric}_vs_alpha.{ext}"), dpi=200)
        plt.close(fig)
    print("figures written to", args.outdir)


if __name__ == "__main__":
    main()
