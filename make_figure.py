"""Key figure: Lipschitz upper bound vs FGSM accuracy at eps=0.1 (validation split).

Marker = quantization, color = weight decay (ordinal blue ramp), logistic regression in gray.
Each point is the mean over 3 seeds; error bars are +/- 1 SD across seeds on both axes.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

EPS = "0.1"
MARKERS = {"fp32": "o", "int8": "s", "int4": "^"}
WD_COLORS = {0.0: "#86b6ef", 1e-4: "#3987e5", 1e-3: "#1c5cab", 1e-2: "#0d366b"}
LR_COLOR = "#6b6b68"


def style(q, col):
    """fp32 = large hollow ring drawn behind, so the int8 marker that sits on it stays visible."""
    if q == "fp32":
        return dict(ms=13, mfc="none", mec=col, mew=1.6, zorder=2)
    return dict(ms=7, mfc=col, mec="white", mew=1.0, zorder=3)


def mean_sd(vals):
    v = np.asarray(vals)
    return v.mean(), v.std(ddof=1)


def main():
    base = json.load(open("results/baseline.json"))
    mlp = json.load(open("results/mlp.json"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    for q, mk in MARKERS.items():
        L = [r["quant"][q]["lipschitz_upper"] for r in base["runs"]]
        A = [r["quant"][q]["fgsm"][EPS]["acc"] for r in base["runs"]]
        (lm, ls), (am, as_) = mean_sd(L), mean_sd(A)
        ax.errorbar(lm, 100 * am, xerr=ls, yerr=100 * as_, fmt=mk, color=LR_COLOR,
                    ecolor=LR_COLOR, elinewidth=1, capsize=0, **style(q, LR_COLOR))
        for wd, col in WD_COLORS.items():
            runs = [r for r in mlp["runs"] if r["wd"] == wd]
            L = [r["quant"][q]["lipschitz_upper"] for r in runs]
            A = [r["quant"][q]["fgsm"][EPS] for r in runs]
            (lm, ls), (am, as_) = mean_sd(L), mean_sd(A)
            ax.errorbar(lm, 100 * am, xerr=ls, yerr=100 * as_, fmt=mk, color=col,
                        ecolor=col, elinewidth=1, capsize=0, **style(q, col))

    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([3, 5, 7.5, 10, 15, 20, 30]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 70)
    ax.set_xlabel("Lipschitz upper bound of logits (product of spectral norms, log scale)")
    ax.set_ylabel("FGSM accuracy at ε = 0.1 (%, validation)")
    ax.set_title("Robust accuracy vs Lipschitz bound, MNIST validation split (mean ± 1 SD, 3 seeds)",
                 fontsize=9.5, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4af")

    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=8, color=LR_COLOR, label="Logistic reg. (baseline)")]
    h += [Line2D([], [], ls="", marker="o", ms=8, color=c, label=f"MLP, weight decay {w:g}")
          for w, c in WD_COLORS.items()]
    h += [Line2D([], [], ls="", marker=m, color="#3a3a38", label=q, **style(q, "#3a3a38"))
          for q, m in MARKERS.items()]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig("baseline.png", bbox_inches="tight")
    print("wrote baseline.png")


if __name__ == "__main__":
    main()
