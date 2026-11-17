"""Build the paper's tables and figures from results/*.json. Runs no experiments.

Writes paper/tables/*.tex and paper/figures/*.pdf. Run from the repository root:
    .venv/bin/python make_paper_assets.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import make_figure

QUANTS = ["fp32", "int8", "int4"]
EPS = ["0.05", "0.1", "0.2"]
WDS = [0.0, 1e-4, 1e-3, 1e-2]
WD_TEX = {0.0: "0", 1e-4: "$10^{-4}$", 1e-3: "$10^{-3}$", 1e-2: "$10^{-2}$"}
BLUE, GRAY, INK = "#2a78d6", "#6b6b68", "#3a3a38"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def cv_table(base):
    """Mean 5-fold CV accuracy per C, averaged over the three fold-shuffle seeds."""
    grid = base["C_grid"]
    mean = np.mean([[c["cv_mean"] for c in r["cv"]] for r in base["runs"]], axis=0)
    se = np.mean([[c["cv_se"] for c in r["cv"]] for r in base["runs"]], axis=0)
    best = int(np.argmax(mean))
    head = " & ".join(f"$10^{{{int(np.log10(c))}}}$" for c in grid)
    acc = " & ".join((r"\textbf{%s}" if i == best else "%s") % pct(m, 2) for i, m in enumerate(mean))
    ses = " & ".join(pct(s, 2) for s in se)
    return "\n".join([
        r"\begin{tabular}{l" + "c" * len(grid) + "}", r"\toprule",
        r"$C = 1/\lambda$ & " + head + r" \\", r"\midrule",
        r"CV accuracy (\%) & " + acc + r" \\",
        r"Fold SE (points) & " + ses + r" \\",
        r"\bottomrule", r"\end{tabular}"])


def baseline_table(base):
    """Logistic regression on validation; 95% bootstrap intervals. All seeds are identical."""
    run = base["runs"][0]
    rows = []
    for q in QUANTS:
        e = run["quant"][q]
        cells = [q, f"{pct(e['clean_acc'])} [{pct(e['clean_ci95'][0])}, {pct(e['clean_ci95'][1])}]"]
        for eps in EPS:
            f = e["fgsm"][eps]
            cells.append(f"{pct(f['acc'])} [{pct(f['ci95'][0])}, {pct(f['ci95'][1])}]")
        cells.append(f"{e['lipschitz_upper']:.2f}")
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{llllll}", r"\toprule",
        r" & & \multicolumn{3}{c}{FGSM accuracy (\%)} & \\", r"\cmidrule(lr){3-5}",
        r"Weights & Clean accuracy (\%) & $\epsilon = 0.05$ & $\epsilon = 0.1$ & $\epsilon = 0.2$ & $L$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def ms(vals, scale=100.0, d=1):
    v = np.asarray(vals) * scale
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}"


def mlp_table(mlp):
    """MLP on validation; mean +/- SD over seeds."""
    rows = []
    for wd in WDS:
        runs = [r for r in mlp["runs"] if r["wd"] == wd]
        for i, q in enumerate(QUANTS):
            es = [r["quant"][q] for r in runs]
            cells = [WD_TEX[wd] if i == 0 else "", q, ms([e["clean_acc"] for e in es])]
            cells += [ms([e["fgsm"][eps] for e in es]) for eps in EPS]
            cells += [ms([e["lipschitz_upper"] for e in es], 1.0, 2),
                      ms([e["lipschitz_lower"] for e in es], 1.0, 2)]
            rows.append(" & ".join(cells) + r" \\")
        if wd != WDS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join([
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r" & & & \multicolumn{3}{c}{FGSM accuracy (\%)} & \multicolumn{2}{c}{Lipschitz bounds} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-8}",
        r"Weight decay & Weights & Clean (\%) & $\epsilon = 0.05$ & $\epsilon = 0.1$ & $\epsilon = 0.2$ & Upper & Lower \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def diff_figure(base, mlp, out):
    """int4 minus fp32 FGSM accuracy, per seed (same trained network before and after rounding)."""
    plt.rcParams.update({"font.size": 8})
    labels = ["LR", "0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), dpi=150, sharey=True)
    for ax, eps in zip(axes, EPS):
        ax.axhline(0, color="#b5b4af", lw=0.8, zorder=1)
        e = base["runs"][0]["quant"]
        d = 100 * (e["int4"]["fgsm"][eps]["acc"] - e["fp32"]["fgsm"][eps]["acc"])
        ax.plot([0], [d], marker="D", ms=5, color=GRAY, mec="white", mew=0.8, ls="", zorder=3)
        for j, wd in enumerate(WDS, start=1):
            runs = sorted((r for r in mlp["runs"] if r["wd"] == wd), key=lambda r: r["seed"])
            ds = np.array([100 * (r["quant"]["int4"]["fgsm"][eps] - r["quant"]["fp32"]["fgsm"][eps])
                           for r in runs])
            ax.plot(j + np.linspace(-0.15, 0.15, len(ds)), ds, marker="o", ms=4, ls="",
                    mfc="none", mec=BLUE, mew=1.0, zorder=2)
            ax.plot([j - 0.28, j + 0.28], [ds.mean()] * 2, color=BLUE, lw=2, zorder=3,
                    solid_capstyle="round")
        ax.set_title(f"ε = {eps}", fontsize=8, loc="left")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel("LR, then MLP weight decay", fontsize=7)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4af")
        ax.tick_params(length=2, color="#b5b4af")
    axes[0].set_ylabel("int4 − fp32 FGSM accuracy\n(percentage points)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    base = json.load(open("results/baseline.json"))
    mlp = json.load(open("results/mlp.json"))
    os.makedirs("paper/tables", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    for name, tex in [("cv", cv_table(base)), ("baseline", baseline_table(base)),
                      ("mlp", mlp_table(mlp))]:
        with open(f"paper/tables/{name}.tex", "w") as f:
            f.write(tex + "\n")
        print(f"wrote paper/tables/{name}.tex")
    make_figure.main(out="paper/figures/key_figure.pdf", title=False, figsize=(5.5, 2.6), fontsize=8)
    diff_figure(base, mlp, "paper/figures/int4_minus_fp32.pdf")


if __name__ == "__main__":
    main()
