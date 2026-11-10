"""Baseline: multinomial logistic regression with an L2 penalty.

Penalty tuned by stratified 5-fold CV inside the 50,000-image training split (repeated with
each seed's fold shuffle), refit on all 50,000, scored once on the 10,000-image validation
split, which the fit and the CV never saw. The MNIST test set is not loaded.
"""
import json
import os
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import common as C

warnings.filterwarnings("ignore", category=ConvergenceWarning)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # C = 1 / lambda (inverse L2 strength)
K = 5


def main():
    os.makedirs("results", exist_ok=True)
    Xtr, ytr, Xv, yv = C.load_train_val()
    Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv)
    runs = []
    for seed in C.SEEDS:
        cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        gs = GridSearchCV(LogisticRegression(max_iter=2000, tol=1e-4), {"C": C_GRID},
                          cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        cv_table = [{"C": c, "cv_mean": float(m), "cv_se": float(s / np.sqrt(K))}
                    for c, m, s in zip(C_GRID, gs.cv_results_["mean_test_score"],
                                       gs.cv_results_["std_test_score"])]
        model = C.linear_from_sklearn(gs.best_estimator_)
        run = {"seed": seed, "best_C": gs.best_params_["C"], "cv": cv_table, "quant": {}}
        for qname, bits in C.BITS.items():
            qm = C.quantize_weights(model, bits)
            with torch.no_grad():
                clean = (qm(Xv_t).argmax(1) == yv_t).float().numpy()
            entry = {"clean_acc": float(clean.mean()),
                     "clean_ci95": C.bootstrap_ci(clean),
                     "lipschitz_upper": C.spectral_bound(qm),
                     "fgsm": {}}
            for eps in C.EPSILONS:
                corr = C.fgsm_correct(qm, Xv_t, yv_t, eps).numpy()
                entry["fgsm"][str(eps)] = {"acc": float(corr.mean()),
                                           "se": float(corr.std(ddof=1) / np.sqrt(len(corr))),
                                           "ci95": C.bootstrap_ci(corr)}
            run["quant"][qname] = entry
            print(f"seed {seed} C={run['best_C']:g} {qname}: clean {entry['clean_acc']:.4f} "
                  f"L={entry['lipschitz_upper']:.2f} FGSM@0.1 {entry['fgsm']['0.1']['acc']:.4f}")
        runs.append(run)
    with open("results/baseline.json", "w") as f:
        json.dump({"C_grid": C_GRID, "K": K, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
