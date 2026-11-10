"""Comparison: two-layer MLPs at each weight decay x seed, quantized fp32/int8/int4.

Training recipe is fixed in advance (not tuned): SGD, momentum 0.9, lr 0.05, batch 128,
15 epochs, coupled L2 weight decay on all parameters. All scores on the validation split.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C

EPOCHS, LR, BATCH = 15, 0.05, 128


def train(wd, seed, Xtr, ytr):
    torch.manual_seed(seed)
    model = C.MLP()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def main():
    torch.set_num_threads(8)
    os.makedirs("results", exist_ok=True)
    Xtr, ytr, Xv, yv = (torch.tensor(a) for a in C.load_train_val())
    runs = []
    for wd in C.WEIGHT_DECAYS:
        for seed in C.SEEDS:
            model = train(wd, seed, Xtr, ytr)
            run = {"wd": wd, "seed": seed, "quant": {}}
            for qname, bits in C.BITS.items():
                qm = C.quantize_weights(model, bits)
                entry = {"clean_acc": C.accuracy(qm, Xv, yv),
                         "lipschitz_upper": C.spectral_bound(qm),
                         "lipschitz_lower": C.jacobian_lower_bound(qm, Xv),
                         "fgsm": {str(e): float(C.fgsm_correct(qm, Xv, yv, e).mean())
                                  for e in C.EPSILONS}}
                run["quant"][qname] = entry
                print(f"wd {wd:g} seed {seed} {qname}: clean {entry['clean_acc']:.4f} "
                      f"L in [{entry['lipschitz_lower']:.2f}, {entry['lipschitz_upper']:.2f}] "
                      f"FGSM@0.1 {entry['fgsm']['0.1']:.4f}", flush=True)
            runs.append(run)
    with open("results/mlp.json", "w") as f:
        json.dump({"epochs": EPOCHS, "lr": LR, "batch": BATCH, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
