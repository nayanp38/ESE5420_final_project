"""Shared pieces: data split, models, weight-only quantization, Lipschitz bounds, FGSM.

Conventions (fixed before looking at any results):
  * Pixels are scaled to [0, 1] by dividing by 255. No standardization, so no statistic is
    learned from data and FGSM's epsilon is in raw pixel units.
  * The 60,000 MNIST training images are split ONCE (SPLIT_SEED) into 50,000 train and
    10,000 validation. The 10,000-image MNIST test set is never loaded by these scripts;
    it is touched once at the very end, after every choice is fixed.
  * Quantization is weight-only, symmetric, per-tensor: scale = max|W| / (2^(b-1) - 1),
    W_q = scale * clamp(round(W / scale), -qmax, qmax). Biases and activations stay fp32,
    so input gradients are exact and FGSM is not fooled by gradient masking.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

DATA_DIR = "data"
SPLIT_SEED = 0
N_VAL = 10_000
SEEDS = [0, 1, 2]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2]
BITS = {"fp32": None, "int8": 8, "int4": 4}
EPSILONS = [0.05, 0.1, 0.2]


def load_train_val():
    """Return (X_tr, y_tr, X_val, y_val) as float32 [0,1] arrays, from the MNIST train set only."""
    ds = datasets.MNIST(DATA_DIR, train=True, download=True)
    X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(X))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


class MLP(nn.Module):
    """Two-layer MLP: 784 -> 256 (ReLU) -> 10."""

    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def linear_from_sklearn(clf):
    """Wrap a fitted sklearn multinomial LogisticRegression as a torch Linear(784, 10)."""
    m = nn.Sequential(nn.Linear(784, 10))
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
        m[0].bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
    return m


def quantize_tensor(w, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def quantize_weights(model, bits):
    """Copy of model with every Linear weight quantized (symmetric, per-tensor); biases fp32."""
    import copy
    q = copy.deepcopy(model)
    if bits is None:
        return q
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(quantize_tensor(mod.weight, bits))
    return q


def spectral_bound(model):
    """Upper bound on the l2 Lipschitz constant of the logits: product of layer spectral norms
    (ReLU is 1-Lipschitz). Exact for a single linear layer."""
    b = 1.0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            b *= torch.linalg.matrix_norm(mod.weight, ord=2).item()
    return b


def jacobian_lower_bound(model, X, batch=2000):
    """Lower bound on the l2 Lipschitz constant: max over X of the spectral norm of the
    input-to-logit Jacobian."""
    best = 0.0
    for i in range(0, len(X), batch):
        x = X[i:i + batch]
        J = torch.func.vmap(torch.func.jacrev(lambda z: model(z.unsqueeze(0)).squeeze(0)))(x)
        best = max(best, torch.linalg.matrix_norm(J, ord=2).max().item())
    return best


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def fgsm_correct(model, X, y, eps):
    """Per-image 0/1 correctness under FGSM (Goodfellow et al. 2015), clipped to [0,1]."""
    x = X.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (g,) = torch.autograd.grad(loss, x)
    x_adv = (X + eps * g.sign()).clamp(0, 1)
    with torch.no_grad():
        return (model(x_adv).argmax(1) == y).float()


def bootstrap_ci(correct, n_boot=2000, seed=0):
    """Percentile 95% bootstrap interval for the mean of a 0/1 vector."""
    c = np.asarray(correct)
    rng = np.random.default_rng(seed)
    means = c[rng.integers(0, len(c), size=(n_boot, len(c)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
