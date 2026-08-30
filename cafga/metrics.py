"""Discrimination, calibration and system-level metrics (Eq. 8-12)."""

import numpy as np


def top_k_accuracy(probs, y, k):
    order = np.argsort(-probs, axis=1)[:, :k]
    return float((order == y[:, None]).any(axis=1).mean())


def expected_calibration_error(probs, y, n_bins):
    """ECE with equal-mass bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    n = len(y)
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    edges = np.linspace(0, n, min(n_bins, n) + 1).astype(int)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            idx = order[lo:hi]
            ece += (hi - lo) / n * abs(correct[idx].mean() - conf[idx].mean())
    return float(ece)


def aurc(probs, y):
    """Area under the risk-coverage curve over kappa = top-1 probability."""
    conf = probs.max(axis=1)
    err = (probs.argmax(axis=1) != y).astype(float)
    order = np.argsort(-conf)
    return float((np.cumsum(err[order]) / np.arange(1, len(y) + 1)).mean())


def summarise(probs, y, actions, lists, c_e, c_f, c_s, always_k, top_k, ece_bins):
    pred = probs.argmax(axis=1)
    accept = actions == "accept"
    in_list = np.array([y[i] in lists[i] for i in range(len(y))])
    error = np.where(accept, pred != y, ~in_list).astype(float)
    size = np.array([len(c) for c in lists], dtype=float)
    cost = np.where(accept,
                    c_e * (pred != y),
                    c_f + c_s * (size - 1) + c_e * (~in_list))

    order = np.argsort(-probs, axis=1)[:, :always_k]
    miss = ~(order == y[:, None]).any(axis=1)
    cost_always = float(np.mean(c_f + c_s * (always_k - 1) + c_e * miss))

    confirm = ~accept
    return {
        "top1": top_k_accuracy(probs, y, 1),
        f"top{top_k}": top_k_accuracy(probs, y, top_k),
        "ece": expected_calibration_error(probs, y, ece_bins),
        "aurc": aurc(probs, y),
        "fir": float(confirm.mean()),
        "ewer": float(error.mean()),
        "cost": float(cost.mean()),
        "nic": float(cost.mean() / cost_always) if cost_always > 0 else float("nan"),
        "k_bar": float(size[confirm].mean()) if confirm.any() else 0.0,
        "n_test": int(len(y)),
    }
