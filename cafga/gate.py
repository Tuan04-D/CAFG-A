"""Cost-optimal confirmation gate: choose the candidate-list size, then accept or confirm."""

import numpy as np


def optimal_list_size(probs_sorted, c_e, c_s):
    above = probs_sorted > c_s / c_e
    k = above.cumprod(axis=1).sum(axis=1)
    return np.clip(k, 2, probs_sorted.shape[1]).astype(int)


def accept_decision(probs_sorted, k_star, c_e, c_f, c_s):
    cumulative = np.cumsum(probs_sorted, axis=1)
    runner_up = cumulative[np.arange(len(k_star)), k_star - 1] - probs_sorted[:, 0]
    budget = c_f / c_e + (c_s / c_e) * (k_star - 1)
    return runner_up <= budget


def cost_gate(probs, c_e, c_f, c_s):
    order = np.argsort(-probs, axis=1)
    probs_sorted = np.take_along_axis(probs, order, axis=1)
    k_star = optimal_list_size(probs_sorted, c_e, c_s)
    accept = accept_decision(probs_sorted, k_star, c_e, c_f, c_s)
    actions = np.where(accept, "accept", "confirm")
    lists = [order[i, :1] if accept[i] else order[i, :k_star[i]]
             for i in range(len(probs))]
    return actions, lists
