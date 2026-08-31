"""Class-prior matching on unlabelled trials, and temperature scaling shrunk
toward a pooled source temperature."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

LOG_T_BOUNDS = (-4.0, 5.0)
GRID = np.linspace(LOG_T_BOUNDS[0], LOG_T_BOUNDS[1], 181)
SIGMA2_FLOOR = 1e-6


def fit_prior_shift(logits, step=0.5, n_iter=50):
    logits = np.asarray(logits, dtype=np.float64)
    k = logits.shape[1]
    target = np.log(np.full(k, 1.0 / k))
    shift = np.zeros(k)
    for _ in range(n_iter):
        mean = softmax(logits + shift, axis=1).mean(axis=0)
        shift = shift + step * (target - np.log(np.clip(mean, 1e-12, None)))
    return shift - shift.mean()


def apply_prior_shift(logits, shift):
    return np.asarray(logits, dtype=np.float64) + shift


def _nll(log_t, logits, y):
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / np.exp(log_t), axis=1)
    return -lp[np.arange(len(y)), y].mean()


def _minimise(objective):
    # The objective is flat in log T near chance accuracy, so scan a grid
    # before refining; a solver started from a single point does not move.
    values = np.array([objective(g) for g in GRID])
    start = float(GRID[int(np.argmin(values))])
    res = minimize(lambda v: objective(v[0]), x0=np.array([start]),
                   method="L-BFGS-B", bounds=[LOG_T_BOUNDS],
                   options={"maxiter": 50})
    best = float(res.x[0]) if res.fun <= values.min() else start
    return float(np.clip(best, *LOG_T_BOUNDS))


def fit_temperature(logits, y):
    return float(np.exp(_minimise(lambda v: _nll(v, logits, y))))


def source_temperature_dispersion(logits, y, subject):
    temps = [fit_temperature(logits[subject == s], y[subject == s])
             for s in np.unique(subject)]
    return float(max(np.var(np.log(temps), ddof=1), SIGMA2_FLOOR))


def shrinkage_strength(n_cal, sigma2_log_t):
    return 1.0 / (n_cal * max(sigma2_log_t, SIGMA2_FLOOR))


def fit_shrunk_temperature(logits, y, t0, lam):
    if not np.isfinite(lam):
        return float(t0)
    log_t0 = np.log(t0)

    def objective(v):
        return _nll(v, logits, y) + 0.5 * lam * (v - log_t0) ** 2

    return float(np.exp(_minimise(objective)))


def apply_temperature(logits, t):
    # float64: a large T shrinks the logit gaps, and in float32 neighbouring
    # classes can round to the same probability and flip the argmax.
    return softmax(np.asarray(logits, dtype=np.float64) / t, axis=1)
