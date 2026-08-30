"""Unlabelled prior matching and subject-adaptive shrinkage temperature (Eq. 3)."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

LOG_T_BOUNDS = (-4.0, 5.0)
GRID = np.linspace(LOG_T_BOUNDS[0], LOG_T_BOUNDS[1], 181)
SIGMA2_FLOOR = 1e-6


def fit_prior_shift(logits, step=0.5, n_iter=50):
    """Per-class logit shift driving the mean prediction onto the uniform prior.

    A cross-subject decoder lands on a new subject with a skewed marginal. The
    task prior is known and uniform, so the skew is removable with no labels,
    from the same unlabelled trials Euclidean Alignment already uses.
    """
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
    """Coarse grid scan before L-BFGS.

    The objective is flat in log T whenever the decoder is close to chance, so a
    solver started from a single point stops wherever it began.
    """
    values = np.array([objective(g) for g in GRID])
    start = float(GRID[int(np.argmin(values))])
    res = minimize(lambda v: objective(v[0]), x0=np.array([start]),
                   method="L-BFGS-B", bounds=[LOG_T_BOUNDS],
                   options={"maxiter": 50})
    best = float(res.x[0]) if res.fun <= values.min() else start
    return float(np.clip(best, *LOG_T_BOUNDS))


def fit_temperature(logits, y):
    """Unregularised MLE; pooled over source subjects this is T_0."""
    return float(np.exp(_minimise(lambda v: _nll(v, logits, y))))


def source_temperature_dispersion(logits, y, subject):
    temps = [fit_temperature(logits[subject == s], y[subject == s])
             for s in np.unique(subject)]
    return float(max(np.var(np.log(temps), ddof=1), SIGMA2_FLOOR))


def lambda_star(n_cal, sigma2_log_t):
    """Empirical-Bayes shrinkage strength for the averaged likelihood of Eq. (3)."""
    return 1.0 / (n_cal * max(sigma2_log_t, SIGMA2_FLOOR))


def fit_sasts_temperature(logits, y, t0, lam):
    """MAP temperature under log T_s ~ N(log T_0, sigma^2) (Eq. 3)."""
    if not np.isfinite(lam):
        return float(t0)
    log_t0 = np.log(t0)

    def objective(v):
        return _nll(v, logits, y) + 0.5 * lam * (v - log_t0) ** 2

    return float(np.exp(_minimise(objective)))


def apply_temperature(logits, t):
    """Scale in float64.

    A large T squeezes the logit gaps toward zero; in float32 two neighbouring
    classes can round to the same probability and flip the argmax, breaking the
    invariance temperature scaling is supposed to have.
    """
    return softmax(np.asarray(logits, dtype=np.float64) / t, axis=1)
