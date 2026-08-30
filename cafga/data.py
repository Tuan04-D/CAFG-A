"""Loading, Euclidean Alignment, and the leakage-checked LOSO splits."""

import hashlib
from dataclasses import dataclass

import numpy as np

from config import CLASSES, DATA_ROOT, DATASET, N_CHANNELS


@dataclass
class SubjectData:
    subject: str
    x: np.ndarray
    y: np.ndarray
    trial_id: np.ndarray


def list_subjects():
    return [f.stem for f in sorted((DATA_ROOT / DATASET).glob("*.npz"))]


def load_subject(subject):
    arch = np.load(DATA_ROOT / DATASET / f"{subject}.npz", allow_pickle=True)
    index = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([index[v] for v in arch["y"]], dtype=np.int64)
    x = arch["X"].astype(np.float32)
    assert x.shape[1] == N_CHANNELS, (x.shape, N_CHANNELS)
    return SubjectData(subject, x, y, arch["trial_id"].astype(np.int64))


def load_all():
    return {s: load_subject(s) for s in list_subjects()}


def euclidean_alignment(x, eps=1e-6):
    """Whiten a subject's trials by their mean spatial covariance (Eq. 1).

    Uses the trials only, never the labels, so it stays valid on the
    unlabelled target block.
    """
    cov = np.einsum("ncs,nds->cd", x, x, optimize=True) / (x.shape[0] * x.shape[2])
    cov = cov + eps * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, eps * w.max(), None)
    inv_sqrt = (v * w ** -0.5) @ v.T
    return np.einsum("cd,nds->ncs", inv_sqrt.astype(np.float32), x)


def align(subjects, enabled=True):
    if not enabled:
        return subjects
    return {k: SubjectData(v.subject, euclidean_alignment(v.x), v.y, v.trial_id)
            for k, v in subjects.items()}


def _stable_seed(seed, key):
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def make_fold(seed, target, subjects, n_srcval):
    others = [s for s in subjects if s != target]
    order = np.random.default_rng(_stable_seed(seed, target)).permutation(len(others))
    srcval = tuple(sorted(others[i] for i in order[:n_srcval]))
    train = tuple(sorted(others[i] for i in order[n_srcval:]))
    assert target not in train and target not in srcval
    assert not set(train) & set(srcval)
    return train, srcval


def calibration_split(y, trial_id, n_cal, seed, subject):
    """Split the target subject into the session block D_cal and D_test."""
    rng = np.random.default_rng(_stable_seed(seed, f"cal-{subject}"))
    pools = {c: rng.permutation(np.flatnonzero(y == c)).tolist()
             for c in np.unique(y)}
    order = []
    while len(order) < len(y):
        progressed = False
        for c in rng.permutation(list(pools)):
            if pools[c]:
                order.append(pools[c].pop())
                progressed = True
        if not progressed:
            break
    order = np.array(order, dtype=np.int64)
    cal_idx = np.sort(order[:n_cal])
    mask = np.ones(len(y), dtype=bool)
    mask[cal_idx] = False
    test_idx = np.flatnonzero(mask)
    assert not set(trial_id[cal_idx]) & set(trial_id[test_idx])
    return cal_idx, test_idx


def stack(subjects, names):
    x = np.concatenate([subjects[n].x for n in names])
    y = np.concatenate([subjects[n].y for n in names])
    return x, y
