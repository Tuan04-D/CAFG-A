"""Paths and hyper-parameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# PLACEHOLDER: point this at the "processed" directory you downloaded.
# It must contain one sub-directory per corpus, e.g.
#   <DATA_ROOT>/karaone/MM05.npz, MM08.npz, ...
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"D:/PATH/TO/dataset/processed")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DATASET = "karaone"
N_CHANNELS = 62
CLASSES = (
    "/iy/", "/uw/", "/piy/", "/tiy/", "/diy/", "/m/", "/n/",
    "pat", "pot", "knew", "gnaw",
)
TOP_K = 3

SEEDS = (0, 1, 2, 3, 4)
N_SRCVAL_SUBJECTS = 3
N_CAL = 40
USE_EUCLIDEAN_ALIGNMENT = True

C_E, C_F, C_S = 1.0, 0.25, 0.07
ALWAYS_CONFIRM_K = 3

BACKBONE = dict(f1=8, depth=2, kernel_length=128, dropout=0.5)
TRAINING = dict(lr=1e-3, weight_decay=1e-2, batch_size=64, max_epochs=200,
                patience=40, smooth_window=11)
ADAPTATION = dict(steps=200, lr=1e-3, l2=1.0, n_folds=5)

ECE_BINS = 15
