"""Build per-subject FEIS epoch archives from the raw recordings.

Trials ship pre-segmented into 5 s epochs at 256 Hz, so filtering is per epoch
and no resampling is needed. No ICA: 14 channels are too few for a stable
decomposition and the montage has no frontal-pole channels.

Band-pass and notch, common-average reference, baseline-correct against the
matching resting epoch, drop bad epochs, z-score per channel.
"""

import csv
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd

from eeg_preprocessing import (
    EPOCH_SAMPLES,
    TARGET_SFREQ,
    common_average_reference,
    drop_bad_epochs,
    filter_continuous,
    zscore_per_channel,
)
from paths import OUT_ROOT, RAW_ROOT

RAW_DIR = (RAW_ROOT / "feis" / "scottwellington-FEIS-7e726fd" / "experiments")
OUT_DIR = OUT_ROOT / "feis"
BANDPASS = (1.0, 43.0)
NOTCH = 50.0
CHANNEL_COLUMNS = [
    "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2",
    "P8", "T8", "F8", "AF4", "FC6", "F4",
]


def read_phase_csv(subject_dir, phase):
    with zipfile.ZipFile(subject_dir / f"{phase}.zip") as zf:
        with zf.open(zf.namelist()[0]) as fh:
            return pd.read_csv(BytesIO(fh.read()))


def epochs_from_dataframe(df):
    result = {}
    for epoch_id, group in df.groupby("Epoch"):
        arr = group[CHANNEL_COLUMNS].to_numpy(dtype=np.float64).T
        if arr.shape[1] == EPOCH_SAMPLES:
            result[epoch_id] = (arr, group["Label"].iloc[0])
    return result


def process_subject(subject):
    subject_dir = RAW_DIR / subject
    thinking = epochs_from_dataframe(read_phase_csv(subject_dir, "thinking"))
    resting = epochs_from_dataframe(read_phase_csv(subject_dir, "resting"))

    n_trials_raw = len(thinking)
    epochs, labels, trial_ids = [], [], []
    for epoch_id, (epoch, label) in sorted(thinking.items()):
        if epoch_id not in resting:
            continue
        baseline, _ = resting[epoch_id]
        epoch = filter_continuous(epoch, TARGET_SFREQ, *BANDPASS, NOTCH)
        epoch = common_average_reference(epoch)
        epochs.append(epoch - baseline.mean(axis=1, keepdims=True))
        labels.append(label)
        trial_ids.append(epoch_id)

    x = np.stack(epochs).astype(np.float32)
    y = np.array(labels)
    trial_id = np.array(trial_ids)

    keep = drop_bad_epochs(x)
    x, y, trial_id = zscore_per_channel(x[keep]), y[keep], trial_id[keep]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{subject}.npz", X=x, y=y, trial_id=trial_id,
                        ch_names=np.array(CHANNEL_COLUMNS), sfreq=TARGET_SFREQ,
                        n_trials_raw=n_trials_raw)
    return {"subject": subject, "n_trials_raw": n_trials_raw,
            "n_trials_kept": len(y), "n_classes": len(set(y.tolist()))}


def manifest_row(subject):
    arch = np.load(OUT_DIR / f"{subject}.npz", allow_pickle=True)
    return {"subject": subject, "n_trials_raw": int(arch["n_trials_raw"]),
            "n_trials_kept": len(arch["y"]),
            "n_classes": len(set(arch["y"].tolist()))}


def main():
    subjects = sorted(d.name for d in RAW_DIR.iterdir()
                      if d.is_dir() and d.name.isdigit())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.csv", "w", newline="") as fh:
        writer = None
        for subject in subjects:
            if (OUT_DIR / f"{subject}.npz").exists():
                print(f"skip, already processed: {subject}")
                row = manifest_row(subject)
            else:
                print(f"processing {subject} ...")
                row = process_subject(subject)
                print(f"  kept {row['n_trials_kept']}/{row['n_trials_raw']} trials")
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
    print(f"wrote {OUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
