"""Turn the raw KaraOne recordings into per-subject epoch archives.

Band-pass and notch on the continuous recording, FastICA eye-artefact removal,
resample to 256 Hz, common-average reference, epoch the imagined-speech
("thinking") window, baseline-correct against that trial's own clearing
interval, drop bad epochs, then z-score per channel within subject.

Writes <DATA_ROOT>/processed/karaone/<subject>.npz plus a manifest.
"""

import csv
import glob

import mne
import numpy as np
import scipy.io as sio

from eeg_preprocessing import (
    EPOCH_SAMPLES,
    TARGET_SFREQ,
    common_average_reference,
    drop_bad_epochs,
    remove_eog_ica,
    resample_continuous,
    zscore_per_channel,
)
from paths import OUT_ROOT, RAW_ROOT

mne.set_log_level("ERROR")

RAW_DIR = RAW_ROOT / "karaone"
OUT_DIR = OUT_ROOT / "karaone"
NATIVE_SFREQ = 1000.0
BANDPASS = (1.0, 45.0)
NOTCH = 60.0
EOG_PROXY_CHANNELS = ["FP1", "FP2"]


def find_subject_dir(subject):
    matches = list(RAW_DIR.glob(f"{subject}/p/spoclab/users/*/EEG/data/{subject}"))
    if not matches:
        raise FileNotFoundError(f"no data directory for subject {subject}")
    return matches[0]


def load_labels(data_dir):
    with open(data_dir / "kinect_data" / "labels.txt") as fh:
        return [line.strip() for line in fh if line.strip()]


def load_epoch_inds(data_dir):
    mat = sio.loadmat(data_dir / "epoch_inds.mat")
    return mat["clearing_inds"][0], mat["thinking_inds"][0]


def process_subject(subject):
    data_dir = find_subject_dir(subject)
    set_file = glob.glob(str(data_dir / "*.set"))[0]

    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    ch_names = raw.ch_names
    raw.notch_filter(NOTCH, verbose=False)
    raw.filter(*BANDPASS, verbose=False)
    raw, ica_excluded = remove_eog_ica(raw, EOG_PROXY_CHANNELS)

    data = raw.get_data()
    data = resample_continuous(data, NATIVE_SFREQ, TARGET_SFREQ)
    data = common_average_reference(data)

    clearing, thinking = load_epoch_inds(data_dir)
    labels = load_labels(data_dir)
    scale = TARGET_SFREQ / NATIVE_SFREQ
    n_trials = min(len(clearing), len(thinking), len(labels))

    epochs, kept_labels, trial_ids = [], [], []
    n_times = data.shape[1]
    for i in range(n_trials):
        c_start, c_end = (clearing[i][0] * scale).astype(int)
        t_start = int(thinking[i][0][0] * scale)
        t_end = t_start + EPOCH_SAMPLES
        if t_end > n_times or c_end <= c_start:
            continue
        baseline = data[:, c_start:c_end]
        epochs.append(data[:, t_start:t_end] - baseline.mean(axis=1, keepdims=True))
        kept_labels.append(labels[i])
        trial_ids.append(i)

    x = np.stack(epochs).astype(np.float32)
    y = np.array(kept_labels)
    trial_id = np.array(trial_ids)

    keep = drop_bad_epochs(x)
    x, y, trial_id = zscore_per_channel(x[keep]), y[keep], trial_id[keep]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{subject}.npz", X=x, y=y, trial_id=trial_id,
                        ch_names=np.array(ch_names), sfreq=TARGET_SFREQ,
                        n_trials_raw=n_trials, n_ica_excluded=len(ica_excluded))
    return {"subject": subject, "n_trials_raw": n_trials,
            "n_trials_kept": len(y), "n_ica_excluded": len(ica_excluded),
            "n_classes": len(set(y.tolist()))}


def manifest_row(subject):
    arch = np.load(OUT_DIR / f"{subject}.npz", allow_pickle=True)
    return {"subject": subject, "n_trials_raw": int(arch["n_trials_raw"]),
            "n_trials_kept": len(arch["y"]),
            "n_ica_excluded": int(arch["n_ica_excluded"]),
            "n_classes": len(set(arch["y"].tolist()))}


def main():
    subjects = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
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
                print(f"  kept {row['n_trials_kept']}/{row['n_trials_raw']} trials,"
                      f" {row['n_ica_excluded']} ICA components removed")
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
    print(f"wrote {OUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
