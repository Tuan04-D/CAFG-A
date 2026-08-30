# CAFG-A

Reference implementation of **CAFG-A** (Calibration-Aware Feedback Gating with
Shrinkage Head Adaptation), the method proposed in

> *CAFG-A: Calibration-Aware Feedback Gating with Shrinkage Head Adaptation for
> Imagined-Speech BCI.*

An imagined-speech BCI wraps an error-prone EEG decoder in a visual confirmation
loop. Confirming every trial is safe but slow; confirming none is fast but
wrong. CAFG-A treats the confirmation decision as selective prediction under an
explicit user-interaction cost model, which yields a closed-form rule for
*whether* to confirm and for *how many* candidates to display — a rule that is
only valid if the decoder's posteriors are calibrated. To make them so, it
spends the short labelled block every session already records on the classifier
head rather than on a single temperature.

This repository contains **only the proposed method at its reported operating
point** (KaraOne, 11 prompts, `n_cal = 40`). Baselines, ablations, sweeps and the
figure/table pipeline of the paper are not included.

## Method

The pipeline has four stages, run inside every leave-one-subject-out fold:

| Stage | File | What it does |
|---|---|---|
| Euclidean Alignment (Eq. 1) | `cafga/data.py` | Whitens each subject's trials by their mean spatial covariance, using no labels. |
| Shrinkage head adaptation (Eq. 2) | `cafga/adapt.py` | Fine-tunes the classifier layer on the calibration block under an L2 penalty toward the source head. The backbone stays frozen. |
| Cross-fitting | `cafga/adapt.py` | Adapts on 4/5 of the block and predicts the held-out fifth, so the temperature is never fitted on the head's own in-sample logits. Without this step calibration error rises from 0.115 to 0.433. |
| Prior matching + SA-STS (Eq. 3) | `cafga/calibrate.py` | Removes the subject-specific marginal skew from unlabelled trials, then fits a temperature under a log-normal prior centred on the pooled source temperature. |
| Cost-optimal gate (Eq. 4–7) | `cafga/gate.py` | Picks the candidate-list size `k*` and decides accept vs. confirm in closed form. |

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12. A CUDA GPU is recommended but not required; the
code falls back to CPU automatically.

If you want a specific CUDA build of PyTorch, install it first from
<https://pytorch.org/get-started/locally/> and then run the `pip install` above.

## Dataset

This code consumes **preprocessed epoch archives**, not raw EEG. Download them
here:

<https://drive.google.com/drive/folders/1NccCSBwlaa9f8TmYOkl9SPJOHJRX00WY?usp=sharing>

Unpack so that the layout is:

```
<somewhere>/processed/
├── karaone/
│   ├── MM05.npz
│   ├── MM08.npz
│   └── ...
└── feis/
    └── ...
```

Each `.npz` holds `X` of shape `(n_trials, n_channels, 1280)` (5 s at 256 Hz,
band-pass filtered, baseline-corrected and z-scored per channel), `y` as string
prompt labels, and `trial_id`.

Then **edit `config.py`** and set the one placeholder:

```python
DATA_ROOT = Path(r"D:/PATH/TO/dataset/processed")   # <-- change this
```

Everything else in `config.py` already holds the values reported in the paper.

## Running

```bash
python run_cafga.py                # all 5 seeds, full LOSO
python run_cafga.py --seeds 0      # one seed, quick check
python run_cafga.py --device cpu   # force CPU
```

The script trains one backbone per (seed, held-out subject) fold, applies the
four stages above, and writes `outputs/cafga_per_subject.csv` with one row per
fold. It prints the headline summary:

```
CAFG-A, mean +/- std over seeds of the dataset mean
  top1   0.1268 +/- 0.0070
  top3   0.3460 +/- 0.0060
  ece    0.1150 +/- 0.0060
  aurc   0.8610 +/- 0.0090
  fir    0.0180 +/- 0.0220
  ewer   0.8601 +/- 0.0134
  nic    0.8226 +/- 0.0130
```

Runtime is roughly 2 GPU-minutes per fold, so ≈2.5 GPU-hours for 5 seeds × 14
subjects. A single seed takes about half an hour.

## Metrics

- **top1 / top3** — decoding accuracy.
- **ece** — expected calibration error, 15 equal-mass bins.
- **aurc** — area under the risk–coverage curve over the top-1 probability.
- **fir** — feedback invocation rate, the fraction of trials routed to the dialog.
- **ewer** — effective word error rate, the fraction of trials on which the user
  is left with the wrong item.
- **nic** — normalised interaction cost, the headline number: mean per-trial cost
  divided by the mean cost of always confirming with `k = 3`. Below 1 means
  cheaper than always confirming.

## Protocol notes

One fold is one (seed, held-out subject) pair. Within a fold the data splits
three ways: all remaining subjects except three train the backbone; three held-out
*source* subjects provide early stopping, the pooled temperature `T_0` and its
cross-subject dispersion; and the target subject is split into the `n_cal`-trial
calibration block and the test set. The target subject never enters training or
source validation, the calibration and test trials never overlap, and alignment
uses no labels — all asserted in `cafga/data.py`.

The costs are `c_e = 1`, `c_f = 0.25`, `c_s = 0.07`: one silent error, one
confirmation dialog, and one extra displayed candidate. Change them in
`config.py` to re-price the interaction.

## Layout

```
CAFG-A/
├── config.py            paths and the reported operating point
├── run_cafga.py         LOSO driver
├── requirements.txt
└── cafga/
    ├── data.py          loading, Euclidean Alignment, splits
    ├── model.py         EEGNet-8,2
    ├── adapt.py         backbone training, head adaptation, cross-fitting
    ├── calibrate.py     prior matching, SA-STS temperature
    ├── gate.py          cost-optimal gate
    └── metrics.py       accuracy, ECE, AURC, FIR, EWER, NIC
```

The KaraOne corpus is by Zhao and Rudzicz (ICASSP 2015); FEIS is by Wellington
and Clayton (Zenodo, 2019). Please cite them if you use the preprocessed
archives.
