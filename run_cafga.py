"""Run the full leave-one-subject-out evaluation and summarise it."""

import argparse
import csv

import numpy as np
import torch

import config as cfg
from cafga.adapt import (
    adapt_head,
    cross_fitted_logits,
    forward_logits,
    train_backbone,
)
from cafga.calibrate import (
    apply_prior_shift,
    apply_temperature,
    fit_prior_shift,
    fit_shrunk_temperature,
    fit_temperature,
    shrinkage_strength,
    source_temperature_dispersion,
)
from cafga.data import (
    align,
    calibration_split,
    load_all,
    make_fold,
    stable_seed,
    stack,
)
from cafga.gate import cost_gate
from cafga.metrics import summarise

N_CLASSES = len(cfg.CLASSES)


def run_fold(seed, target, subjects, device):
    train_names, srcval_names = make_fold(
        seed, target, sorted(subjects), cfg.N_SRCVAL_SUBJECTS)
    x_tr, y_tr = stack(subjects, train_names)
    x_va, y_va = stack(subjects, srcval_names)
    subject_va = np.concatenate(
        [[n] * len(subjects[n].y) for n in srcval_names])
    target_data = subjects[target]

    model = train_backbone(x_tr, y_tr, x_va, y_va, cfg.N_CHANNELS, N_CLASSES,
                           stable_seed(seed, f"train-{target}") % (2 ** 31),
                           device, cfg.BACKBONE, cfg.TRAINING)

    idx_cal, idx_test = calibration_split(
        target_data.y, target_data.trial_id, cfg.N_CAL, seed, target)

    logits_srcval = forward_logits(model, x_va, device)
    t0 = fit_temperature(logits_srcval, y_va)
    sigma2 = source_temperature_dispersion(logits_srcval, y_va, subject_va)
    lam = shrinkage_strength(cfg.N_CAL, sigma2)

    adapted = adapt_head(model, target_data.x[idx_cal],
                         target_data.y[idx_cal], device, cfg.ADAPTATION)
    logits_adapted = forward_logits(adapted, target_data.x, device)
    logits_oof = cross_fitted_logits(model, target_data, idx_cal, N_CLASSES,
                                     device, cfg.ADAPTATION)

    shift = fit_prior_shift(logits_adapted)
    logits_test = apply_prior_shift(logits_adapted[idx_test], shift)
    logits_fit = apply_prior_shift(logits_oof[idx_cal], shift)
    t_s = fit_shrunk_temperature(logits_fit, target_data.y[idx_cal], t0, lam)

    probs = apply_temperature(logits_test, t_s)
    actions, lists = cost_gate(probs, cfg.C_E, cfg.C_F, cfg.C_S)
    row = summarise(probs, target_data.y[idx_test], actions, lists,
                    cfg.C_E, cfg.C_F, cfg.C_S, cfg.ALWAYS_CONFIRM_K,
                    cfg.TOP_K, cfg.ECE_BINS)
    row.update(seed=seed, subject=target, t0=t0, t_s=t_s, shrinkage=lam)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    subjects = align(load_all(), cfg.USE_EUCLIDEAN_ALIGNMENT)
    names = sorted(subjects)
    print(f"{len(names)} subjects, device={args.device}, n_cal={cfg.N_CAL}")

    rows = []
    for seed in args.seeds:
        for target in names:
            row = run_fold(seed, target, subjects, args.device)
            rows.append(row)
            print(f"  seed {seed} | {target:>5} | top1 {row['top1']:.3f} "
                  f"| NIC {row['nic']:.3f} | T_s {row['t_s']:.2f}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_DIR / "cafga_per_subject.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["top1", f"top{cfg.TOP_K}", "ece", "aurc", "fir", "ewer", "nic"]
    print("\nCAFG-A, mean +/- std over seeds of the dataset mean")
    for metric in metrics:
        per_seed = [np.mean([r[metric] for r in rows if r["seed"] == s])
                    for s in args.seeds]
        print(f"  {metric:6s} {np.mean(per_seed):.4f} +/- {np.std(per_seed, ddof=1):.4f}"
              if len(per_seed) > 1 else f"  {metric:6s} {np.mean(per_seed):.4f}")
    print(f"\nper-subject rows -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
