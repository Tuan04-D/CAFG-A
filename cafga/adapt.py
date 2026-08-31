"""Backbone training, classifier-head adaptation on the calibration block, and
cross-fitted logits for that same block."""

import copy

import numpy as np
import torch
from torch import nn

from cafga.model import EEGNet


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def forward_logits(model, x, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        chunk = torch.from_numpy(x[i:i + batch]).to(device)
        out.append(model(chunk).float().cpu().numpy())
    return np.concatenate(out)


def train_backbone(x_tr, y_tr, x_va, y_va, n_channels, n_classes, seed, device,
                   backbone, training):
    set_seed(seed)
    model = EEGNet(n_channels, n_classes, x_tr.shape[2], **backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                            weight_decay=training["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_va_t = torch.from_numpy(x_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = {"score": np.inf, "epoch": -1, "state": None}
    window = []
    for epoch in range(training["max_epochs"]):
        model.train()
        perm = torch.randperm(len(x_tr_t), generator=gen).to(device)
        for i in range(0, len(perm), training["batch_size"]):
            idx = perm[i:i + training["batch_size"]]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x_tr_t[idx]), y_tr_t[idx]).backward()
            opt.step()

        # Early stopping tracks a smoothed balanced accuracy; validation
        # cross-entropy is minimised at epoch 0 while accuracy is still rising.
        model.eval()
        with torch.no_grad():
            pred = model(x_va_t).argmax(1)
            hit = (pred == y_va_t).float()
            per_class = torch.zeros(n_classes, device=device)
            counts = torch.zeros(n_classes, device=device)
            per_class.index_add_(0, y_va_t, hit)
            counts.index_add_(0, y_va_t, torch.ones_like(hit))
            seen = counts > 0
            bacc = float((per_class[seen] / counts[seen]).mean())

        window.append(-bacc)
        score = float(np.mean(window[-training["smooth_window"]:]))
        if score < best["score"] - 1e-6:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= training["patience"]:
            break

    model.load_state_dict(best["state"])
    model.eval()
    return model


def adapt_head(model, x_cal, y_cal, device, adaptation):
    adapted = copy.deepcopy(model)
    source = {k: v.detach().clone()
              for k, v in adapted.classifier.module.named_parameters()}
    for param in adapted.parameters():
        param.requires_grad_(False)
    for param in adapted.classifier.module.parameters():
        param.requires_grad_(True)

    opt = torch.optim.Adam(adapted.classifier.module.parameters(),
                           lr=adaptation["lr"])
    loss_fn = nn.CrossEntropyLoss()
    xb = torch.from_numpy(x_cal).to(device)
    yb = torch.from_numpy(y_cal).to(device)
    adapted.eval()
    with torch.no_grad():
        feats = adapted.features(xb)

    l2 = adaptation["l2"]
    for _ in range(adaptation["steps"]):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(adapted.classifier(feats), yb)
        for name, param in adapted.classifier.module.named_parameters():
            loss = loss + 0.5 * l2 * ((param - source[name]) ** 2).sum()
        loss.backward()
        opt.step()
    adapted.eval()
    return adapted


def cross_fitted_logits(model, target, idx_cal, n_classes, device, adaptation):
    # The head is fitted on the calibration block, so its in-sample logits
    # there are overconfident and unusable for calibration. Adapt on 4/5 of
    # the block and predict the held-out fifth instead.
    out = np.full((len(target.y), n_classes), np.nan)
    for held in np.array_split(np.arange(len(idx_cal)), adaptation["n_folds"]):
        if len(held) == 0 or len(held) == len(idx_cal):
            continue
        keep = np.setdiff1d(np.arange(len(idx_cal)), held)
        fit_idx, eval_idx = idx_cal[keep], idx_cal[held]
        fitted = adapt_head(model, target.x[fit_idx], target.y[fit_idx],
                            device, adaptation)
        out[eval_idx] = forward_logits(fitted, target.x[eval_idx], device)
    return out
