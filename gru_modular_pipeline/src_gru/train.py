"""Training, prediction, submission, and ensembling helpers for the GRU M5 pipeline."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import M5WindowDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    X, y, X_cat, X_num, X_future_cal, X_future_event, X_price, end_pos, cfg
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Use the most recent forecast-origin window as validation."""
    val_end_pos = end_pos.max()
    train_mask = end_pos < val_end_pos
    val_mask = end_pos == val_end_pos

    arrays = {
        "train_mask": train_mask,
        "val_mask": val_mask,
    }

    train_ds = M5WindowDataset(
        X[train_mask], y[train_mask], X_cat[train_mask], X_num[train_mask],
        X_future_cal[train_mask], X_future_event[train_mask], X_price[train_mask]
    )
    val_ds = M5WindowDataset(
        X[val_mask], y[val_mask], X_cat[val_mask], X_num[val_mask],
        X_future_cal[val_mask], X_future_event[val_mask], X_price[val_mask]
    )

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
    )
    print(f"train windows: {len(train_ds):,} | val windows: {len(val_ds):,}")
    return train_loader, val_loader, arrays


def get_criterion(cfg):
    loss = cfg["training"].get("loss", "smooth_l1").lower()
    if loss == "mse":
        return nn.MSELoss()
    if loss == "mae":
        return nn.L1Loss()
    return nn.SmoothL1Loss()


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    n_obs = 0

    for xb, yb, cb, nb, fcb, feb, pb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        cb = cb.to(device)
        nb = nb.to(device)
        fcb = fcb.to(device)
        feb = feb.to(device)
        pb = pb.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(xb, cb, nb, fcb, feb, pb)
            loss = criterion(pred, yb)
            if train:
                loss.backward()
                optimizer.step()

        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        n_obs += batch_n

    return total_loss / max(1, n_obs)


def train_model(model, train_loader, val_loader, cfg, device):
    """Train model for fixed epochs; restore best validation state."""
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train loss={train_loss:.5f} | val loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best validation model: val loss={best_val:.5f}")

    return model, pd.DataFrame(history)


def predict_future_block(model, arrays, cfg, device) -> Tuple[np.ndarray, np.ndarray]:
    """Predict a block of 28 days from prebuilt arrays."""
    X, y, X_cat, X_num, X_future_cal, X_future_event, X_price = arrays
    ds = M5WindowDataset(X, y, X_cat, X_num, X_future_cal, X_future_event, X_price)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=bool(cfg["training"].get("pin_memory", False)),
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _, cb, nb, fcb, feb, pb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            nb = nb.to(device)
            fcb = fcb.to(device)
            feb = feb.to(device)
            pb = pb.to(device)
            pred_log = model(xb, cb, nb, fcb, feb, pb).cpu().numpy()
            preds.append(pred_log)

    pred_log = np.vstack(preds).astype("float32")
    pred_raw = np.expm1(pred_log)
    pred_raw = np.clip(pred_raw, 0, None).astype("float32")
    return pred_raw, pred_log


def build_submission(pred_raw: np.ndarray, sales_df: pd.DataFrame, sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Build M5 validation+evaluation submission. Evaluation copies same 28-day block by default."""
    f_cols = [f"F{i}" for i in range(1, pred_raw.shape[1] + 1)]
    base_ids = sales_df["id"].str.replace("_validation", "", regex=False).values

    preds_df = pd.DataFrame(pred_raw, columns=f_cols)
    preds_df["id"] = base_ids

    val = preds_df.copy()
    val["id"] = val["id"] + "_validation"
    eva = preds_df.copy()
    eva["id"] = eva["id"] + "_evaluation"

    sub = pd.concat([val, eva], axis=0, ignore_index=True)
    sub = sub.set_index("id").loc[sample_sub["id"]].reset_index()
    return sub


def save_submission(submission: pd.DataFrame, cfg: dict, suffix: str = "") -> str:
    out_path = cfg["output"]["submission_path"]
    if suffix:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_{suffix}{ext}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path} | shape={submission.shape}")
    return out_path


def ensemble_submissions(paths, weights=None, output_path="output/submission_ensemble.csv") -> pd.DataFrame:
    """Average multiple submission CSVs. IDs must match."""
    subs = [pd.read_csv(p) for p in paths]
    f_cols = [c for c in subs[0].columns if c.startswith("F")]

    for sub in subs[1:]:
        if not subs[0]["id"].equals(sub["id"]):
            raise ValueError("Submission IDs/order do not match. Align before ensembling.")

    if weights is None:
        weights = np.ones(len(subs)) / len(subs)
    weights = np.array(weights, dtype="float64")
    weights = weights / weights.sum()

    out = subs[0].copy()
    out[f_cols] = sum(w * s[f_cols] for w, s in zip(weights, subs))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved ensemble: {output_path}")
    return out
