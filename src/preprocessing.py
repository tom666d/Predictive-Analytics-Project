"""
preprocessing.py
────────────────
Loads raw M5 CSVs, melts wide→long, merges calendar and prices,
applies memory optimisations, and trims to the last N days.
 
Usage (from train.py or a notebook):
    from src.preprocessing import load_and_preprocess
    df, calendar, sell_prices = load_and_preprocess(cfg)
"""
 
import os
import numpy as np
import pandas as pd
 
 
def load_and_preprocess(cfg: dict):
    """
    Parameters
    ----------
    cfg : dict  (loaded from config.yaml)
 
    Returns
    -------
    df          : preprocessed long-format DataFrame
    calendar    : raw calendar DataFrame (needed for recursive prediction)
    sell_prices : raw sell_prices DataFrame (needed for recursive prediction)
    """
    data_path = cfg["data"]["path"]
    n_days    = cfg["data"]["n_days"]
 
    # ── Load raw files ───────────────────────────────────────
    print("Loading raw CSVs...")
    sales       = pd.read_csv(os.path.join(data_path, "sales_train_validation.csv"))
    calendar    = pd.read_csv(os.path.join(data_path, "calendar.csv"))
    sell_prices = pd.read_csv(os.path.join(data_path, "sell_prices.csv"))
 
    calendar["date"] = pd.to_datetime(calendar["date"])
 
    # ── Melt wide → long ─────────────────────────────────────
    print("Melting wide → long...")
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
 
    df = sales[id_cols + day_cols].melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
 
    # ── Merge calendar ───────────────────────────────────────
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    df = df.merge(calendar[cal_cols], on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])
 
    # ── Merge prices ─────────────────────────────────────────
    df = df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
 
    # ── SNAP: collapse 3 state columns → 1 ──────────────────
    df["snap"] = 0
    df.loc[df["state_id"] == "CA", "snap"] = df["snap_CA"]
    df.loc[df["state_id"] == "TX", "snap"] = df["snap_TX"]
    df.loc[df["state_id"] == "WI", "snap"] = df["snap_WI"]
    df.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)
 
    # ── Events: keep type only, fill NaN → "None" ───────────
    df["event_type"] = df["event_type_1"].fillna("None")
    df.drop(columns=["event_name_1", "event_name_2", "event_type_1", "event_type_2"],
            inplace=True)
 
    # ── Drop columns no longer needed ───────────────────────
    df.drop(columns=["d", "wm_yr_wk", "weekday", "year"], inplace=True)
 
    # ── Memory optimisation ──────────────────────────────────
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
 
    # ── Categorical dtypes ───────────────────────────────────
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
 
    # ── Trim to last N days ──────────────────────────────────
    cutoff = df["date"].max() - pd.Timedelta(days=n_days)
    df = df[df["date"] > cutoff].copy()
 
    # ── Sort ─────────────────────────────────────────────────
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
 
    print(f"Preprocessing done. df shape: {df.shape}")
    return df, calendar, sell_prices