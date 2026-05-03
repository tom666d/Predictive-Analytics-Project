"""Feature engineering for the modular GRU M5 pipeline.

This module keeps the v4 notebook logic: lag/rolling features, future calendar
features, event embeddings, and optional price features.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def prepare_calendar_features(calendar: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Prepare normalized numeric calendar features and event ids indexed by d_num."""
    cal = calendar.copy()
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal["date"] = pd.to_datetime(cal["date"])

    cal["dayofweek"] = cal["date"].dt.dayofweek.astype("float32")
    cal["month"] = cal["date"].dt.month.astype("float32")
    cal["weekofyear"] = cal["date"].dt.isocalendar().week.astype("float32")

    event_col = cfg["features"].get("event_col", "event_type_1")
    cal[event_col] = cal[event_col].fillna("none").astype(str)
    event_map = {v: i for i, v in enumerate(sorted(cal[event_col].unique()))}
    cal["event_type_id"] = cal[event_col].map(event_map).astype("int64")

    cal_num_cols = cfg["features"]["calendar_numeric_cols"]
    cal_num = cal.set_index("d_num")[cal_num_cols].copy().astype("float32")

    # Normalize cyclical/calendar scales into roughly 0-1 ranges.
    if "dayofweek" in cal_num:
        cal_num["dayofweek"] /= 6.0
    if "month" in cal_num:
        cal_num["month"] /= 12.0
    if "weekofyear" in cal_num:
        cal_num["weekofyear"] /= 53.0
    if "wday" in cal_num:
        cal_num["wday"] /= 7.0

    event_ids = cal.set_index("d_num")["event_type_id"].astype("int64")
    return cal_num, event_ids, event_map


def make_lag_roll_features(series_raw, end_idx, cfg):
    """
    Create original lag and rolling mean features for one series at forecast origin end_idx.
    """
    lags = cfg["features"].get("lags", [1, 7, 28, 365])
    rolling_means = cfg["features"].get("rolling_means", [3, 7, 28])

    feats = []

    for lag in lags:
        idx = end_idx - lag
        if idx >= 0:
            feats.append(float(series_raw[idx]))
        else:
            feats.append(0.0)

    hist = series_raw[:end_idx]

    for window in rolling_means:
        if len(hist) == 0:
            feats.append(0.0)
        else:
            recent = hist[-window:] if len(hist) >= window else hist
            feats.append(float(np.mean(recent)))

    return np.array(feats, dtype="float32")

def make_lag_roll_activity_features(series_raw, end_idx, cfg):
    """
    Original lag/rolling features + optional activity features.

    Activity v2 features:
    - nonzero_ratio_28
    - nonzero_ratio_90
    - log1p(mean_nonzero_90)
    """
    base = make_lag_roll_features(series_raw, end_idx, cfg)

    use_activity = bool(cfg["features"].get("use_activity_features", False))
    if not use_activity:
        return base.astype("float32")

    activity_version = cfg["features"].get("activity_version", "v2")

    hist = series_raw[:end_idx]
    if len(hist) == 0:
        hist = np.array([0.0], dtype="float32")

    recent_28 = hist[-28:] if len(hist) >= 28 else hist
    recent_90 = hist[-90:] if len(hist) >= 90 else hist

    nonzero_ratio_28 = float(np.mean(recent_28 > 0)) if len(recent_28) else 0.0
    nonzero_ratio_90 = float(np.mean(recent_90 > 0)) if len(recent_90) else 0.0

    nonzero_90 = recent_90[recent_90 > 0]
    mean_nonzero_90 = float(nonzero_90.mean()) if len(nonzero_90) else 0.0

    if activity_version == "v2":
        activity = np.array([
            nonzero_ratio_28,
            nonzero_ratio_90,
            np.log1p(mean_nonzero_90),
        ], dtype="float32")
    else:
        raise ValueError(f"Unsupported activity_version: {activity_version}")

    return np.concatenate([base, activity]).astype("float32")


def build_price_matrix(
    sales_df: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    n_days: int,
) -> Optional[np.ndarray]:
    """Build a series x day price matrix. Missing prices are forward/back filled."""
    if prices is None or prices.empty:
        print("Price file missing; price features disabled.")
        return None

    cal = calendar.copy()
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal[(cal["d_num"] >= 1) & (cal["d_num"] <= n_days)]

    key_df = sales_df[["item_id", "store_id"]].copy().reset_index().rename(columns={"index": "row_idx"})
    price_matrix = np.full((len(sales_df), n_days), np.nan, dtype="float32")

    print("Building price matrix...")
    for wk, day_group in cal.groupby("wm_yr_wk"):
        day_idx = day_group["d_num"].values.astype(int) - 1
        wk_prices = prices.loc[prices["wm_yr_wk"] == wk, ["item_id", "store_id", "sell_price"]]
        if wk_prices.empty:
            continue
        merged = key_df.merge(wk_prices, on=["item_id", "store_id"], how="left")
        valid = merged["sell_price"].notna().values
        if valid.any():
            rows = merged.loc[valid, "row_idx"].values.astype(int)
            vals = merged.loc[valid, "sell_price"].values.astype("float32")
            price_matrix[np.ix_(rows, day_idx)] = vals[:, None]

    # Fill missing values row-wise using pandas for convenience.
    price_df = pd.DataFrame(price_matrix)
    price_df = price_df.ffill(axis=1).bfill(axis=1).fillna(0.0)
    return price_df.values.astype("float32")


def make_price_features(price_series: Optional[np.ndarray], end_idx: int, target_start: int, target_end: int) -> np.ndarray:
    """Create compact price features for one training/prediction sample."""
    if price_series is None:
        return np.zeros(0, dtype="float32")

    hist = price_series[:target_start]
    future = price_series[target_start:target_end]
    hist_nonzero = hist[hist > 0]

    last_price = float(hist_nonzero[-1]) if len(hist_nonzero) else 0.0
    future_nonzero = future[future > 0]
    future_price_mean = float(future_nonzero.mean()) if len(future_nonzero) else last_price
    hist_mean = float(hist_nonzero.mean()) if len(hist_nonzero) else max(last_price, 1.0)

    if last_price > 0:
        price_change = future_price_mean / last_price
    else:
        price_change = 1.0
    price_vs_mean = future_price_mean / hist_mean if hist_mean > 0 else 1.0

    return np.array([last_price, future_price_mean, price_change, price_vs_mean], dtype="float32")


def build_training_windows(
    sales_values: np.ndarray,
    sales_log: np.ndarray,
    cat_matrix: np.ndarray,
    calendar_num: pd.DataFrame,
    event_ids: pd.Series,
    price_matrix: Optional[np.ndarray],
    cfg: dict,
):
    """Create sliding window arrays for GRU direct-output training."""
    input_len = int(cfg["forecast"]["input_len"])
    horizon = int(cfg["forecast"]["horizon"])
    n_windows = int(cfg["forecast"]["n_windows"])
    step = int(cfg["forecast"]["step"])
    use_price = bool(cfg["features"].get("use_price_features", True)) and price_matrix is not None

    X_list, y_list, cat_list = [], [], []
    num_list, future_cal_list, future_event_list, end_pos_list = [], [], [], []
    price_list = []

    n_series, n_days = sales_values.shape
    for i in range(n_series):
        series_raw = sales_values[i]
        series_log = sales_log[i]
        cats = cat_matrix[i]
        price_series = price_matrix[i] if use_price else None

        for w in range(n_windows):
            target_end = n_days - w * step
            target_start = target_end - horizon
            input_start = target_start - input_len
            if input_start < 0:
                continue

            future_days = np.arange(target_start + 1, target_end + 1)

            X_list.append(series_log[input_start:target_start])
            y_list.append(series_log[target_start:target_end])
            cat_list.append(cats)
            num_list.append(make_lag_roll_activity_features(series_raw, target_start, cfg))
            future_cal_list.append(calendar_num.loc[future_days].values.astype("float32"))
            future_event_list.append(event_ids.loc[future_days].values.astype("int64"))
            price_list.append(make_price_features(price_series, target_start, target_start, target_end))
            end_pos_list.append(target_end)

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32")
    X_cat = np.array(cat_list, dtype="int64")
    X_num = np.array(num_list, dtype="float32")
    X_future_cal = np.array(future_cal_list, dtype="float32")
    X_future_event = np.array(future_event_list, dtype="int64")
    X_price = np.array(price_list, dtype="float32")
    end_pos = np.array(end_pos_list)

    print("Window shapes:", X.shape, y.shape, X_cat.shape, X_num.shape, X_future_cal.shape, X_future_event.shape, X_price.shape)
    return X, y, X_cat, X_num, X_future_cal, X_future_event, X_price, end_pos


def build_prediction_block(
    sales_values: np.ndarray,
    sales_log_input: np.ndarray,
    cat_matrix: np.ndarray,
    calendar_num: pd.DataFrame,
    event_ids: pd.Series,
    price_matrix: Optional[np.ndarray],
    cfg: dict,
    future_start_idx: int,
):
    """Create one prediction block for future_start_idx+1 through +horizon."""
    input_len = int(cfg["forecast"]["input_len"])
    horizon = int(cfg["forecast"]["horizon"])
    use_price = bool(cfg["features"].get("use_price_features", True)) and price_matrix is not None

    X = sales_log_input[:, -input_len:].astype("float32")
    dummy_y = np.zeros((len(X), horizon), dtype="float32")
    future_days = np.arange(future_start_idx + 1, future_start_idx + horizon + 1)

    X_num, X_future_cal, X_future_event, X_price = [], [], [], []
    for i in range(len(X)):
        series_raw = sales_values[i]
        price_series = price_matrix[i] if use_price else None
        X_num.append(make_lag_roll_activity_features(series_raw, future_start_idx, cfg))
        X_future_cal.append(calendar_num.loc[future_days].values.astype("float32"))
        X_future_event.append(event_ids.loc[future_days].values.astype("int64"))
        X_price.append(make_price_features(price_series, future_start_idx, future_start_idx, future_start_idx + horizon))

    return (
        X,
        dummy_y,
        cat_matrix.astype("int64"),
        np.array(X_num, dtype="float32"),
        np.array(X_future_cal, dtype="float32"),
        np.array(X_future_event, dtype="int64"),
        np.array(X_price, dtype="float32"),
    )
