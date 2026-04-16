"""
features.py
───────────
Builds all lag, rolling, price, calendar, and hierarchical features.
Works on both the training DataFrame and the recursive prediction DataFrame.
 
Usage:
    from src.features import build_features, get_feature_list
    df = build_features(df, cfg)
    features = get_feature_list(df, cfg)
"""
 
import numpy as np
import pandas as pd
 
 
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds all feature columns to df in-place (returns df for chaining).
 
    Parameters
    ----------
    df  : sorted long-format DataFrame (must have store_id, item_id, date, sales, sell_price)
    cfg : dict loaded from config.yaml
 
    Returns
    -------
    df with new feature columns added
    """
    feat_cfg = cfg["features"]
 
    df = df.sort_values(["store_id", "item_id", "date"])
 
    g  = df.groupby(["store_id", "item_id"], observed=True)["sales"]
    pg = df.groupby(["store_id", "item_id"], observed=True)["sell_price"]
 
    # ── Lag features ─────────────────────────────────────────
    for lag in feat_cfg["lags"]:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
 
    # ── Rolling mean features ────────────────────────────────
    for window in feat_cfg["rolling_means"]:
        df[f"rmean_{window}"] = (
            g.shift(1).rolling(window).mean().astype("float32")
        )
 
    # ── Rolling std (optional) ───────────────────────────────
    if feat_cfg.get("use_rolling_std", False):
        df["rolling_std_7"] = (
            g.shift(1).rolling(7).std().astype("float32")
        )
 
    # ── Price features ───────────────────────────────────────
    price_lag_1      = pg.shift(1).astype("float32")
    df["price_change"] = (df["sell_price"] / price_lag_1).astype("float32")
    df["price_change"] = df["price_change"].replace([np.inf, -np.inf], 1).fillna(1)
 
    df["price_mean_7"] = (
        pg.shift(1).rolling(7).mean().astype("float32")
    )
 
    # ── Calendar features ────────────────────────────────────
    df["dayofweek"]  = df["date"].dt.dayofweek.astype("int8")
    df["month"]      = df["date"].dt.month.astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
 
    # ── Hierarchical features ────────────────────────────────
    df["store_sales_mean_7"] = (
        df.groupby(["store_id", "date"], observed=True)["sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
        .astype("float32")
    )
 
    df["cat_sales_mean_7"] = (
        df.groupby(["cat_id", "date"], observed=True)["sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
        .astype("float32")
    )
 
    return df
 
 
def build_features_for_day(full_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Lightweight version used inside the recursive prediction loop.
    Recomputes only the features that depend on sales (lags, rolling means, price).
    Calendar features are static and only need to be set once.
 
    Parameters
    ----------
    full_df : concat of history + future rows (sales for future days filled in progressively)
    cfg     : dict loaded from config.yaml
 
    Returns
    -------
    full_df with updated feature columns
    """
    feat_cfg = cfg["features"]
 
    g  = full_df.groupby(["store_id", "item_id"], observed=True)
    pg = g["sell_price"]
 
    # Lag features
    for lag in feat_cfg["lags"]:
        full_df[f"lag_{lag}"] = g["sales"].shift(lag)
 
    # Rolling means
    for window in feat_cfg["rolling_means"]:
        full_df[f"rmean_{window}"] = g["sales"].shift(1).rolling(window).mean()
 
    # Price
    price_lag_1 = pg.shift(1)
    full_df["price_change"] = (full_df["sell_price"] / price_lag_1)
    full_df["price_change"] = full_df["price_change"].replace([np.inf, -np.inf], 1).fillna(1)
    full_df["price_mean_7"] = pg.shift(1).rolling(7).mean()
 
    # Calendar (safe to recompute, cheap)
    full_df["dayofweek"]  = full_df["date"].dt.dayofweek
    full_df["month"]      = full_df["date"].dt.month
    full_df["weekofyear"] = full_df["date"].dt.isocalendar().week.astype("int")
    full_df["is_weekend"] = (full_df["dayofweek"] >= 5).astype(int)
 
    return full_df
 
 
def get_feature_list(df: pd.DataFrame, cfg: dict) -> list:
    """
    Returns the final list of feature column names to pass to the model,
    filtered by what actually exists in df and what is listed in config.
 
    Parameters
    ----------
    df  : DataFrame after build_features()
    cfg : dict loaded from config.yaml
 
    Returns
    -------
    list of column names
    """
    requested = cfg["features"]["use"]
    available = set(df.columns) - {"sales", "date", "id"}
    features  = [f for f in requested if f in available]
 
    missing = [f for f in requested if f not in available]
    if missing:
        print(f"[features] WARNING: these features are in config but missing from df: {missing}")
 
    return features
