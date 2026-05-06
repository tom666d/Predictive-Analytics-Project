import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

from preprocessing import load_and_preprocess
from models import train_store_models, predict_store


# ============================================================
# Direct-safe non-recursive LightGBM experiment
# ============================================================
# RUN_MODE:
#   "smoke" = quick test, only CA_1, fewer days, fewer trees
#   "full"  = full experiment, all stores, 500 days, stronger model
# ============================================================

RUN_MODE = "full"  


def get_cfg():
    if RUN_MODE == "smoke":
        n_days = 160
        lags = [28, 35, 42, 56, 84]
        rolling_means = [7, 28]
        n_estimators = 30
        stores_filter = ["CA_1"]
        submission_path = "submission_direct_clean_smoke_v5.csv"

    elif RUN_MODE == "medium":
        n_days = 500
        lags = [28, 35, 42, 56,]
        rolling_means = [7, 28, 56]
        n_estimators = 200
        stores_filter = ["CA_1"]
        submission_path = "submission_direct_clean_medium.csv"

    else:
        n_days = 500
        lags = [28, 35, 42, 56, 84, 364, 365, 392]
        rolling_means = [7, 28, 56]
        n_estimators = 500
        stores_filter = None
        submission_path = "submission_direct_clean_full.csv"

    cfg = {
        "data": {
            "path": "data",
            "n_days": n_days,
            "val_days": 28,
        },
        "features": {
            "lags": lags,
            "rolling_means": rolling_means,
            "rolling_shift": 28,
        },
        "model": {
            "active_model": "lgbm",
            "lgbm": {
                "objective": "regression",
                "tweedie_variance_power": 1.1235258496200073,
                "n_estimators": n_estimators,
                "learning_rate": 0.012849210995834437,
                "num_leaves": 158,
                "subsample": 0.7093306709222214,
                "colsample_bytree": 0.9224038879058452,
                "early_stopping_rounds": 50,
                "log_every": 100,
            },
        },
        "output": {
            "submission_path": submission_path,
            "models_dir": "models_direct_clean",
        },
        "stores_filter": stores_filter,
    }
    return cfg


def time_split(df: pd.DataFrame, val_days: int):
    cutoff = df["date"].max() - pd.Timedelta(days=val_days)
    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()

    print(
        f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} "
        f"({len(train_df):,} rows)"
    )
    print(
        f"Val  : {val_df['date'].min().date()} → {val_df['date'].max().date()} "
        f"({len(val_df):,} rows)"
    )
    return train_df, val_df


def build_direct_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Build direct-safe features.

    Key idea:
    - Future prediction will use target-date rows.
    - All sales-dependent features must use lags >= 28 or rolling_shift = 28.
    - This avoids recursive prediction and avoids using unknown future sales.
    """
    feat_cfg = cfg["features"]
    lags = feat_cfg["lags"]
    rolling_means = feat_cfg["rolling_means"]
    rolling_shift = feat_cfg.get("rolling_shift", 28)

    df = df.sort_values(["store_id", "item_id", "date"]).copy()

    g_sales = df.groupby(["store_id", "item_id"], observed=True)["sales"]
    g_price = df.groupby(["store_id", "item_id"], observed=True)["sell_price"]

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = g_sales.shift(lag).astype("float32")

    # Direct-safe rolling means: shift by 28
    for window in rolling_means:
        df[f"rmean_{window}"] = (
            df.groupby(["store_id", "item_id"], observed=True)["sales"]
            .transform(lambda x: x.shift(rolling_shift).rolling(window).mean())
            .astype("float32")
        )

    # Weekday-specific features
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    df["weekday_rmean_28"] = (
        df.groupby(["store_id", "item_id", "dayofweek"], observed=True)["sales"]
        .transform(lambda x: x.shift(4).rolling(4).mean())
        .astype("float32")
    )

    df["weekday_rmean_56"] = (
        df.groupby(["store_id", "item_id", "dayofweek"], observed=True)["sales"]
        .transform(lambda x: x.shift(4).rolling(8).mean())
        .astype("float32")
    )
        # Fill SNAP missing values
    if "snap" in df.columns:
        df["snap"] = df["snap"].fillna(0).astype("int8")

    # Price features
    price_lag_1 = g_price.shift(1).astype("float32")
    df["price_change"] = (df["sell_price"] / price_lag_1).astype("float32")
    df["price_change"] = df["price_change"].replace([np.inf, -np.inf], 1).fillna(1)

    df["price_mean_7"] = (
        df.groupby(["store_id", "item_id"], observed=True)["sell_price"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
        .astype("float32")
    )

    price_hist_mean = g_price.transform("mean").astype("float32")
    df["price_vs_mean"] = (df["sell_price"] / price_hist_mean).astype("float32")
    df["price_vs_mean"] = df["price_vs_mean"].replace([np.inf, -np.inf], 1).fillna(1)

    # High impact event
    if "event_name_1" in df.columns:
        high_impact = ["LaborDay", "SuperBowl", "Easter"]
        df["is_high_impact_event"] = df["event_name_1"].isin(high_impact).astype("int8")
    else:
        df["is_high_impact_event"] = 0

    # Days since first sale
    first_sale = (
        df[df["sales"] > 0]
        .groupby(["store_id", "item_id"], observed=True)["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "first_sale_date"})
    )
    df = df.merge(first_sale, on=["store_id", "item_id"], how="left")
    df["days_since_first_sale"] = (df["date"] - df["first_sale_date"]).dt.days
    df["days_since_first_sale"] = (
        df["days_since_first_sale"].fillna(0).clip(lower=0).astype("float32")
    )
    df.drop(columns=["first_sale_date"], inplace=True)

    return df


def get_feature_list(df: pd.DataFrame, cfg: dict):
    lag_features = [f"lag_{lag}" for lag in cfg["features"]["lags"]]
    rolling_features = [f"rmean_{w}" for w in cfg["features"]["rolling_means"]]

    requested = (
        lag_features
        + rolling_features
        + [
            "sell_price",
            "price_change",
            "price_mean_7",
            "price_vs_mean",
            "dayofweek",
            "wday",
            "weekofyear",
            "month",
            "is_weekend",
            "snap",
            "event_type",
            "is_high_impact_event",
            "weekday_rmean_28",
            "weekday_rmean_56",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "days_since_first_sale",
        ]
    )

    available = set(df.columns) - {"sales", "date", "id"}
    features = [f for f in requested if f in available]
    missing = [f for f in requested if f not in available]

    if missing:
        print(f"[features] WARNING missing features: {missing}")

    return features


def compute_wrmsse(val_preds, val_actuals, train_df, sell_prices, calendar):
    items = (
        train_df[["store_id", "item_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    items["idx"] = items.index

    train_pivot = (
        train_df
        .merge(items, on=["store_id", "item_id"])
        .pivot_table(index="idx", columns="date", values="sales", aggfunc="first")
        .sort_index()
    )

    sales_matrix = train_pivot.values.astype(float)
    diffs = np.diff(sales_matrix, axis=1)
    scales = np.sqrt(np.nanmean(diffs ** 2, axis=1))
    scales = np.where(scales == 0, 1.0, scales)

    mse = np.mean((val_preds - val_actuals) ** 2, axis=1)
    rmsse = np.sqrt(mse) / scales

    val_wks = calendar.loc[
        calendar["date"] >= train_df["date"].max() + pd.Timedelta(days=1),
        "wm_yr_wk"
    ].unique()

    price_avg = (
        sell_prices[sell_prices["wm_yr_wk"].isin(val_wks)]
        .groupby(["store_id", "item_id"])["sell_price"]
        .mean()
        .reset_index()
        .rename(columns={"sell_price": "avg_price"})
    )

    items_w = items.merge(price_avg, on=["store_id", "item_id"], how="left")
    items_w["avg_sales"] = val_actuals.mean(axis=1)
    items_w["revenue"] = items_w["avg_price"] * items_w["avg_sales"]
    items_w["revenue"] = items_w["revenue"].fillna(0)

    weights = items_w["revenue"].values
    total = weights.sum()
    weights = weights / total if total > 0 else weights

    return float(np.sum(weights * rmsse))


def build_future_frame(history_df, calendar, sell_prices, future_dates):
    base = history_df[
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ].drop_duplicates()

    future_df = base.merge(pd.DataFrame({"date": future_dates}), how="cross")
    future_df = future_df.merge(calendar, on="date", how="left")
    future_df = future_df.merge(
        sell_prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    future_df["sales"] = np.nan

    # Recreate snap column for future rows
    if {"snap_CA", "snap_TX", "snap_WI"}.issubset(future_df.columns):
        future_df["snap"] = 0
        future_df.loc[future_df["state_id"] == "CA", "snap"] = future_df["snap_CA"]
        future_df.loc[future_df["state_id"] == "TX", "snap"] = future_df["snap_TX"]
        future_df.loc[future_df["state_id"] == "WI", "snap"] = future_df["snap_WI"]

    # Make SNAP explicitly 0/1 instead of missing
    if "snap" in future_df.columns:
        future_df["snap"] = future_df["snap"].fillna(0).astype("int8")
        # Recreate event_type for future rows
        if "event_type_1" in future_df.columns:
            future_df["event_type"] = future_df["event_type_1"].fillna("None")

        return future_df


def align_categories(full_df, reference_df, cat_cols):
    for col in cat_cols:
        if col in full_df.columns and col in reference_df.columns:
            full_df[col] = full_df[col].astype("category")
            if hasattr(reference_df[col], "cat"):
                full_df[col] = full_df[col].cat.set_categories(reference_df[col].cat.categories)
    return full_df


def direct_predict(history_df, calendar, sell_prices, features, cat_cols, models, cfg):
    last_date = history_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=28)

    max_lag = max(cfg["features"]["lags"])
    max_roll = max(cfg["features"]["rolling_means"]) + cfg["features"].get("rolling_shift", 28)
    history_days = max(max_lag, max_roll) + 5

    hist_cutoff = last_date - pd.Timedelta(days=history_days)
    hist_df = history_df[history_df["date"] > hist_cutoff].copy()

    future_df = build_future_frame(hist_df, calendar, sell_prices, future_dates)

    full_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)
    full_df = align_categories(full_df, history_df, cat_cols)

    full_df = build_direct_features(full_df, cfg)

    preds = []
    for i, day in enumerate(future_dates, 1):
        print(f"  Predicting F{i}: {day.date()}")
        X_day = full_df.loc[full_df["date"] == day, features]
        y_pred = predict_store(X_day, models)
        y_pred = np.maximum(y_pred, 0)
        preds.append(y_pred)

    return np.stack(preds, axis=1)


def build_submission(preds, df, sample_sub_path):
    base_ids = df["id"].drop_duplicates().str.replace("_validation", "", regex=False)

    preds_df = pd.DataFrame(preds, columns=[f"F{i}" for i in range(1, 29)])
    preds_df["id"] = base_ids.values

    preds_val = preds_df.copy()
    preds_val["id"] = preds_val["id"] + "_validation"

    preds_eval = preds_df.copy()
    preds_eval["id"] = preds_eval["id"] + "_evaluation"

    submission = pd.concat([preds_val, preds_eval], axis=0)

    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        submission = (
            submission.set_index("id")
            .loc[sample_sub["id"]]
            .reset_index()
        )

    return submission


def main():
    cfg = get_cfg()

    print(f"RUN_MODE = {RUN_MODE}")
    print("\n── Step 1: Load and preprocess ──")
    df, calendar, sell_prices = load_and_preprocess(cfg)

    # Optional smoke filter
    stores_filter = cfg.get("stores_filter")
    if stores_filter is not None:
        df = df[df["store_id"].isin(stores_filter)].copy()
        sell_prices = sell_prices[sell_prices["store_id"].isin(stores_filter)].copy()
        print(f"Smoke mode stores: {stores_filter}")
        print(f"Filtered df shape: {df.shape}")

    print("\n── Step 2: Build direct-safe features ──")
    df = build_direct_features(df, cfg)

    print("\n── Step 3: Train/validation split ──")
    train_df, val_df = time_split(df, cfg["data"]["val_days"])

    features = get_feature_list(df, cfg)
    cat_cols = [c for c in features if df[c].dtype.name == "category"]

    print(f"Features ({len(features)}): {features}")
    print(f"Categorical: {cat_cols}")

    # Inner split for early stopping
    print("\n── Step 4: Train validation model ──")
    train_inner, val_inner = time_split(train_df, cfg["data"]["val_days"])

    val_models = train_store_models(
        train_inner,
        val_inner,
        features,
        target="sales",
        cat_cols=cat_cols,
        cfg=cfg,
        model_name="lgbm",
    )

    print("\n── Step 5: Direct validation prediction ──")
    val_preds = direct_predict(
        train_df,
        calendar,
        sell_prices,
        features,
        cat_cols,
        val_models,
        cfg,
    )

    items = train_df[["store_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    items["idx"] = items.index

    val_pivot = (
        val_df
        .merge(items, on=["store_id", "item_id"])
        .pivot_table(index="idx", columns="date", values="sales", aggfunc="first")
        .sort_index()
    )
    val_actuals = val_pivot.values.astype(float)

    print(f"Validation preds shape: {val_preds.shape}")
    print(f"Validation actuals shape: {val_actuals.shape}")

    wrmsse = compute_wrmsse(val_preds, val_actuals, train_df, sell_prices, calendar)
    print(f"\nValidation WRMSSE: {wrmsse:.4f}")

    if RUN_MODE == "smoke":
        print("\nSmoke test finished. If this works, change RUN_MODE = 'full'.")
        return

    print("\n── Step 6: Train final model on full data ──")
    final_models = train_store_models(
        df,
        None,
        features,
        target="sales",
        cat_cols=cat_cols,
        cfg=cfg,
        model_name="lgbm",
    )

    print("\n── Step 7: Direct future prediction ──")
    preds = direct_predict(
        df,
        calendar,
        sell_prices,
        features,
        cat_cols,
        final_models,
        cfg,
    )

    print(f"Final preds shape: {preds.shape}")

    print("\n── Step 8: Build submission ──")
    sample_path = os.path.join(cfg["data"]["path"], "sample_submission.csv")
    submission = build_submission(preds, df, sample_path)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_path = cfg["output"]["submission_path"].replace(".csv", f"_{timestamp}.csv")
    submission.to_csv(out_path, index=False)

    print(f"Submission saved to: {out_path}")
    print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()