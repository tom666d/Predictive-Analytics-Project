"""
Memory-lean feature ablation helper for the M5 notebook baseline.
This version avoids the large pandas -> numpy float64 copy inside LightGBM
by encoding categoricals to integer codes and casting numeric features to float32
store by store.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb


BASELINE_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sell_price",
    "price_change",
    "dayofweek",
    "month",
    "weekofyear",
    "is_weekend",
    "snap_effective",
    "event_type_1",
    "lag_1",
    "lag_7",
    "lag_28",
    "rmean_3",
    "rmean_7",
    "rmean_28",
    "store_sales_mean_7",
    "cat_sales_mean_7",
]

CANDIDATE_GROUPS = {
    "intermittency": [
        "days_since_last_sale",
        "nonzero_count_7",
        "nonzero_count_28",
        "zero_ratio_28",
        "recently_active_7",
        "recently_active_28",
    ],
    "advanced_rolling": [
        "rmean_14",
        "rmean_56",
        "rmedian_7",
        "rmedian_28",
        "rmax_7",
        "rmax_28",
    ],
    "same_weekday": [
        "same_dow_mean_4w",
        "same_dow_mean_8w",
        "same_dow_median_8w",
    ],
    "price_depth": [
        "price_vs_mean",
        "price_vs_max",
        "discount_depth",
        "weeks_since_price_change",
        "is_price_drop",
    ],
    "hierarchy": [
        "store_dept_mean_7",
        "store_cat_mean_7",
        "state_cat_mean_7",
        "item_mean_across_stores_7",
    ],
    "event_window": [
        "is_event_minus_1",
        "is_event_minus_2",
        "is_event_plus_1",
    ],
}

NUMERIC_FILL_ZERO = {
    "price_change",
    "price_vs_mean",
    "price_vs_max",
    "discount_depth",
    "weeks_since_price_change",
    "lag_1",
    "lag_7",
    "lag_28",
    "rmean_3",
    "rmean_7",
    "rmean_14",
    "rmean_28",
    "rmean_56",
    "rmedian_7",
    "rmedian_28",
    "rmax_7",
    "rmax_28",
    "days_since_last_sale",
    "nonzero_count_7",
    "nonzero_count_28",
    "zero_ratio_28",
    "same_dow_mean_4w",
    "same_dow_mean_8w",
    "same_dow_median_8w",
    "store_sales_mean_7",
    "cat_sales_mean_7",
    "store_dept_mean_7",
    "store_cat_mean_7",
    "state_cat_mean_7",
    "item_mean_across_stores_7",
}


def _ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _build_store_cat_daily_feature(
    df: pd.DataFrame,
    group_cols: List[str],
    feature_name: str,
    window: int,
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols + [date_col], as_index=False, observed=False)[target_col]
        .sum()
        .sort_values(group_cols + [date_col])
    )
    agg[feature_name] = (
        agg.groupby(group_cols, observed=False)[target_col]
        .transform(lambda s: s.shift(1).rolling(window).mean())
    )
    return df.merge(
        agg[group_cols + [date_col, feature_name]],
        on=group_cols + [date_col],
        how="left",
    )


def _weeks_since_change(price_series: pd.Series) -> pd.Series:
    changed = price_series.ne(price_series.shift(1)).fillna(False)
    out = np.zeros(len(price_series), dtype="float32")
    counter = 0
    for i, is_changed in enumerate(changed):
        if i == 0 or is_changed:
            counter = 0
        else:
            counter += 1
        out[i] = counter
    return pd.Series(out, index=price_series.index)


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "snap_effective" not in df.columns:
        if "snap" in df.columns:
            df["snap_effective"] = df["snap"].fillna(0).astype("int8")
        elif all(col in df.columns for col in ["snap_CA", "snap_TX", "snap_WI"]):
            df["snap_effective"] = np.select(
                [
                    df["state_id"].astype(str) == "CA",
                    df["state_id"].astype(str) == "TX",
                    df["state_id"].astype(str) == "WI",
                ],
                [df["snap_CA"], df["snap_TX"], df["snap_WI"]],
                default=0,
            ).astype("int8")
        else:
            df["snap_effective"] = 0

    if "event_type_1" not in df.columns and "event_type" in df.columns:
        df["event_type_1"] = df["event_type"]

    df = _ensure_datetime(df).copy()
    df = df.sort_values(["store_id", "item_id", "date"], ignore_index=True)

    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1", "event_name_1"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month.astype("int8")
    if "weekofyear" not in df.columns:
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int16")
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    sales_g = df.groupby(["store_id", "item_id"], observed=False)["sales"]

    for lag in [1, 7, 28]:
        col = f"lag_{lag}"
        if col not in df.columns:
            df[col] = sales_g.shift(lag)

    for lag in [14, 21, 35, 42, 49, 56]:
        col = f"lag_{lag}"
        if col not in df.columns:
            df[col] = sales_g.shift(lag)

    for window in [3, 7, 28]:
        col = f"rmean_{window}"
        if col not in df.columns:
            df[col] = sales_g.shift(1).rolling(window).mean()

    for window in [14, 56]:
        df[f"rmean_{window}"] = sales_g.shift(1).rolling(window).mean()
    for window in [7, 28]:
        df[f"rmedian_{window}"] = sales_g.shift(1).rolling(window).median()
        df[f"rmax_{window}"] = sales_g.shift(1).rolling(window).max()

    sold_flag = (df["sales"] > 0).astype("float32")
    sold_g = sold_flag.groupby([df["store_id"], df["item_id"]], observed=False)

    df["nonzero_count_7"] = sold_g.shift(1).rolling(7).sum()
    df["nonzero_count_28"] = sold_g.shift(1).rolling(28).sum()
    df["zero_ratio_28"] = 1.0 - (df["nonzero_count_28"] / 28.0)
    df["recently_active_7"] = (df["nonzero_count_7"] > 0).astype("int8")
    df["recently_active_28"] = (df["nonzero_count_28"] > 0).astype("int8")

    last_sale_date = df["date"].where(df["sales"] > 0)
    df["last_sale_date"] = last_sale_date.groupby([df["store_id"], df["item_id"]], observed=False).ffill().shift(1)
    df["days_since_last_sale"] = (df["date"] - df["last_sale_date"]).dt.days.astype("float32")
    df.drop(columns=["last_sale_date"], inplace=True)

    lag_cols_4w = ["lag_7", "lag_14", "lag_21", "lag_28"]
    lag_cols_8w = ["lag_7", "lag_14", "lag_21", "lag_28", "lag_35", "lag_42", "lag_49", "lag_56"]
    df["same_dow_mean_4w"] = df[lag_cols_4w].mean(axis=1)
    df["same_dow_mean_8w"] = df[lag_cols_8w].mean(axis=1)
    df["same_dow_median_8w"] = df[lag_cols_8w].median(axis=1)

    price_g = df.groupby(["store_id", "item_id"], observed=False)["sell_price"]
    price_lag_1 = price_g.shift(1)

    if "price_change" not in df.columns:
        df["price_change"] = df["sell_price"] / price_lag_1.replace(0, np.nan)

    expanding_mean = price_g.transform(lambda s: s.shift(1).expanding().mean())
    expanding_max = price_g.transform(lambda s: s.shift(1).expanding().max())
    df["price_vs_mean"] = df["sell_price"] / expanding_mean.replace(0, np.nan)
    df["price_vs_max"] = df["sell_price"] / expanding_max.replace(0, np.nan)
    df["discount_depth"] = 1.0 - df["price_vs_max"]
    df["is_price_drop"] = (df["sell_price"] < price_lag_1).astype("int8")

    df["weeks_since_price_change"] = (
        df.groupby(["store_id", "item_id"], observed=False)["sell_price"]
        .transform(_weeks_since_change)
        / 7.0
    )

    if "store_sales_mean_7" not in df.columns:
        df = _build_store_cat_daily_feature(df, ["store_id"], "store_sales_mean_7", 7)
    if "cat_sales_mean_7" not in df.columns:
        df = _build_store_cat_daily_feature(df, ["cat_id"], "cat_sales_mean_7", 7)

    df = _build_store_cat_daily_feature(df, ["store_id", "dept_id"], "store_dept_mean_7", 7)
    df = _build_store_cat_daily_feature(df, ["store_id", "cat_id"], "store_cat_mean_7", 7)
    df = _build_store_cat_daily_feature(df, ["state_id", "cat_id"], "state_cat_mean_7", 7)
    df = _build_store_cat_daily_feature(df, ["item_id"], "item_mean_across_stores_7", 7)

    event_flag = (df["event_type_1"].astype(str).ne("nan") & df["event_type_1"].notna()).astype("int8")
    event_by_date = pd.DataFrame({"date": df["date"], "_event_flag": event_flag})
    event_by_date = (
        event_by_date.groupby("date", as_index=False, observed=False)["_event_flag"]
        .max()
        .sort_values("date")
    )
    event_lookup = event_by_date.set_index("date")["_event_flag"]

    df["is_event_minus_1"] = df["date"].add(pd.Timedelta(days=1)).map(event_lookup).fillna(0).astype("int8")
    df["is_event_minus_2"] = df["date"].add(pd.Timedelta(days=2)).map(event_lookup).fillna(0).astype("int8")
    df["is_event_plus_1"] = df["date"].sub(pd.Timedelta(days=1)).map(event_lookup).fillna(0).astype("int8")

    for col in df.columns:
        if col in NUMERIC_FILL_ZERO:
            df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train_hist: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    y_train_hist = np.asarray(y_train_hist, dtype="float64")

    if len(y_train_hist) < 2:
        return np.nan

    diffs = np.diff(y_train_hist)
    denom = np.mean(np.square(diffs))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    mse = np.mean(np.square(y_true - y_pred))
    if not np.isfinite(mse) or mse < 0:
        return np.nan

    score = np.sqrt(mse / denom)
    return float(score) if np.isfinite(score) else np.nan


def local_series_avg_rmsse(
    val_df: pd.DataFrame,
    pred_col: str,
    train_df: pd.DataFrame,
    id_cols=("store_id", "item_id"),
    target_col="sales",
) -> float:
    scores = []
    train_hist = (
        train_df.groupby(list(id_cols), observed=False)[target_col]
        .apply(lambda s: s.to_numpy())
        .to_dict()
    )

    for keys, grp in val_df.groupby(list(id_cols), observed=False):
        hist = train_hist.get(keys)
        if hist is None or len(hist) < 2:
            continue

        score = rmsse(
            grp[target_col].to_numpy(),
            grp[pred_col].to_numpy(),
            hist,
        )
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if scores else np.nan


def _make_lgb_arrays(df: pd.DataFrame, features: List[str]):
    """Convert features to one compact float32 matrix.
    Category columns become integer codes; numeric columns become float32.
    """
    parts = []
    for col in features:
        s = df[col]
        if str(s.dtype) == "category":
            arr = s.cat.codes.to_numpy(dtype=np.int32, copy=False).astype(np.float32, copy=False)
        elif pd.api.types.is_integer_dtype(s):
            arr = s.to_numpy(dtype=np.int32, copy=False).astype(np.float32, copy=False)
        elif pd.api.types.is_bool_dtype(s):
            arr = s.to_numpy(dtype=np.int8, copy=False).astype(np.float32, copy=False)
        else:
            arr = s.to_numpy(dtype=np.float32, copy=False)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        parts.append(arr)

    if not parts:
        return np.empty((len(df), 0), dtype=np.float32)

    return np.hstack(parts).astype(np.float32, copy=False)


def train_store_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    target_col: str = "sales",
    store_col: str = "store_id",
) -> pd.DataFrame:
    valid_features = [c for c in features if c in train_df.columns and c in val_df.columns]
    preds = []

    grouped_train = train_df.groupby(store_col, observed=False, sort=False)
    grouped_val = {k: g for k, g in val_df.groupby(store_col, observed=False, sort=False)}

    for store, tr in grouped_train:
        va = grouped_val.get(store)
        if va is None or tr.empty or va.empty:
            continue

        X_tr = _make_lgb_arrays(tr, valid_features)
        y_tr = tr[target_col].to_numpy(dtype=np.float32, copy=False)
        X_va = _make_lgb_arrays(va, valid_features)
        y_va = va[target_col].to_numpy(dtype=np.float32, copy=False)

        model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.1,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        out = va.copy()
        out["pred"] = model.predict(X_va).clip(0)
        preds.append(out)

        del X_tr, y_tr, X_va, y_va, model

    if not preds:
        return val_df.iloc[0:0].copy()

    return pd.concat(preds, axis=0, ignore_index=True)


def run_feature_ablation(
    df: pd.DataFrame,
    base_features: List[str],
    candidate_groups: Dict[str, List[str]],
    target_col: str = "sales",
    date_col: str = "date",
    store_col: str = "store_id",
    validation_days: int = 28,
) -> pd.DataFrame:
    df = _ensure_datetime(df, date_col).sort_values([store_col, "item_id", date_col], ignore_index=True)

    cutoff = df[date_col].max() - pd.Timedelta(days=validation_days)
    train_df = df[df[date_col] <= cutoff].copy()
    val_df = df[df[date_col] > cutoff].copy()

    existing_base = [c for c in base_features if c in df.columns]
    results = []

    baseline_pred = train_store_models(
        train_df,
        val_df,
        existing_base,
        target_col=target_col,
        store_col=store_col,
    )
    baseline_score = local_series_avg_rmsse(
        val_df=baseline_pred,
        pred_col="pred",
        train_df=train_df,
        id_cols=("store_id", "item_id"),
        target_col=target_col,
    )
    results.append(
        {
            "group_name": "baseline_only",
            "n_features": len(existing_base),
            "local_avg_rmsse": baseline_score,
            "delta_vs_baseline": 0.0,
            "features_added": ", ".join(existing_base),
        }
    )

    for group_name, feature_list in candidate_groups.items():
        added_features = [c for c in feature_list if c in df.columns]
        test_features = existing_base + added_features

        pred_df = train_store_models(
            train_df,
            val_df,
            test_features,
            target_col=target_col,
            store_col=store_col,
        )
        score = local_series_avg_rmsse(
            val_df=pred_df,
            pred_col="pred",
            train_df=train_df,
            id_cols=("store_id", "item_id"),
            target_col=target_col,
        )

        delta = score - baseline_score if np.isfinite(score) and np.isfinite(baseline_score) else np.nan

        results.append(
            {
                "group_name": group_name,
                "n_features": len(test_features),
                "local_avg_rmsse": score,
                "delta_vs_baseline": delta,
                "features_added": ", ".join(added_features),
            }
        )

    out = pd.DataFrame(results).sort_values(
        ["local_avg_rmsse", "group_name"],
        na_position="last",
    ).reset_index(drop=True)
    return out
