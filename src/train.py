"""
train.py
────────
Main entry point for the M5 pipeline.

Steps:
  1.  Load config
  2.  Preprocess raw data
  3.  Build features
  4.  Time-based train/val split
  5.  (Optional) Hyperparameter tuning via Optuna
  6.  Train model(s) with validation + early stopping
  7.  Retrain on full data with best n_estimators
  8.  Recursive 28-day prediction
  9.  Validation WRMSSE
  10. Feature Importance
  11. Generate submission CSV

Run:
    python src/train.py
    python src/train.py --config configs/config.yaml

Tuning:
    Set tuning.enabled: true in config.yaml to run Optuna hyperparameter search.
    Best params are automatically applied before final training.
"""
 
import os
import argparse
import numpy as np
import pandas as pd
import yaml
import joblib
 
from preprocessing import load_and_preprocess
from features      import build_features, build_features_for_day, get_feature_list
from models        import (
    train_store_models,
    train_ensemble_models,
    predict_store,
    predict_store_ensemble,
)
 
 
# ── Helpers ──────────────────────────────────────────────────
 
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def save_models(models: dict, models_dir: str, model_name: str = "lgbm"):
    """
    Save trained store models to disk.
 
    Parameters
    ----------
    models     : {store_id: trained_model}  or  {model_name: {store_id: trained_model}}
    models_dir : folder to save into (created if it doesn't exist)
    model_name : used as filename prefix
    """
    os.makedirs(models_dir, exist_ok=True)
 
    # Handle ensemble dict: {model_name: {store: model}}
    if isinstance(next(iter(models.values())), dict):
        for name, store_models in models.items():
            for store, model in store_models.items():
                path = os.path.join(models_dir, f"{name}_{store}.joblib")
                joblib.dump(model, path)
        print(f"Ensemble models saved to: {models_dir}")
    else:
        for store, model in models.items():
            path = os.path.join(models_dir, f"{model_name}_{store}.joblib")
            joblib.dump(model, path)
        print(f"Models saved to: {models_dir}")
 
 
def load_models(models_dir: str, model_name: str = "lgbm") -> dict:
    """
    Load saved store models from disk.
 
    Parameters
    ----------
    models_dir : folder containing saved .joblib files
    model_name : prefix used when saving (e.g. "lgbm", "xgb")
 
    Returns
    -------
    {store_id: trained_model}
    """
    models = {}
    for fname in os.listdir(models_dir):
        if fname.startswith(model_name) and fname.endswith(".joblib"):
            store = fname.replace(f"{model_name}_", "").replace(".joblib", "")
            models[store] = joblib.load(os.path.join(models_dir, fname))
    print(f"Loaded {len(models)} models from: {models_dir}")
    return models
 
 
def time_split(df: pd.DataFrame, val_days: int):
    cutoff   = df["date"].max() - pd.Timedelta(days=val_days)
    train_df = df[df["date"] <= cutoff].copy()
    val_df   = df[df["date"] > cutoff].copy()
    print(f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} "
          f"({len(train_df):,} rows)")
    print(f"Val  : {val_df['date'].min().date()} → {val_df['date'].max().date()} "
          f"({len(val_df):,} rows)")
    return train_df, val_df
 
def compute_wrmsse(val_preds, val_actuals, train_df, sell_prices, calendar):
    """
    Compute WRMSSE on validation set (vectorized).

    Parameters
    ----------
    val_preds   : np.ndarray (n_items, 28)
    val_actuals : np.ndarray (n_items, 28)
    train_df    : training DataFrame (before val cutoff)
    sell_prices : raw sell_prices DataFrame
    calendar    : raw calendar DataFrame

    Returns
    -------
    float : WRMSSE score
    """
    # ── Item index ───────────────────────────────────────────
    items = (
        train_df[["store_id", "item_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    items["idx"] = items.index

    # ── Scale: vectorized first-order diff RMSE ──────────────
    train_pivot = (
        train_df
        .merge(items, on=["store_id", "item_id"])
        .pivot_table(index="idx", columns="date", values="sales", aggfunc="first")
        .sort_index()
    )
    sales_matrix = train_pivot.values.astype(float)          # (n_items, n_days)
    diffs        = np.diff(sales_matrix, axis=1)              # (n_items, n_days-1)
    scales       = np.sqrt(np.nanmean(diffs ** 2, axis=1))   # (n_items,)
    scales       = np.where(scales == 0, 1.0, scales)

    # ── RMSSE per item ────────────────────────────────────────
    mse   = np.mean((val_preds - val_actuals) ** 2, axis=1)  # (n_items,)
    rmsse = np.sqrt(mse) / scales                             # (n_items,)

    # ── Weights ───────────────────────────────────────────────
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
    items_w["revenue"]   = items_w["avg_price"] * items_w["avg_sales"]
    items_w["revenue"]   = items_w["revenue"].fillna(0)

    weights = items_w["revenue"].values
    total   = weights.sum()
    weights = weights / total if total > 0 else weights

    return float(np.sum(weights * rmsse))

def build_submission(preds: np.ndarray, df: pd.DataFrame, sample_sub_path: str) -> pd.DataFrame:
    """Formats preds (30490 × 28) into a Kaggle submission DataFrame."""
    base_ids = df["id"].drop_duplicates().str.replace("_validation", "", regex=False)
 
    preds_df      = pd.DataFrame(preds, columns=[f"F{i}" for i in range(1, 29)])
    preds_df["id"] = base_ids.values
 
    preds_val        = preds_df.copy()
    preds_val["id"]  = preds_val["id"] + "_validation"
 
    preds_eval       = preds_df.copy()
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
 
 
# ── Recursive prediction ──────────────────────────────────────
 
def recursive_predict(df, calendar, sell_prices, features, cat_cols, models_or_all, cfg):
    active = cfg["model"]["active_model"]
    is_ensemble = (active == "ensemble")
    
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=28)
    
    all_preds = []
    stores = df["store_id"].unique()
    
    for store in stores:
        print(f"  Predicting store: {store}")
        store_df = df[df["store_id"] == store].copy()
        
        # Build future frame for this store only
        base = store_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
        future_df = base.merge(pd.DataFrame({"date": future_dates}), how="cross")
        future_df = future_df.merge(calendar, on="date", how="left")
        future_df["snap"] = 0
        future_df.loc[future_df["state_id"] == "CA", "snap"] = future_df["snap_CA"]
        future_df.loc[future_df["state_id"] == "TX", "snap"] = future_df["snap_TX"]
        future_df.loc[future_df["state_id"] == "WI", "snap"] = future_df["snap_WI"]
        future_df = future_df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        
        full_df = pd.concat([store_df, future_df], ignore_index=True)
        full_df = full_df.sort_values(["item_id", "date"])
        full_df["sales"] = full_df["sales"].astype("float64")
        
        for col in cat_cols:
            if col in full_df.columns:
                full_df[col] = full_df[col].astype("category")
                full_df[col] = full_df[col].cat.set_categories(df[col].cat.categories)
        
        store_preds = []
        for i, day in enumerate(future_dates, 1):
            full_df = build_features_for_day(full_df, cfg)
            mask = full_df["date"] == day
            X = full_df.loc[mask, features]
            
            if is_ensemble:
                y_pred = predict_store_ensemble(X, models_or_all, cfg)
            else:
                store_models = {store: models_or_all[store]}
                y_pred = predict_store(X, store_models)

            y_pred = np.clip(y_pred, 0, None)
            full_df.loc[mask, "sales"] = y_pred
            store_preds.append(y_pred)
        
        all_preds.append(np.stack(store_preds, axis=1))
    
    print("\nRecursive prediction done.")
    return np.vstack(all_preds)  # shape: (30490, 28)
 
def tune_hyperparams(train_df, val_df, features, cat_cols, target, cfg, calendar, sell_prices, n_trials=20):
    """
    Use Optuna to tune LightGBM hyperparameters based on validation WRMSSE.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # ── Search space ─────────────────────────────────────
        cfg["model"]["lgbm"]["num_leaves"]              = trial.suggest_int("num_leaves", 64, 512)
        cfg["model"]["lgbm"]["learning_rate"]           = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        cfg["model"]["lgbm"]["subsample"]               = trial.suggest_float("subsample", 0.5, 1.0)
        cfg["model"]["lgbm"]["colsample_bytree"]        = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        cfg["model"]["lgbm"]["tweedie_variance_power"]  = trial.suggest_float("tweedie_variance_power", 1.0, 1.5)

        # Train on train_df
        models = train_store_models(train_df, val_df, features, target, cat_cols, cfg)

        items = train_df[["store_id", "item_id"]].drop_duplicates().reset_index(drop=True)
        items["idx"] = items.index
        # Recursive predict on val period
        val_preds_matrix = np.zeros((len(items), 28))
        for store_id, store_items in items.groupby("store_id"):
            store_mask = items["store_id"] == store_id
            idxs = items.loc[store_mask, "idx"].values
            val_store = val_df[val_df["store_id"] == store_id].sort_values(["item_id", "date"])
            X_val = val_store[features]
            preds = models[store_id].predict(X_val)
            # reshape to (n_items_in_store, 28)
            n_items_store = store_mask.sum()
            val_preds_matrix[idxs] = preds.reshape(n_items_store, 28)

        # Build actuals matrix
        val_pivot = (
            val_df.merge(items, on=["store_id", "item_id"])
            .pivot_table(index="idx", columns="date", values="sales", aggfunc="first")
            .sort_index()
        )
        val_actuals = val_pivot.values.astype(float)

        wrmsse = compute_wrmsse(val_preds_matrix, val_actuals, train_df, sell_prices, calendar)
        print(f"  Trial {trial.number}: WRMSSE={wrmsse:.4f} | params={trial.params}")
        return wrmsse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest WRMSSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params
# ── Main ─────────────────────────────────────────────────────
 
def main(config_path: str = "configs/config.yaml"):
    print("main() started")
    cfg = load_config(config_path)
    active = cfg["model"]["active_model"]
 
    # ── 1. Preprocess ────────────────────────────────────────
    print("\n── Step 1: Preprocessing ──")
    df, calendar, sell_prices = load_and_preprocess(cfg)
 
    # ── 2. Build features ────────────────────────────────────
    print("\n── Step 2: Feature engineering ──")
    df = build_features(df, cfg)
 
    # ── 3. Train/val split ───────────────────────────────────
    print("\n── Step 3: Train/val split ──")
    train_df, val_df = time_split(df, cfg["data"]["val_days"])
 
    # ── 4. Get feature & cat lists ───────────────────────────
    features = get_feature_list(df, cfg)
    cat_cols = [c for c in features if df[c].dtype.name == "category"]
    target   = "sales"
    print(f"Features ({len(features)}): {features}")
    print(f"Categorical: {cat_cols}")

    # ── Optional: Hyperparameter Tuning ──────────────────────────
    if cfg.get("tuning", {}).get("enabled", False):
        print(f"\n── Hyperparameter Tuning ──")
        n_trials = cfg.get("tuning", {}).get("n_trials", 20)
        best_params = tune_hyperparams(
            train_df, val_df, features, cat_cols, target, cfg,
            calendar, sell_prices, n_trials=n_trials
        )
        cfg["model"]["lgbm"].update(best_params)
        print("Config updated with best params. Retraining with best params...")
    
    # ── 5. Train with validation (early stopping) ────────────
    skip = cfg["output"].get("skip_training", False)
 
    if skip:
        print(f"\n── Step 4: Skipping training — loading saved models ──")
        models_dir = cfg["output"]["models_dir"]
        final_models = load_models(models_dir, model_name=active)
    else:
        print(f"\n── Step 4: Training ({active}) with validation ──")
        if active == "ensemble":
            val_models = train_ensemble_models(train_df, val_df, features, target, cat_cols, cfg)
        else:
            val_models = train_store_models(train_df, val_df, features, target, cat_cols, cfg)
        

        max_best_iter = max(m.best_iteration_ for m in val_models.values())
        print(f"Max best iteration: {max_best_iter} → retraining with {int(max_best_iter * 1.1)} trees")
        cfg["model"]["lgbm"]["n_estimators"] = int(max_best_iter * 1.1)

        # ── 6. Retrain on full data ──────────────────────────────
        print(f"\n── Step 5: Retraining on full data ──")
        if active == "ensemble":
            final_models = train_ensemble_models(df, None, features, target, cat_cols, cfg)
        else:
            final_models = train_store_models(df, None, features, target, cat_cols, cfg)
 
        # ── Save models ──────────────────────────────────────────
        models_dir = cfg["output"]["models_dir"]
        save_models(final_models, models_dir, model_name=active)
 
    # ── 7. Recursive prediction ──────────────────────────────
    print(f"\n── Step 6: Recursive 28-day prediction ──")
    preds = recursive_predict(df, calendar, sell_prices, features, cat_cols, final_models, cfg)
    print(f"Predictions shape: {preds.shape}")

    # ── 8. Validation WRMSSE ─────────────────────────────────────
    print(f"\n── Step 7: Validation WRMSSE ──")

    if not skip:
        # Predict on validation period using val_models
        val_preds_matrix = recursive_predict(
            train_df, calendar, sell_prices, features, cat_cols, val_models, cfg
        )

        # Build actuals matrix (n_items x 28) — vectorized
        items = train_df[["store_id", "item_id"]].drop_duplicates().reset_index(drop=True)
        items["idx"] = items.index

        val_pivot = (
            val_df
            .merge(items, on=["store_id", "item_id"])
            .pivot_table(index="idx", columns="date", values="sales", aggfunc="first")
            .sort_index()
        )
        val_actuals = val_pivot.values.astype(float)  # (n_items, 28)

        wrmsse = compute_wrmsse(val_preds_matrix, val_actuals, train_df, sell_prices, calendar)
        print(f"  Validation WRMSSE: {wrmsse:.4f}")
    else:
        print("  Skipped (skip_training=True)")


    # ── 9. Feature Importance ────────────────────────────────
    print(f"\n── Step 8: Feature Importance ──")

    # take the average of feature importances across all store models
    importance_list = []
    for store, model in final_models.items():
        importance_list.append(model.feature_importances_)

    avg_importance = np.mean(importance_list, axis=0)

    importance = pd.DataFrame({
        "feature": features,
        "importance": avg_importance
    }).sort_values("importance", ascending=False)

    print(importance.to_string(index=False))
    importance.to_csv("feature_importance.csv", index=False)
    print("Feature importance saved to: feature_importance.csv")

    # ── 9. Submission ────────────────────────────────────────
    print(f"\n── Step 9: Building submission ──")
    sample_path = os.path.join(cfg["data"]["path"], "sample_submission.csv")
    submission  = build_submission(preds, df, sample_path)
 
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_path = cfg["output"]["submission_path"].replace(".csv", f"_{timestamp}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}  shape: {submission.shape}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)