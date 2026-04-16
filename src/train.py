"""
train.py
────────
Main entry point for the M5 pipeline.
 
Steps:
  1. Load config
  2. Preprocess raw data
  3. Build features
  4. Time-based train/val split
  5. Train model(s) with validation + early stopping
  6. Retrain on full data with best n_estimators
  7. Recursive 28-day prediction
  8. Generate submission CSV
 
Run:
    python src/train.py
    python src/train.py --config configs/config.yaml
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
    """
    Parameters
    ----------
    models_or_all : either {store: model}  (single model)
                    or     {name: {store: model}}  (ensemble)
    """
    active   = cfg["model"]["active_model"]
    is_ensemble = (active == "ensemble")
 
    last_date    = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=28)
 
    # Build future frame
    base      = df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
    future_df = base.merge(pd.DataFrame({"date": future_dates}), how="cross")
    future_df = future_df.merge(calendar, on="date", how="left")
    future_df = future_df.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
 
    # Combine history + future
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = full_df.sort_values(["store_id", "item_id", "date"])
    full_df["sales"] = full_df["sales"].astype("float64")
 
    # Align categorical columns
    for col in cat_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype("category")
            full_df[col] = full_df[col].cat.set_categories(df[col].cat.categories)
 
    preds = []
 
    print("Running recursive prediction...")
    for i, day in enumerate(future_dates, 1):
        print(f"  Day {i}/28: {day.date()}", end="\r")
 
        full_df = build_features_for_day(full_df, cfg)
 
        mask = full_df["date"] == day
        X    = full_df.loc[mask, features]
 
        if is_ensemble:
            y_pred = predict_store_ensemble(X, models_or_all, cfg)
        else:
            y_pred = predict_store(X, models_or_all)
 
        full_df.loc[mask, "sales"] = y_pred
        preds.append(y_pred)
 
    print("\nRecursive prediction done.")
    return np.stack(preds, axis=1)   # shape: (30490, 28)
 
 
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
 
    # ── 8. Submission ────────────────────────────────────────
    print(f"\n── Step 7: Building submission ──")
    sample_path = os.path.join(cfg["data"]["path"], "sample_submission.csv")
    submission  = build_submission(preds, df, sample_path)
 
    out_path = cfg["output"]["submission_path"]
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}  shape: {submission.shape}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)