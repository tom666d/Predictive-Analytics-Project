"""
models.py
─────────
Defines, trains, and ensembles models.
To add a new model: add a function get_<name>_model(cfg) and register it in get_model().
 
Usage:
    from src.models import train_store_models, predict_store
"""
 
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
 
# Optional: XGBoost (only imported if needed)
def _get_xgb():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
 
 
# ── Model constructors ───────────────────────────────────────
 
def get_lgbm_model(cfg: dict) -> LGBMRegressor:
    p = cfg["model"]["lgbm"]
    return LGBMRegressor(
        objective             = p["objective"],
        tweedie_variance_power= p["tweedie_variance_power"],
        n_estimators          = p["n_estimators"],
        learning_rate         = p["learning_rate"],
        num_leaves            = p["num_leaves"],
        subsample             = p["subsample"],
        colsample_bytree      = p["colsample_bytree"],
    )
 
 
def get_xgb_model(cfg: dict):
    XGBRegressor = _get_xgb()
    p = cfg["model"]["xgb"]
    return XGBRegressor(
        objective             = p["objective"],
        tweedie_variance_power= p["tweedie_variance_power"],
        n_estimators          = p["n_estimators"],
        learning_rate         = p["learning_rate"],
        max_depth             = p["max_depth"],
        subsample             = p["subsample"],
        colsample_bytree      = p["colsample_bytree"],
        tree_method           = p["tree_method"],
    )
 
 
def get_model(name: str, cfg: dict):
    """Factory: returns an untrained model instance by name."""
    if name == "lgbm":
        return get_lgbm_model(cfg)
    elif name == "xgb":
        return get_xgb_model(cfg)
    else:
        raise ValueError(f"Unknown model name: '{name}'. Add it to models.py.")
 
 
# ── Training ─────────────────────────────────────────────────
 
def train_store_models(
    train_df,
    val_df,
    features: list,
    target: str,
    cat_cols: list,
    cfg: dict,
    model_name: str = None,
) -> dict:
    """
    Trains one model per store.
 
    Parameters
    ----------
    train_df    : training split DataFrame
    val_df      : validation split DataFrame (pass None to skip early stopping)
    features    : list of feature column names
    target      : target column name ("sales")
    cat_cols    : list of categorical feature column names
    cfg         : config dict
    model_name  : override active_model from config (optional)
 
    Returns
    -------
    dict: {store_id: trained_model}
    """
    if model_name is None:
        model_name = cfg["model"]["active_model"]
 
    if model_name == "ensemble":
        raise ValueError(
            "Use train_ensemble_models() for ensemble training, "
            "not train_store_models()."
        )
 
    model_cfg  = cfg["model"][model_name]
    early_stop = model_cfg.get("early_stopping_rounds")
    log_every  = model_cfg.get("log_every", 100)
 
    stores  = train_df["store_id"].unique()
    models  = {}
 
    for store in stores:
        print(f"  Training {model_name} — store: {store}")
 
        tr = train_df[train_df["store_id"] == store]
        X_tr, y_tr = tr[features], tr[target]
 
        model = get_model(model_name, cfg)
 
        if val_df is not None and early_stop:
            va = val_df[val_df["store_id"] == store]
            X_va, y_va = va[features], va[target]
 
            if model_name == "lgbm":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="rmse",
                    categorical_feature=cat_cols,
                    callbacks=[
                        lgb.early_stopping(early_stop, verbose=False),
                        lgb.log_evaluation(log_every),
                    ],
                )
            else:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=early_stop,
                    verbose=log_every,
                )
        else:
            # Final retrain on full data — no early stopping
            if model_name == "lgbm":
                model.fit(X_tr, y_tr, categorical_feature=cat_cols)
            else:
                model.fit(X_tr, y_tr)
 
        models[store] = model
 
    return models
 
 
def train_ensemble_models(
    train_df,
    val_df,
    features: list,
    target: str,
    cat_cols: list,
    cfg: dict,
) -> dict:
    """
    Trains all ensemble member models per store.
 
    Returns
    -------
    dict: {model_name: {store_id: trained_model}}
    """
    ensemble_cfg = cfg["model"]["ensemble"]
    all_models   = {}
 
    for name in ensemble_cfg["models"]:
        print(f"\n── Training ensemble member: {name} ──")
        all_models[name] = train_store_models(
            train_df, val_df, features, target, cat_cols, cfg, model_name=name
        )
 
    return all_models
 
 
# ── Prediction ───────────────────────────────────────────────
 
def predict_store(X, models: dict, store_col: str = "store_id") -> np.ndarray:
    """
    Runs store-wise prediction with a single model dict.
 
    Parameters
    ----------
    X       : feature DataFrame for one day (all stores)
    models  : {store_id: trained_model}
 
    Returns
    -------
    np.ndarray of predictions (same order as X)
    """
    y_pred = np.zeros(len(X))
    for store in X[store_col].unique():
        mask = X[store_col] == store
        y_pred[mask] = models[store].predict(X.loc[mask])
    return y_pred
 
 
def predict_store_ensemble(X, all_models: dict, cfg: dict) -> np.ndarray:
    """
    Runs weighted ensemble prediction across multiple model types.
 
    Parameters
    ----------
    X          : feature DataFrame for one day (all stores)
    all_models : {model_name: {store_id: trained_model}}
    cfg        : config dict
 
    Returns
    -------
    np.ndarray of weighted-average predictions
    """
    ensemble_cfg = cfg["model"]["ensemble"]
    model_names  = ensemble_cfg["models"]
    weights      = ensemble_cfg["weights"]
 
    assert abs(sum(weights) - 1.0) < 1e-6, "Ensemble weights must sum to 1.0"
 
    combined = np.zeros(len(X))
    for name, w in zip(model_names, weights):
        preds     = predict_store(X, all_models[name])
        combined += w * preds
 
    return combined