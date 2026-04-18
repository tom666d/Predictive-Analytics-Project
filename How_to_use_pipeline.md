# M5 Forecasting — Pipeline Guide

## Overview

The pipeline is split into four modular files. Each person can work on a different file without conflicts.

```
src/
├── train.py           ← Main entry point. Run this to train and generate submission.
├── preprocessing.py   ← Data loading, merging, and cleaning.
├── features.py        ← All feature engineering logic.
└── models.py          ← Model definitions and ensemble logic.

configs/
└── config.yaml        ← Controls all parameters. Change settings here, not in code.
```

---

## How to Run

```bash
# Default config
python src/train.py

# Custom config
python src/train.py --config configs/config_yourname.yaml
```

---

## File Responsibilities

### `preprocessing.py`
Loads the 3 raw CSVs, melts wide → long, merges calendar and prices, and applies memory optimizations.

**Edit this file if you want to:**
- Change how data is cleaned
- Modify memory optimization logic
- Change how SNAP or event columns are handled

---

### `features.py`
Builds all features on top of the preprocessed DataFrame.

**Edit this file if you want to add or modify features.**

Example — adding `is_high_impact_event` and `price_vs_mean`:

```python
# In build_features() inside features.py

# High-impact holiday flag
high_impact = ['LaborDay', 'SuperBowl', 'Easter']
df['is_high_impact_event'] = df['event_name_1'].isin(high_impact).astype('int8')

# Price vs historical mean
df['price_vs_mean'] = (
    df.groupby('item_id')['sell_price']
    .transform(lambda x: x / x.mean())
    .astype('float32')
)
```

Then add the new feature names to `config.yaml` under `features.use` (see below).

> ⚠️ Also update `build_features_for_day()` with the same features if they depend on sales or price — this function is used during recursive prediction.

---

### `models.py`
Defines model constructors and training/prediction logic.

**Edit this file if you want to add a new model type.**

Example — adding CatBoost:

```python
# Step 1: Add constructor
def get_catboost_model(cfg):
    from catboost import CatBoostRegressor
    p = cfg["model"]["catboost"]
    return CatBoostRegressor(loss_function='Tweedie', learning_rate=p["learning_rate"])

# Step 2: Register in get_model()
def get_model(name, cfg):
    if name == "lgbm": return get_lgbm_model(cfg)
    elif name == "xgb": return get_xgb_model(cfg)
    elif name == "catboost": return get_catboost_model(cfg)  # add this
```

Then add settings to `config.yaml` and set `active_model: catboost`.

---

### `train.py`
Orchestrates the full pipeline. You usually don't need to edit this file.

**The pipeline runs in this order:**
1. Load config
2. Preprocess raw data
3. Build features
4. Train/val split
5. Train model(s) with early stopping
6. Retrain on full data
7. Recursive 28-day prediction
8. Generate `submission.csv`

---

## config.yaml Reference

All parameters are controlled here. **You should never hardcode values in `.py` files.**

```yaml
data:
  path: data/                  # Path to raw CSV files
  n_days: 500                  # How many days of history to use (increase if memory allows)
  val_days: 28                 # Last N days used as validation set

model:
  active_model: lgbm           # Options: lgbm | xgb | ensemble

  lgbm:
    objective: tweedie
    tweedie_variance_power: 1.1
    n_estimators: 1000
    learning_rate: 0.05
    num_leaves: 128
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
    log_every: 100

  xgb:
    objective: reg:tweedie
    tweedie_variance_power: 1.5
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    tree_method: hist
    early_stopping_rounds: 50
    log_every: 100

  ensemble:
    models: [lgbm, xgb]        # Which models to combine
    weights: [0.6, 0.4]        # Must sum to 1.0

features:
  lags: [1, 7, 28]             # Lag windows (add 14, 35, 42, 364, 365 if n_days is large enough)
  rolling_means: [3, 7, 28]    # Rolling mean windows
  use_rolling_std: false       # Set to true to add rolling_std_7

  use:                         # Final list of features passed to the model
    - lag_1
    - lag_7
    - lag_28
    - rmean_3
    - rmean_7
    - rmean_28
    - sell_price
    - price_change
    - price_mean_7
    - dayofweek
    - month
    - weekofyear
    - is_weekend
    - snap
    - event_type
    - store_sales_mean_7
    - cat_sales_mean_7
    - item_id
    - dept_id
    - cat_id
    - store_id
    - state_id

output:
  submission_path: submission.csv
  models_dir: models/
  skip_training: false         # Set to true to skip training and load saved models
```

---

## Common Workflows

### Run a full experiment
```bash
python src/train.py --config configs/config.yaml
```

### Skip training, just regenerate submission
```yaml
# In config.yaml
output:
  skip_training: true
```
```bash
python src/train.py
```

### Try ensemble (LightGBM + XGBoost)
```yaml
# In config.yaml
model:
  active_model: ensemble
  ensemble:
    models: [lgbm, xgb]
    weights: [0.6, 0.4]
```

### Run your own experiment without affecting others
```bash
# Copy the config and give it your name
cp configs/config.yaml configs/config_yourname.yaml

# Edit your copy
# Then run with your config
python src/train.py --config configs/config_yourname.yaml
```

---

## Feature Importance — How to Evaluate Features

After training, always check feature importance to see which features are actually helping the model.

```python
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# Load one store's model (e.g. CA_1)
import joblib
model = joblib.load('models/lgbm_CA_1.joblib')

# Get feature importance
importance = pd.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))

# Plot
importance.head(20).plot(
    kind='barh', x='feature', y='importance',
    title='Top 20 Feature Importances (CA_1)',
    figsize=(10, 8)
)
plt.tight_layout()
plt.show()
```

**How to interpret:**
- High importance → feature is frequently used by the model to make splits → keep it
- Near-zero importance → feature is not helping → consider removing it
- If a new feature you added has near-zero importance → it's not useful for this model

**Recommended workflow:**
1. Add a new feature to `features.py` and `config.yaml`
2. Train the model
3. Check feature importance
4. If importance is near zero → remove it and try something else
5. If importance is high → keep it and submit to Kaggle to verify the score improves

---

## Data Setup (Reminder)

Download data from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place it here:

```
data/
├── sales_train_validation.csv
├── sales_train_evaluation.csv
├── calendar.csv
├── sell_prices.csv
└── sample_submission.csv
```

> ⚠️ The folder MUST be named `data/` — this is required for `.gitignore` to prevent the data from being pushed to GitHub.

---

## Branch & Commit Convention

```bash
# Always pull before starting
git pull origin main

# Work on your own branch
git checkout -b feat/your-name-task
# e.g. git checkout -b feat/alice-add-holiday-features

# Push when done
git add .
git commit -m "Add is_high_impact_event and price_vs_mean features"
git push origin feat/alice-add-holiday-features
```