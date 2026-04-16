# M5 Forecasting - Walmart Sales

## Objective
Predict the unit sales of 30,490 Walmart products for the next 28 days.

## Data Source
[Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)


## Project Structure
 
```
PREDICTIVE-ANALYTICS-PROJECT/
├── configs/
│   └── config.yaml        ← Edit this to change features, model params, paths
├── data/
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   ├── sales_train_evaluation.csv
│   ├── sell_prices.csv
│   └── sample_submission.csv
├── notebooks/
│   └── feature_importance_analysis_v1.ipynb  ← Run once to decide feature list
├── src/
│   ├── preprocessing.py   ← Load CSVs, melt, merge, memory optimisation
│   ├── features.py        ← All feature engineering logic
│   ├── models.py          ← Model definitions, training, ensemble
│   └── train.py           ← Main entry point (run this)
├── .gitignore
├── README.md
└── requirements.txt
```
 
## How to Run
 
```bash
python src/train.py
# or specify a different config:
python src/train.py --config configs/config.yaml
```
 
## How to Change Things
 
| What you want to do | Where to change |
|---|---|
| Change which features to use | `configs/config.yaml` → `features.use` |
| Change lag / rolling windows | `configs/config.yaml` → `features.lags` / `features.rolling_means` |
| Change model hyperparameters | `configs/config.yaml` → `model.lgbm` or `model.xgb` |
| Switch to XGBoost | `configs/config.yaml` → `model.active_model: "xgb"` |
| Use ensemble (LGBM + XGB) | `configs/config.yaml` → `model.active_model: "ensemble"` |
| Add a new model type | `src/models.py` → add `get_<name>_model()` and register in `get_model()` |
| Add a new feature | `src/features.py` → add to `build_features()`, then add name to `configs/config.yaml` |
| Change data or output path | `configs/config.yaml` → `data.path` / `output.submission_path` |
 
## Team Workflow
 
**Step 1 — Feature selection** (run once):
- Open `notebooks/feature_importance_analysis_v1.ipynb`
- Run it to see which features matter
- Update `configs/config.yaml` → `features.use` accordingly
**Step 2 — Experiment with models**:
- Change `model.active_model` in `configs/config.yaml`
- Run `python src/train.py`
- Compare WRMSSE scores
**Step 3 — Try ensemble**:
- Set `model.active_model: "ensemble"` in config
- Adjust `model.ensemble.weights` to weight better-performing models higher
