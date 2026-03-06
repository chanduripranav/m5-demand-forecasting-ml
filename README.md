## M5 Demand Forecasting (Accuracy) – End-to-End Project

This repository contains a complete, production-style Python project for demand forecasting using the Kaggle **M5 Forecasting – Accuracy** dataset.  
The goal is to train an XGBoost model that predicts **daily unit sales for each product–store time series** and produce **next 28‑day forecasts** for a configurable subset of series so it runs comfortably on a laptop.

---

### 1. Repository Structure

- **`data/`**: Place the raw Kaggle CSVs here (not tracked).
- **`notebooks/`**
  - `01_data_loading_and_eda.ipynb`
  - `02_feature_engineering.ipynb`
  - `03_model_training.ipynb`
  - `04_forecast_next_28_days.ipynb`
- **`src/`**
  - `config.py`: Central configuration (paths, subsetting, split, model, forecast).
  - `data_loaders.py`: Load CSVs, melt sales to long format, merge calendar & prices.
  - `preprocessing.py`: Apply subsetting strategies and basic preprocessing.
  - `features.py`: Lag/rolling features, time, price, event, SNAP features.
  - `model.py`: XGBoost model utilities (build, train, save/load, feature cols).
  - `train.py`: CLI script to train and evaluate the model.
  - `predict.py`: CLI script to forecast the next 28 days and plot results.
  - `evaluate.py`: Metric helpers (MAE, RMSE).
  - `utils.py`: Logging, metrics, JSON I/O, time-based split helpers.
- **`models/`**: Saved model artifacts (`xgb_model.pkl`, `feature_cols.pkl`).
- **`outputs/`**: Plots, metrics, and forecast CSVs (`metrics.json`, `forecast_next_28.csv`, `forecast_plots.png`).
- **`requirements.txt`**: Python dependencies.

---

### 2. Dataset and Manual Download (Kaggle)

This project assumes you **download the files manually** from Kaggle (no Kaggle API required).

- **Competition**: “M5 Forecasting – Accuracy” on Kaggle.
- Required CSVs (place all in `./data/`):
  - `calendar.csv`
  - `sales_train_validation.csv`
  - `sell_prices.csv`
  - `sample_submission.csv` (not used directly in training but useful for reference).

After download, your `data/` folder should look like:

- `data/calendar.csv`
- `data/sales_train_validation.csv`
- `data/sell_prices.csv`
- `data/sample_submission.csv`

If any file is missing, the scripts will raise a clear error.

---

### 3. Environment Setup (Windows, Python 3.10+)

#### 3.1. Create and activate a Conda environment

```powershell
cd C:\projects\m5-demand-forecast

conda create -n m5-forecast python=3.10 -y
conda activate m5-forecast
```

#### 3.2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

---

### 4. Subsetting Strategies (Runtime-Friendly)

Training on the full M5 dataset is heavy, so the project uses configurable subsetting to run comfortably on a laptop.

Configuration is defined in `src/config.py`, and can be overridden via CLI arguments.

- **Option A – `state_store_cat` (default)**  
  Train on a single **state + store + category**:
  - Default: `state_id="CA"`, `store_id="CA_1"`, `cat_id="HOBBIES"`.

- **Option B – `random_series`**  
  Train on **N random series IDs**:
  - Default: `n_random_series = 200`.

You choose the option via the `--subset` argument for both training and forecasting:

- `--subset state_store_cat` (plus `--state`, `--store`, `--cat` if you want to override defaults)
- `--subset random_series --n_series 200`

---

### 5. How the Pipeline Works

#### 5.1. Data loading and reshape

- Load `sales_train_validation.csv` in **wide** format (columns `d_1`..`d_1913`).
- Melt to **long** format with columns:
  - `[id, item_id, dept_id, cat_id, store_id, state_id, d, sales]`.
- Merge with `calendar.csv` on `d` to add:
  - `date`, `wm_yr_wk`, `event_name_1`, `event_type_1`, `snap_CA`, `snap_TX`, `snap_WI`, etc.
- Merge with `sell_prices.csv` using keys:
  - `[store_id, item_id, wm_yr_wk]`.
- Convert `date` to `datetime`, downcast numerics to `float32` where appropriate.

#### 5.2. Feature engineering

All features are computed **per time series (`id`)**:

- **Lag features**:
  - `lag_1`, `lag_7`, `lag_14`, `lag_28` from `sales`, shifted appropriately.

- **Rolling features** (no leakage; use shifted sales):
  - `rolling_mean_7`, `rolling_mean_28`, `rolling_std_28`.

- **Time features**:
  - `dayofweek`, `weekofyear`, `month`, `year`, `is_weekend`.

- **Price features**:
  - `sell_price` (float32).
  - `price_change_1`: week-to-week percentage change per `store_id` + `item_id`.
  - `price_roll_mean_4w`: 4-week rolling mean of `sell_price`.

- **Event features**:
  - `is_event`: indicator if `event_name_1` is non-null.
  - `event_type_1`: one-hot encoded via `pandas.get_dummies` (e.g., `event_type_Sporting`, etc.).

- **SNAP feature**:
  - Uses the correct SNAP column based on `state_id`:
    - CA → `snap_CA`
    - TX → `snap_TX`
    - WI → `snap_WI`
  - Combined into a single `snap` flag.

Rows that don’t have enough history to fill all lag/rolling features (e.g. first 28 days) are dropped from the modeling dataset.

#### 5.3. Train/validation split (time-based)

- A strictly **time-based** split is applied:
  - Last `validation_days` (default **56**) days form the validation set.
  - Earlier days are used for training only.
  - Validation rows occur strictly after training rows to avoid leakage.

- **Baseline model**:
  - Naive forecast: `ŷ_t = lag_1`.
  - Metrics: MAE and RMSE on the validation window.

#### 5.4. Model training

- Model: **XGBoost regressor** (CPU only, `tree_method="hist"`).
- Reasonable defaults (overridable via config/CLI):
  - `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, regularization.
- Uses **early stopping** on the validation set.
- Saves:
  - Model: `./models/xgb_model.pkl`
  - Feature columns: `./models/feature_cols.pkl`
  - Metrics: `./outputs/metrics.json`

---

### 6. Running Training and Forecasting

All commands assume you are in the repo root (`C:\projects\m5-demand-forecast`) with the Conda environment activated.

#### 6.1. Train the model

- **Example: default subset (1 state + 1 store + 1 category)**:

```powershell
python -m src.train --subset state_store_cat --state CA --store CA_1 --cat HOBBIES
```

- **Example: random 200 series**:

```powershell
python -m src.train --subset random_series --n_series 200
```

- Optional overrides:
  - `--validation_days 56`
  - `--n_estimators 400`
  - `--learning_rate 0.05`

Outputs:
- **Console**: prints baseline (lag-1) MAE/RMSE and model MAE/RMSE.
- **Artifacts**:
  - `models/xgb_model.pkl`
  - `models/feature_cols.pkl`
  - `outputs/metrics.json`

#### 6.2. Forecast the next 28 days

- **Example: forecast 28 days (default)**:

```powershell
python -m src.predict --days 28
```

- With custom subset (must match how you trained, or at least be compatible):

```powershell
python -m src.predict --subset random_series --n_series 200 --days 28
```

Outputs:

- `outputs/forecast_next_28.csv`:
  - Columns: `[id, date, horizon, y_pred]`
  - `horizon` runs from 1 to 28 for each series.
- `outputs/forecast_plots.png`:
  - A plot of historical + forecasted values for **3 sample series**.

---

### 7. Notebooks

Each notebook is designed to run **independently** and imports reusable logic from `src/` instead of duplicating code.

- **`01_data_loading_and_eda.ipynb`**
  - Loads raw CSVs via `data_loaders`.
  - Explores basic shapes and summary stats.
  - Simple time series plots for a few series.

- **`02_feature_engineering.ipynb`**
  - Uses `preprocessing.prepare_base_dataframe` and `features.build_features`.
  - Shows examples of lag, rolling, time, price, event, and SNAP features.

- **`03_model_training.ipynb`**
  - Demonstrates a full training run in notebook form (mirroring `src/train.py`).
  - Displays baseline vs model metrics.

- **`04_forecast_next_28_days.ipynb`**
  - Loads a trained model and feature list from `models/`.
  - Calls the forecasting functions to generate next-28-day predictions.
  - Visualizes forecasts for several series.

---

### 8. Features and Evaluation Metrics

- **Target**:
  - Daily unit sales (`sales`) per product–store series (`id`).

- **Features**:
  - Lag features: `lag_1`, `lag_7`, `lag_14`, `lag_28`.
  - Rolling features: `rolling_mean_7`, `rolling_mean_28`, `rolling_std_28`.
  - Time: `dayofweek`, `weekofyear`, `month`, `year`, `is_weekend`.
  - Price: `sell_price`, `price_change_1`, `price_roll_mean_4w`.
  - Event: `is_event`, `event_type_*` one-hot columns.
  - SNAP: combined `snap` flag.

- **Metrics**:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)

These are computed both for the baseline (`lag_1`) and for the XGBoost model on the validation window.

---

### 9. Expected Outputs and Where to Find Them

- **Training outputs**
  - `models/xgb_model.pkl` – trained XGBoost model.
  - `models/feature_cols.pkl` – list of feature column names used for training.
  - `outputs/metrics.json` – JSON with baseline and model metrics (MAE, RMSE).

- **Forecasting outputs**
  - `outputs/forecast_next_28.csv` – next 28‑day forecasts for each series.
  - `outputs/forecast_plots.png` – visualization of historical vs forecast for 3 sample series.

---

### 10. Notes and Limitations

- This project is optimized to run on **CPU** only and uses subsetting to keep runtime within **10–20 minutes** on a typical laptop.
- The forecasting step uses an **iterative strategy**:
  - Starting from the last observed date, it builds features step by step, uses the model to predict the next day, appends that prediction to the history, and repeats for horizons 1..28.
  - Future `sell_price` is assumed to stay at the **last observed value** per item–store; event and SNAP information come from the `calendar` data when available.
- The implementation focuses on clarity and reproducibility rather than squeezing out the absolute best competition score.

You can now explore, train, and forecast using the M5 dataset end‑to‑end from this repository.

