from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ProjectConfig, SubsetConfig, get_default_config
from src.data_loaders import load_calendar
from src.features import add_time_price_event_snap_features
from src.model import load_feature_columns, load_model
from src.preprocessing import prepare_base_dataframe
from src.utils import get_logger, ensure_dir


logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast next N days using trained model.")

    parser.add_argument(
        "--subset",
        type=str,
        default="state_store_cat",
        choices=["state_store_cat", "random_series"],
        help="Subsetting strategy: 'state_store_cat' or 'random_series'.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="CA",
        help="State ID for state_store_cat subset (e.g. CA, TX, WI).",
    )
    parser.add_argument(
        "--store",
        type=str,
        default="CA_1",
        help="Store ID for state_store_cat subset (e.g. CA_1).",
    )
    parser.add_argument(
        "--cat",
        type=str,
        default="HOBBIES",
        help="Category ID for state_store_cat subset (e.g. HOBBIES, FOODS, HOUSEHOLD).",
    )
    parser.add_argument(
        "--n_series",
        type=int,
        default=200,
        help="Number of random series to sample when subset='random_series'.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=28,
        help="Number of days to forecast (default: 28).",
    )

    return parser.parse_args()


def update_config_from_args(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    subset_cfg: SubsetConfig = cfg.subset
    subset_cfg.subset_type = args.subset  # type: ignore[assignment]
    subset_cfg.state_id = args.state
    subset_cfg.store_id = args.store
    subset_cfg.cat_id = args.cat
    subset_cfg.n_random_series = args.n_series

    cfg.forecast.forecast_days = args.days
    return cfg


def _get_future_calendar(calendar: pd.DataFrame, last_date: pd.Timestamp, days: int) -> pd.DataFrame:
    mask = (calendar["date"] > last_date) & (calendar["date"] <= last_date + pd.Timedelta(days=days))
    return calendar.loc[mask].copy()


def _build_future_rows_for_series(
    hist_df: pd.DataFrame,
    future_calendar: pd.DataFrame,
    forecast_days: int,
) -> pd.DataFrame:
    """Create base rows for future dates for a single series (without dynamic lags)."""

    last_row = hist_df.sort_values("date").iloc[-1]
    series_id = last_row["id"]

    records: List[Dict[str, object]] = []
    for _, cal_row in future_calendar.iterrows():
        rec: Dict[str, object] = {
            "id": series_id,
            "item_id": last_row["item_id"],
            "dept_id": last_row["dept_id"],
            "cat_id": last_row["cat_id"],
            "store_id": last_row["store_id"],
            "state_id": last_row["state_id"],
            "d": cal_row["d"],
            "date": cal_row["date"],
            "wm_yr_wk": cal_row["wm_yr_wk"],
            "event_name_1": cal_row["event_name_1"],
            "event_type_1": cal_row["event_type_1"],
            "snap_CA": cal_row["snap_CA"],
            "snap_TX": cal_row["snap_TX"],
            "snap_WI": cal_row["snap_WI"],
            # Price features: keep last known sell_price and derived stats constant
            "sell_price": last_row.get("sell_price"),
        }
        # Keep price_change_1 and price_roll_mean_4w constant at last observed
        if "price_change_1" in hist_df.columns:
            rec["price_change_1"] = last_row.get("price_change_1")
        if "price_roll_mean_4w" in hist_df.columns:
            rec["price_roll_mean_4w"] = last_row.get("price_roll_mean_4w")

        records.append(rec)

    return pd.DataFrame.from_records(records)


def forecast_next_days(
    base_df: pd.DataFrame,
    calendar: pd.DataFrame,
    model_path: Path,
    feature_cols_path: Path,
    forecast_days: int,
) -> pd.DataFrame:
    """Iteratively forecast the next `forecast_days` for each series in base_df."""

    model = load_model(model_path)
    feature_cols = load_feature_columns(feature_cols_path)

    last_date: pd.Timestamp = base_df["date"].max()
    logger.info("Last observed date in training data: %s", last_date.date())

    future_calendar = _get_future_calendar(calendar, last_date, forecast_days)
    future_calendar = future_calendar.sort_values("date")
    if len(future_calendar) < forecast_days:
        logger.warning(
            "Requested %d forecast days but calendar only has %d future days; using %d.",
            forecast_days,
            len(future_calendar),
            len(future_calendar),
        )
        forecast_days = len(future_calendar)

    forecasts: List[Dict[str, object]] = []

    for series_id, hist_df in base_df.groupby("id"):
        hist_df = hist_df.sort_values("date")
        sales_history = hist_df["sales"].astype(np.float32).tolist()

        future_base = _build_future_rows_for_series(hist_df, future_calendar, forecast_days)

        # For each horizon, build dynamic features based on updated history
        for horizon_idx, (_, row) in enumerate(future_base.iterrows(), start=1):
            # Dynamic lag features
            def _get_lag(offset: int) -> float:
                if len(sales_history) < offset:
                    return float("nan")
                return float(sales_history[-offset])

            lag_1 = _get_lag(1)
            lag_7 = _get_lag(7)
            lag_14 = _get_lag(14)
            lag_28 = _get_lag(28)

            # Rolling features based on history excluding current prediction
            def _rolling_mean(window: int) -> float:
                if not sales_history:
                    return float("nan")
                window_vals = sales_history[-window:]
                return float(np.mean(window_vals))

            def _rolling_std(window: int) -> float:
                if len(sales_history) < 2:
                    return float("nan")
                window_vals = sales_history[-window:]
                return float(np.std(window_vals, ddof=1))

            record = row.to_dict()
            record.update(
                {
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_14": lag_14,
                    "lag_28": lag_28,
                    "rolling_mean_7": _rolling_mean(7),
                    "rolling_mean_28": _rolling_mean(28),
                    "rolling_std_28": _rolling_std(28),
                }
            )

            feat_df = pd.DataFrame([record])
            feat_df = add_time_price_event_snap_features(feat_df)

            # Align columns with training feature set
            for col in feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            extra_cols = [c for c in feat_df.columns if c not in feature_cols]
            feat_df = feat_df[feature_cols].astype(np.float32)

            y_pred = float(model.predict(feat_df.values)[0])
            sales_history.append(y_pred)

            forecasts.append(
                {
                    "id": series_id,
                    "date": row["date"],
                    "horizon": horizon_idx,
                    "y_pred": y_pred,
                }
            )

    return pd.DataFrame(forecasts)


def plot_sample_forecasts(
    base_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: Path,
    n_series: int = 3,
) -> None:
    """Plot historical and forecasted sales for a few sample series."""

    ensure_dir(output_path.parent)
    sample_ids = list(base_df["id"].unique())[:n_series]

    plt.figure(figsize=(12, 8))
    for idx, series_id in enumerate(sample_ids, start=1):
        plt.subplot(n_series, 1, idx)
        hist = base_df[base_df["id"] == series_id].sort_values("date")
        fut = forecast_df[forecast_df["id"] == series_id].sort_values("date")

        plt.plot(hist["date"], hist["sales"], label="History")
        plt.plot(fut["date"], fut["y_pred"], label="Forecast", linestyle="--")
        plt.title(f"Series: {series_id}")
        plt.ylabel("Sales")
        if idx == 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved forecast plots to %s", output_path)


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)

    logger.info("Configuration for forecasting: %s", cfg)

    # Load model and features
    model_path = cfg.paths.models_dir / "xgb_model.pkl"
    feature_cols_path = cfg.paths.models_dir / "feature_cols.pkl"
    if not model_path.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(
            f"Model or feature column file not found. "
            f"Expected model at {model_path} and feature columns at {feature_cols_path}. "
            "Please run training first: python -m src.train"
        )

    # Prepare historical data
    base_df = prepare_base_dataframe(cfg.paths, cfg.subset)

    # Load calendar for future dates
    calendar = load_calendar(cfg.paths.data_dir)

    forecast_df = forecast_next_days(
        base_df=base_df,
        calendar=calendar,
        model_path=model_path,
        feature_cols_path=feature_cols_path,
        forecast_days=cfg.forecast.forecast_days,
    )

    output_csv = cfg.paths.outputs_dir / "forecast_next_28.csv"
    ensure_dir(output_csv.parent)
    forecast_df.to_csv(output_csv, index=False)
    logger.info("Saved forecasts to %s", output_csv)

    # Plot few sample series
    plot_path = cfg.paths.outputs_dir / "forecast_plots.png"
    plot_sample_forecasts(base_df, forecast_df, plot_path, n_series=3)


if __name__ == "__main__":
    main()

