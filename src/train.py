from __future__ import annotations

import argparse

import numpy as np

from src.config import ProjectConfig, SubsetConfig, get_default_config, ensure_project_dirs
from src.evaluate import evaluate_predictions
from src.features import build_features, get_feature_columns
from src.model import build_xgb_model, save_feature_columns, save_model
from src.preprocessing import prepare_base_dataframe
from src.utils import get_logger, save_json, train_val_time_split

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for M5 demand forecasting."
    )

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
        "--validation_days",
        type=int,
        default=56,
        help="Number of days to hold out for validation.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Override number of boosting rounds for XGBoost.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate for XGBoost.",
    )

    return parser.parse_args()


def update_config_from_args(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    subset_cfg: SubsetConfig = cfg.subset
    subset_cfg.subset_type = args.subset  # type: ignore[assignment]
    subset_cfg.state_id = args.state
    subset_cfg.store_id = args.store
    subset_cfg.cat_id = args.cat
    subset_cfg.n_random_series = args.n_series

    cfg.split.validation_days = args.validation_days

    if args.n_estimators is not None:
        cfg.model.n_estimators = args.n_estimators
    if args.learning_rate is not None:
        cfg.model.learning_rate = args.learning_rate

    return cfg


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg = update_config_from_args(cfg, args)
    ensure_project_dirs(cfg.paths)

    logger.info("Configuration: %s", cfg)

    # Data preparation
    logger.info("Preparing base dataframe")
    base_df = prepare_base_dataframe(cfg.paths, cfg.subset)
    logger.info(
        "Base dataframe shape after preprocessing: %s (nunique series=%d, ndates=%d)",
        base_df.shape,
        base_df["id"].nunique(),
        base_df["date"].nunique(),
    )

    # Feature engineering
    feat_df = build_features(base_df)
    feature_cols = get_feature_columns(feat_df)
    logger.info("Using %d feature columns", len(feature_cols))

    # Train/validation split (time-based)
    train_mask, val_mask = train_val_time_split(
        dates=feat_df["date"].values,
        validation_days=cfg.split.validation_days,
    )
    train_df = feat_df.loc[train_mask].copy()
    val_df = feat_df.loc[val_mask].copy()

    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df["sales"].astype(np.float32).values
    X_val = val_df[feature_cols].astype(np.float32).values
    y_val = val_df["sales"].astype(np.float32).values

    logger.info(
        "Train size: %d, Validation size: %d, Features: %d",
        X_train.shape[0],
        X_val.shape[0],
        len(feature_cols),
    )

    # Baseline (naive lag-1)
    if "lag_1" not in val_df.columns:
        raise RuntimeError("lag_1 feature missing from validation set for baseline calculation.")
    baseline_pred = val_df["lag_1"].values
    baseline_metrics = evaluate_predictions(y_val, baseline_pred)
    logger.info(
        "Baseline (naive lag-1) MAE=%.4f RMSE=%.4f",
        baseline_metrics["mae"],
        baseline_metrics["rmse"],
    )

    # Model training (robust: fit directly; no missing wrapper function)
    xgb_model = build_xgb_model(cfg.model)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Validation metrics
    val_pred = xgb_model.predict(X_val)
    model_metrics = evaluate_predictions(y_val, val_pred)
    logger.info(
        "Model metrics: MAE=%.4f RMSE=%.4f",
        model_metrics["mae"],
        model_metrics["rmse"],
    )

    # Save artifacts
    model_path = cfg.paths.models_dir / "xgb_model.pkl"
    feature_cols_path = cfg.paths.models_dir / "feature_cols.pkl"
    metrics_path = cfg.paths.outputs_dir / "metrics.json"

    save_model(xgb_model, model_path)
    save_feature_columns(feature_cols, feature_cols_path)
    all_metrics = {"baseline": baseline_metrics, "model": model_metrics}
    save_json(all_metrics, metrics_path)

    logger.info("Training complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()