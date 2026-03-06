from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from xgboost import XGBRegressor

from src.config import ModelConfig
from src.utils import get_logger


logger = get_logger()


def build_xgb_model(cfg: ModelConfig) -> XGBRegressor:
    """Create an XGBRegressor with sensible defaults for CPU training."""

    params: Dict[str, Any] = {
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "reg_alpha": cfg.reg_alpha,
        "min_child_weight": cfg.min_child_weight,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": cfg.n_jobs,
        "random_state": 42,
    }

    params.update(cfg.model.extra_params if hasattr(cfg, "model") else cfg.extra_params)
    model = XGBRegressor(**params)
    return model


def train_xgb_regressor(
    model: XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ModelConfig,
) -> XGBRegressor:
    """Train XGBoost regressor with early stopping."""

    logger.info("Starting model training with %d training samples", X_train.shape[0])
    eval_set: List[Tuple[np.ndarray, np.ndarray]] = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="rmse",
        verbose=True,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    logger.info("Training finished. Best iteration: %s", getattr(model, "best_iteration", None))
    return model


def save_model(model: XGBRegressor, path: Path) -> None:
    """Persist model to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> XGBRegressor:
    """Load model from disk."""

    model: XGBRegressor = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def save_feature_columns(feature_cols: List[str], path: Path) -> None:
    """Persist feature column names."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, path)
    logger.info("Saved feature column list to %s", path)


def load_feature_columns(path: Path) -> List[str]:
    """Load feature column names."""

    feature_cols: List[str] = joblib.load(path)
    logger.info("Loaded feature column list from %s", path)
    return feature_cols

