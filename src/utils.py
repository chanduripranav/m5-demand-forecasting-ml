from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


LOGGER_NAME = "m5_forecast"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean absolute error."""

    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float64)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root mean squared error."""

    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float64)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def train_val_time_split(
    dates: Iterable[np.datetime64],
    validation_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for train and validation based on the last `validation_days`.

    Parameters
    ----------
    dates:
        Sequence of pandas / numpy datetime-like values.
    validation_days:
        Number of days to allocate to validation from the end of the series.
    """

    dates_arr = np.asarray(dates)
    max_date = dates_arr.max()
    threshold = max_date - np.timedelta64(validation_days, "D")
    train_mask = dates_arr <= threshold
    val_mask = dates_arr > threshold
    return train_mask, val_mask

