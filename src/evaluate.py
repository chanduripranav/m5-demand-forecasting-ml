from __future__ import annotations

from typing import Dict, Iterable

from src.utils import mae, rmse


def evaluate_predictions(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Dict[str, float]:
    """Compute standard regression metrics."""

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

