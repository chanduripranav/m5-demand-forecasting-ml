from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils import get_logger


logger = get_logger()


def add_lag_and_rolling_features(df: pd.DataFrame, group_col: str = "id") -> pd.DataFrame:
    """Add lag and rolling-window features per series.

    Lags: 1, 7, 14, 28 days.
    Rolling means and stds are computed on shifted sales to avoid leakage.
    """

    df = df.sort_values([group_col, "date"]).copy()
    group = df.groupby(group_col)["sales"]

    for lag in (1, 7, 14, 28):
        col = f"lag_{lag}"
        logger.info("Creating %s", col)
        df[col] = group.shift(lag).astype(np.float32)

    shifted = group.shift(1)
    logger.info("Creating rolling_mean_7")
    df["rolling_mean_7"] = (
        shifted.rolling(window=7, min_periods=1).mean().astype(np.float32)
    )
    logger.info("Creating rolling_mean_28 and rolling_std_28")
    df["rolling_mean_28"] = (
        shifted.rolling(window=28, min_periods=1).mean().astype(np.float32)
    )
    df["rolling_std_28"] = (
        shifted.rolling(window=28, min_periods=1).std().astype(np.float32)
    )

    return df


def add_time_price_event_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, price, event, and SNAP features.

    This function does not rely on future sales and can be applied to
    both historical and future rows.
    """

    df = df.copy()

    # Time features
    df["dayofweek"] = df["date"].dt.weekday.astype(np.int8)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)

    # Price features
    if "sell_price" in df.columns:
        df["sell_price"] = df["sell_price"].astype(np.float32)
        grp_price = df.groupby(["store_id", "item_id"])["sell_price"]
        df["price_change_1"] = grp_price.pct_change().astype(np.float32)
        df["price_roll_mean_4w"] = (
            grp_price.transform(lambda s: s.rolling(window=4, min_periods=1).mean())
            .astype(np.float32)
        )

    # Event features
    df["is_event"] = df["event_name_1"].notna().astype(np.int8)
    event_type = df["event_type_1"].fillna("None")
    dummies = pd.get_dummies(event_type, prefix="event_type")
    df = pd.concat([df, dummies], axis=1)

    # SNAP feature: map correct snap column by state_id
    snap = (
        ((df["state_id"] == "CA") & (df.get("snap_CA", 0) == 1))
        | ((df["state_id"] == "TX") & (df.get("snap_TX", 0) == 1))
        | ((df["state_id"] == "WI") & (df.get("snap_WI", 0) == 1))
    )
    df["snap"] = snap.astype(np.int8)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline for training.

    Applies:
    - Time, price, event, and SNAP features.
    - Lag and rolling features.
    - Drops rows with missing lag/rolling values.
    """

    logger.info("Building full feature set")
    df_feat = add_time_price_event_snap_features(df)
    df_feat = add_lag_and_rolling_features(df_feat, group_col="id")

    # Drop initial rows where lag features are not fully available
    lag_cols = ["lag_1", "lag_7", "lag_14", "lag_28"]
    roll_cols = ["rolling_mean_7", "rolling_mean_28", "rolling_std_28"]
    all_feature_cols = lag_cols + roll_cols
    before = len(df_feat)
    df_feat = df_feat.dropna(subset=all_feature_cols)
    after = len(df_feat)
    logger.info("Dropped %d rows with incomplete lag/rolling features", before - after)

    return df_feat


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names (excluding identifiers and target)."""

    exclude = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "date",
        "event_name_1",
        "event_type_1",
        "sales",
    }

    feature_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == "O":
            # Skip raw object/string columns
            continue
        feature_cols.append(col)

    logger.info("Using %d feature columns", len(feature_cols))
    return feature_cols

