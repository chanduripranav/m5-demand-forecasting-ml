from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger


logger = get_logger()


class MissingCSVError(FileNotFoundError):
    """Raised when a required CSV file is missing in the data directory."""


def _require_file(path: Path) -> None:
    if not path.exists():
        raise MissingCSVError(
            f"Required CSV not found: {path}. "
            "Please download the M5 dataset from Kaggle and place the CSVs in the ./data directory."
        )


def load_calendar(data_dir: Path) -> pd.DataFrame:
    """Load calendar.csv."""

    path = data_dir / "calendar.csv"
    _require_file(path)
    logger.info("Loading calendar from %s", path)
    df = pd.read_csv(path)
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sell_prices(data_dir: Path) -> pd.DataFrame:
    """Load sell_prices.csv."""

    path = data_dir / "sell_prices.csv"
    _require_file(path)
    logger.info("Loading sell_prices from %s", path)
    df = pd.read_csv(path)
    # Cast to smaller dtypes where possible
    df["wm_yr_wk"] = df["wm_yr_wk"].astype(np.int32)
    df["sell_price"] = df["sell_price"].astype(np.float32)
    return df


def load_sales_train_validation(data_dir: Path) -> pd.DataFrame:
    """Load sales_train_validation.csv (wide format)."""

    path = data_dir / "sales_train_validation.csv"
    _require_file(path)
    logger.info("Loading sales_train_validation from %s", path)
    df = pd.read_csv(path)
    return df


def melt_sales_to_long(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide daily sales (d_1..d_1913) to long format.

    Result columns: [id, item_id, dept_id, cat_id, store_id, state_id, d, sales]
    """

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    logger.info("Melting sales data to long format with %d value columns", len(value_cols))
    df_long = sales_wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="d",
        value_name="sales",
    )
    df_long["sales"] = df_long["sales"].astype(np.float32)
    return df_long


def merge_calendar_and_prices(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Merge long sales with calendar and sell_prices information."""

    logger.info("Merging sales with calendar")
    cal_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    calendar_small = calendar[cal_cols].copy()
    df = sales_long.merge(calendar_small, on="d", how="left")

    logger.info("Merging with sell_prices")
    df = df.merge(
        sell_prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # Ensure date is datetime and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def load_base_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for raw CSVs."""

    calendar = load_calendar(data_dir)
    sell_prices = load_sell_prices(data_dir)
    sales = load_sales_train_validation(data_dir)
    return calendar, sell_prices, sales

