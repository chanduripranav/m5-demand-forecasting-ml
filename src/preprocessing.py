from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PathsConfig, SubsetConfig
from src.data_loaders import (
    load_base_data,
    melt_sales_to_long,
    merge_calendar_and_prices,
)
from src.utils import get_logger


logger = get_logger()


def subset_sales_wide(
    sales: pd.DataFrame,
    subset_cfg: SubsetConfig,
) -> pd.DataFrame:
    """Apply subsetting strategy on the wide-format sales data."""

    if subset_cfg.subset_type == "state_store_cat":
        logger.info(
            "Subsetting by state_store_cat: state=%s, store=%s, cat=%s",
            subset_cfg.state_id,
            subset_cfg.store_id,
            subset_cfg.cat_id,
        )
        mask = (
            (sales["state_id"] == subset_cfg.state_id)
            & (sales["store_id"] == subset_cfg.store_id)
            & (sales["cat_id"] == subset_cfg.cat_id)
        )
        subset = sales.loc[mask].copy()
    elif subset_cfg.subset_type == "random_series":
        logger.info(
            "Subsetting by random_series: n_series=%d, seed=%d",
            subset_cfg.n_random_series,
            subset_cfg.random_seed,
        )
        random.seed(subset_cfg.random_seed)
        series_ids = list(sales["id"].unique())
        if subset_cfg.n_random_series < len(series_ids):
            chosen_ids = random.sample(series_ids, subset_cfg.n_random_series)
        else:
            chosen_ids = series_ids
        subset = sales[sales["id"].isin(chosen_ids)].copy()
    else:
        raise ValueError(f"Unknown subset_type: {subset_cfg.subset_type}")

    logger.info("Subset has %d series (rows) in wide format", subset.shape[0])
    return subset


def prepare_base_dataframe(
    paths: PathsConfig,
    subset_cfg: SubsetConfig,
) -> pd.DataFrame:
    """Load raw data, apply subsetting, reshape, and merge calendar and prices.

    Returns a long-format DataFrame with:
    [id, item_id, dept_id, cat_id, store_id, state_id, d, sales,
     date, wm_yr_wk, event_name_1, event_type_1, snap_CA, snap_TX, snap_WI,
     sell_price]
    """

    calendar, sell_prices, sales = load_base_data(paths.data_dir)
    sales_subset = subset_sales_wide(sales, subset_cfg)
    sales_long = melt_sales_to_long(sales_subset)
    df = merge_calendar_and_prices(sales_long, calendar, sell_prices)

    # Reduce memory where possible
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "id", "d"]:
        df[col] = df[col].astype("category")

    df["sales"] = df["sales"].astype(np.float32)
    df["wm_yr_wk"] = df["wm_yr_wk"].astype(np.int32)

    return df


def get_series_ids_and_dates(df: pd.DataFrame) -> Tuple[int, int]:
    """Return number of unique series and dates (for logging / EDA)."""

    n_series = df["id"].nunique()
    n_dates = df["date"].nunique()
    return n_series, n_dates

