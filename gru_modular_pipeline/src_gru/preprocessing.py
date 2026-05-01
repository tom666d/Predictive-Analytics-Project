"""Preprocessing utilities for the modular GRU M5 pipeline."""

import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible PyTorch/Numpy runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(cfg: dict) -> str:
    """Return the first existing data path from config."""
    for path in cfg["data"]["path_candidates"]:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No data path found. Edit configs/gru_config.yaml -> data.path_candidates."
    )


def load_m5_raw(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, sample submission, calendar, and sell price CSVs."""
    data_dir = resolve_data_dir(cfg)
    print(f"Using DATA_DIR: {data_dir}")

    sales_path = os.path.join(data_dir, cfg["data"]["sales_file"])
    sample_path = os.path.join(data_dir, cfg["data"]["sample_submission_file"])
    calendar_path = os.path.join(data_dir, cfg["data"]["calendar_file"])
    price_path = os.path.join(data_dir, cfg["data"]["prices_file"])

    sales_df = pd.read_csv(sales_path)
    sample_sub = pd.read_csv(sample_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(price_path) if os.path.exists(price_path) else pd.DataFrame()

    max_series = cfg["data"].get("max_series")
    if max_series is not None:
        sales_df = sales_df.iloc[: int(max_series)].copy()
        print(f"Debug mode: using first {len(sales_df)} series")

    return sales_df, sample_sub, calendar, prices


def get_day_cols(sales_df: pd.DataFrame) -> list:
    """Return d_ columns sorted by day number."""
    return sorted(
        [c for c in sales_df.columns if c.startswith("d_")],
        key=lambda x: int(x.replace("d_", "")),
    )


def build_sales_matrices(sales_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Create raw sales and log1p sales matrices from wide M5 sales data."""
    day_cols = get_day_cols(sales_df)
    sales_values = sales_df[day_cols].values.astype("float32")
    sales_log = np.log1p(sales_values).astype("float32")
    print(f"sales_values: {sales_values.shape}")
    return sales_values, sales_log, day_cols


def encode_static_categories(sales_df: pd.DataFrame, cat_cols: list) -> Tuple[np.ndarray, Dict[str, Dict[str, int]], list]:
    """Encode static item/store/category columns for embeddings."""
    cat_maps = {}
    cat_sizes = []
    encoded_cols = []

    for col in cat_cols:
        values = sales_df[col].astype(str)
        uniques = sorted(values.unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_maps[col] = mapping
        cat_sizes.append(len(mapping))
        encoded_cols.append(values.map(mapping).values.astype("int64"))
        print(f"{col}: {len(mapping)} unique values")

    cat_matrix = np.vstack(encoded_cols).T.astype("int64")
    return cat_matrix, cat_maps, cat_sizes
