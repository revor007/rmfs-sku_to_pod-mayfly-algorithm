from functools import lru_cache
from math import isfinite
from pathlib import Path

import pandas as pd


ORDER_SKU_CANDIDATES = ["商品编码", "item_code"]
PRODUCT_SKU_CANDIDATES = ["item_code", "商品编码"]


def normalize_column_name(name):
    return str(name).replace("\ufeff", "").strip()


def find_column(columns, candidates):
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def normalize_item_code(value):
    if pd.isna(value):
        return ""

    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return text


def find_preprocessing_dir(base_dir):
    base_dir = Path(base_dir).resolve()
    candidates = [
        base_dir / "Preprocessing",
        base_dir.parent / "Preprocessing",
        base_dir.parent.parent / "Preprocessing",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate the 'Preprocessing' directory from {base_dir}."
    )


def find_product_data_path(preprocessing_dir):
    path = Path(preprocessing_dir) / "preprocessed_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {path}")
    return path


def find_order_data_path(preprocessing_dir):
    candidates = sorted(
        path
        for path in Path(preprocessing_dir).glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {preprocessing_dir}."
        )
    return candidates[0]


@lru_cache(maxsize=None)
def _load_aligned_sku_set(base_dir_str):
    base_dir = Path(base_dir_str)
    preprocessing_dir = find_preprocessing_dir(base_dir)
    product_path = find_product_data_path(preprocessing_dir)
    order_path = find_order_data_path(preprocessing_dir)

    df_product = pd.read_csv(
        product_path,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )
    product_sku_col = find_column(df_product.columns, PRODUCT_SKU_CANDIDATES)
    product_skus = {
        sku
        for sku in df_product[product_sku_col].map(normalize_item_code)
        if sku
    }

    df_order = pd.read_csv(
        order_path,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )
    order_sku_col = find_column(df_order.columns, ORDER_SKU_CANDIDATES)
    order_skus = {
        sku
        for sku in df_order[order_sku_col].map(normalize_item_code)
        if sku
    }

    return frozenset(product_skus & order_skus)


def load_aligned_sku_set(base_dir):
    return set(_load_aligned_sku_set(str(Path(base_dir).resolve())))
