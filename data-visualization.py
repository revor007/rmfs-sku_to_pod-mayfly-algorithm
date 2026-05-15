from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = BASE_DIR.parent / "Preprocessing"

ORDER_SUMMARY_OUTPUT_PATH = BASE_DIR / "order_data_summary_statistics.csv"
LINE_ITEM_SUMMARY_OUTPUT_PATH = BASE_DIR / "order_line_item_summary_statistics.csv"
ORDER_LEVEL_SUMMARY_OUTPUT_PATH = BASE_DIR / "order_level_summary_statistics.csv"
DAILY_SUMMARY_OUTPUT_PATH = BASE_DIR / "daily_order_summary_statistics.csv"

ORDER_ID_CANDIDATES = [
    "\u8ba2\u5355\u53f7",
    "order_id",
]
SKU_CANDIDATES = [
    "\u5546\u54c1\u7f16\u7801",
    "item_code",
]
PRODUCT_NAME_CANDIDATES = [
    "\u5546\u54c1\u540d\u79f0",
    "item_name",
]
QUANTITY_CANDIDATES = [
    "\u5546\u54c1\u6570\u91cf",
    "quantity",
]
CREATED_TIME_CANDIDATES = [
    "\u521b\u5efa\u65f6\u95f4",
    "created_at",
]
PRICE_CANDIDATES = [
    "price_per_piece",
    "unit_price",
]
SALES_CANDIDATES = [
    "sales",
    "revenue",
]


def find_order_data_path():
    candidates = sorted(
        path
        for path in PREPROCESSING_DIR.glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {PREPROCESSING_DIR}."
        )

    return candidates[0]


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


def get_writable_output_path(path):
    if not path.exists():
        return path

    try:
        with open(path, "a", encoding="utf-8"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_latest{path.suffix}")


def save_dataframe(df, path, **kwargs):
    output_path = get_writable_output_path(path)
    df.to_csv(output_path, **kwargs)
    return output_path


def load_order_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",")

    columns = {
        "order_id": find_column(df.columns, ORDER_ID_CANDIDATES),
        "sku": find_column(df.columns, SKU_CANDIDATES),
        "product_name": find_column(df.columns, PRODUCT_NAME_CANDIDATES),
        "quantity": find_column(df.columns, QUANTITY_CANDIDATES),
        "created_at": find_column(df.columns, CREATED_TIME_CANDIDATES),
        "price": find_column(df.columns, PRICE_CANDIDATES),
        "sales": find_column(df.columns, SALES_CANDIDATES),
    }

    df = df.copy()
    df[columns["order_id"]] = df[columns["order_id"]].astype(str).str.strip()
    df[columns["sku"]] = df[columns["sku"]].astype(str).str.strip()
    df[columns["product_name"]] = df[columns["product_name"]].astype(str).str.strip()
    df["_created_timestamp"] = pd.to_datetime(
        df[columns["created_at"]],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df["_created_date"] = df["_created_timestamp"].dt.normalize()

    return df, columns


def summarize_numeric_frame(frame, rename_map):
    summary = (
        frame.rename(columns=rename_map)
        .agg(["count", "mean", "median", "std", "min", "max", "sum"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "series"})
    )
    numeric_columns = summary.columns.drop("series")
    summary[numeric_columns] = summary[numeric_columns].round(4)
    return summary


def build_overview_summary(df, columns, source_path):
    order_col = columns["order_id"]
    sku_col = columns["sku"]
    product_col = columns["product_name"]
    quantity_col = columns["quantity"]
    price_col = columns["price"]
    sales_col = columns["sales"]

    order_level = df.groupby(order_col).agg(
        line_item_count=(order_col, "size"),
        distinct_sku_count=(sku_col, "nunique"),
        total_quantity=(quantity_col, "sum"),
        total_sales=(sales_col, "sum"),
    )

    dated_rows = df.dropna(subset=["_created_date"])
    daily_level = dated_rows.groupby("_created_date").agg(
        daily_order_count=(order_col, "nunique"),
        daily_line_item_count=(order_col, "size"),
        daily_quantity=(quantity_col, "sum"),
        daily_sales=(sales_col, "sum"),
    )

    start_timestamp = df["_created_timestamp"].min()
    end_timestamp = df["_created_timestamp"].max()

    records = [
        ("source_file", source_path.name),
        ("total_line_items", len(df)),
        ("unique_orders", df[order_col].nunique()),
        ("unique_skus", df[sku_col].nunique()),
        ("unique_product_names", df[product_col].nunique()),
        ("rows_with_missing_timestamp", int(df["_created_timestamp"].isna().sum())),
        (
            "date_range_start",
            start_timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(start_timestamp) else None,
        ),
        (
            "date_range_end",
            end_timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(end_timestamp) else None,
        ),
        ("active_days", int(daily_level.shape[0])),
        ("total_quantity", float(df[quantity_col].sum())),
        ("total_sales", float(df[sales_col].sum())),
        ("average_price_per_piece", float(df[price_col].mean())),
        ("median_price_per_piece", float(df[price_col].median())),
        ("average_line_items_per_order", float(order_level["line_item_count"].mean())),
        ("median_line_items_per_order", float(order_level["line_item_count"].median())),
        ("average_distinct_skus_per_order", float(order_level["distinct_sku_count"].mean())),
        ("median_distinct_skus_per_order", float(order_level["distinct_sku_count"].median())),
        ("average_quantity_per_order", float(order_level["total_quantity"].mean())),
        ("median_quantity_per_order", float(order_level["total_quantity"].median())),
        ("average_sales_per_order", float(order_level["total_sales"].mean())),
        ("median_sales_per_order", float(order_level["total_sales"].median())),
    ]

    if not daily_level.empty:
        records.extend(
            [
                ("average_daily_orders", float(daily_level["daily_order_count"].mean())),
                ("median_daily_orders", float(daily_level["daily_order_count"].median())),
                ("average_daily_line_items", float(daily_level["daily_line_item_count"].mean())),
                ("median_daily_line_items", float(daily_level["daily_line_item_count"].median())),
                ("average_daily_quantity", float(daily_level["daily_quantity"].mean())),
                ("median_daily_quantity", float(daily_level["daily_quantity"].median())),
                ("average_daily_sales", float(daily_level["daily_sales"].mean())),
                ("median_daily_sales", float(daily_level["daily_sales"].median())),
            ]
        )

    overview = pd.DataFrame(records, columns=["metric", "value"])
    return overview, order_level, daily_level


def main():
    source_path = find_order_data_path()
    df_order, columns = load_order_data(source_path)
    overview, order_level, daily_level = build_overview_summary(
        df_order,
        columns,
        source_path,
    )

    line_item_summary = summarize_numeric_frame(
        df_order[[columns["quantity"], columns["price"], columns["sales"]]],
        {
            columns["quantity"]: "item_quantity",
            columns["price"]: "price_per_piece",
            columns["sales"]: "sales",
        },
    )
    order_level_summary = summarize_numeric_frame(
        order_level,
        {
            "line_item_count": "line_items_per_order",
            "distinct_sku_count": "distinct_skus_per_order",
            "total_quantity": "quantity_per_order",
            "total_sales": "sales_per_order",
        },
    )
    daily_summary = summarize_numeric_frame(
        daily_level,
        {
            "daily_order_count": "orders_per_day",
            "daily_line_item_count": "line_items_per_day",
            "daily_quantity": "quantity_per_day",
            "daily_sales": "sales_per_day",
        },
    )

    order_summary_path = save_dataframe(
        overview,
        ORDER_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    line_item_summary_path = save_dataframe(
        line_item_summary,
        LINE_ITEM_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    order_level_summary_path = save_dataframe(
        order_level_summary,
        ORDER_LEVEL_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    daily_summary_path = save_dataframe(
        daily_summary,
        DAILY_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    print(f"Source file: {source_path}")
    print(f"Overview summary saved to: {order_summary_path}")
    print(f"Line-item summary saved to: {line_item_summary_path}")
    print(f"Order-level summary saved to: {order_level_summary_path}")
    print(f"Daily summary saved to: {daily_summary_path}")
    print("\nOverview summary:")
    print(overview.to_string(index=False))


if __name__ == "__main__":
    main()
