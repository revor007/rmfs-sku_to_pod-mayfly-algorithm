from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MINIMUM_INVENTORY_OUTPUT_PATH = BASE_DIR / "minimum_inventory.csv"
EVALUATION_OUTPUT_PATH = BASE_DIR / "minimum_inventory_evaluation.csv"

LEAD_TIME_DAYS = 2
TARGET_SERVICE_LEVEL = 0.95
BOOTSTRAP_REPETITIONS = 10_000
RESAMPLING_METHOD = "with_replacement"
RANDOM_SEED = 42
TEST_FRACTION = 0.20

ORDER_ID_CANDIDATES = [
    "订单号",
    "order_id",
]
SKU_CANDIDATES = [
    "商品编码",
    "item_code",
]
PRODUCT_NAME_CANDIDATES = [
    "商品名称",
    "item_name",
]
QUANTITY_CANDIDATES = [
    "商品数量",
    "quantity",
]
CREATED_TIME_CANDIDATES = [
    "创建时间",
    "created_at",
]


def find_preprocessing_dir():
    candidates = [
        BASE_DIR / "Preprocessing",
        BASE_DIR.parent / "Preprocessing",
        BASE_DIR.parent.parent / "Preprocessing",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not locate the 'Preprocessing' directory.")


def find_order_data_path(preprocessing_dir):
    candidates = sorted(
        path
        for path in preprocessing_dir.glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {preprocessing_dir}."
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


def save_dataframe(df, path):
    output_path = get_writable_output_path(path)
    df.to_csv(
        output_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
    )
    return output_path


def make_console_safe(text):
    return str(text).encode("unicode_escape").decode("ascii")


def load_order_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",").copy()

    columns = {
        "order_id": find_column(df.columns, ORDER_ID_CANDIDATES),
        "sku": find_column(df.columns, SKU_CANDIDATES),
        "product_name": find_column(df.columns, PRODUCT_NAME_CANDIDATES),
        "quantity": find_column(df.columns, QUANTITY_CANDIDATES),
        "created_at": find_column(df.columns, CREATED_TIME_CANDIDATES),
    }

    df[columns["order_id"]] = df[columns["order_id"]].astype(str).str.strip()
    df[columns["sku"]] = df[columns["sku"]].astype(str).str.strip()
    df[columns["product_name"]] = df[columns["product_name"]].astype(str).str.strip()
    df[columns["quantity"]] = pd.to_numeric(df[columns["quantity"]], errors="coerce")
    if df[columns["quantity"]].isna().any():
        raise ValueError("Quantity column contains non-numeric values after parsing.")

    df["_created_timestamp"] = pd.to_datetime(
        df[columns["created_at"]],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df["_created_date"] = df["_created_timestamp"].dt.normalize()
    df = df.dropna(subset=["_created_date"]).copy()

    return df, columns


def build_daily_demand_series(df, columns):
    sku_col = columns["sku"]
    product_col = columns["product_name"]
    quantity_col = columns["quantity"]

    daily_demand = (
        df.groupby([sku_col, "_created_date"], as_index=False)[quantity_col]
        .sum()
        .rename(columns={quantity_col: "daily_demand"})
        .sort_values([sku_col, "_created_date"])
    )

    product_lookup = (
        df[[sku_col, product_col]]
        .drop_duplicates()
        .sort_values([sku_col, product_col])
        .drop_duplicates(subset=[sku_col], keep="first")
        .set_index(sku_col)[product_col]
    )

    series_by_sku = {}
    for sku, sku_frame in daily_demand.groupby(sku_col, sort=True):
        sku_frame = sku_frame.sort_values("_created_date")
        date_index = pd.date_range(
            sku_frame["_created_date"].min(),
            sku_frame["_created_date"].max(),
            freq="D",
        )
        dense_series = (
            sku_frame.set_index("_created_date")["daily_demand"]
            .reindex(date_index, fill_value=0)
            .astype(float)
        )

        series_by_sku[sku] = {
            "product_name": product_lookup.get(sku, ""),
            "first_observed_date": date_index.min(),
            "last_observed_date": date_index.max(),
            "demand": dense_series.to_numpy(dtype=np.float64),
        }

    return series_by_sku


def empirical_quantile(samples, alpha):
    sorted_samples = np.sort(np.asarray(samples, dtype=np.float64))
    if sorted_samples.size == 0:
        raise ValueError("Cannot compute a quantile from an empty sample.")

    index = int(np.ceil(alpha * sorted_samples.size) - 1)
    index = min(max(index, 0), sorted_samples.size - 1)
    return sorted_samples[index]


def rolling_lead_time_demand(demand, lead_time_days):
    demand = np.asarray(demand, dtype=np.float64)
    if demand.size < lead_time_days:
        return np.array([], dtype=np.float64)

    if lead_time_days == 1:
        return demand.copy()

    kernel = np.ones(lead_time_days, dtype=np.float64)
    return np.convolve(demand, kernel, mode="valid")


def simulate_lead_time_demand(
    demand,
    lead_time_days,
    bootstrap_repetitions,
    with_replacement,
    rng,
):
    demand = np.asarray(demand, dtype=np.float64)
    if demand.size == 0:
        raise ValueError("Demand history must contain at least one observation.")

    if lead_time_days == 1:
        return demand.copy()

    if with_replacement:
        sampled = rng.choice(
            demand,
            size=(bootstrap_repetitions, lead_time_days),
            replace=True,
        )
    else:
        if demand.size < lead_time_days:
            raise ValueError(
                "Without-replacement resampling requires history length >= lead time."
            )
        sampled = np.vstack(
            [
                rng.choice(demand, size=lead_time_days, replace=False)
                for _ in range(bootstrap_repetitions)
            ]
        )

    return sampled.sum(axis=1)


def calculate_inventory_parameters(
    demand,
    lead_time_days,
    service_level,
    bootstrap_repetitions,
    resampling_method,
    rng,
):
    with_replacement = resampling_method == "with_replacement"
    lead_time_samples = simulate_lead_time_demand(
        demand=demand,
        lead_time_days=lead_time_days,
        bootstrap_repetitions=bootstrap_repetitions,
        with_replacement=with_replacement,
        rng=rng,
    )

    reorder_point = int(np.ceil(max(0, empirical_quantile(lead_time_samples, service_level))))
    mean_lead_time_demand = float(np.mean(lead_time_samples))
    safety_stock = float(max(0.0, reorder_point - mean_lead_time_demand))

    return {
        "lead_time_samples": lead_time_samples,
        "mean_lead_time_demand": mean_lead_time_demand,
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "minimum_inventory": reorder_point,
    }


def split_train_test(demand, lead_time_days, test_fraction):
    demand = np.asarray(demand, dtype=np.float64)
    history_days = demand.size
    if history_days <= lead_time_days:
        return None, None

    test_days = max(lead_time_days, int(np.ceil(history_days * test_fraction)))
    test_days = min(test_days, history_days - 1)
    train_days = history_days - test_days
    if train_days < 1:
        return None, None

    return demand[:train_days], demand[train_days:]


def evaluate_holdout(
    demand,
    lead_time_days,
    service_level,
    bootstrap_repetitions,
    resampling_method,
    rng,
    test_fraction,
):
    train_demand, test_demand = split_train_test(demand, lead_time_days, test_fraction)
    if train_demand is None or test_demand is None:
        return {
            "evaluation_status": "not_enough_history_for_holdout",
            "train_days": np.nan,
            "test_days": np.nan,
            "number_of_test_cases": np.nan,
            "evaluation_reorder_point": np.nan,
            "evaluation_mean_lead_time_demand": np.nan,
            "evaluation_safety_stock": np.nan,
            "achieved_csl": np.nan,
            "shortage_units": np.nan,
            "fill_rate": np.nan,
            "total_actual_lead_time_demand": np.nan,
        }

    inventory_result = calculate_inventory_parameters(
        demand=train_demand,
        lead_time_days=lead_time_days,
        service_level=service_level,
        bootstrap_repetitions=bootstrap_repetitions,
        resampling_method=resampling_method,
        rng=rng,
    )

    actual_lead_time_demand = rolling_lead_time_demand(test_demand, lead_time_days)
    if actual_lead_time_demand.size == 0:
        return {
            "evaluation_status": "not_enough_test_windows",
            "train_days": train_demand.size,
            "test_days": test_demand.size,
            "number_of_test_cases": 0,
            "evaluation_reorder_point": inventory_result["reorder_point"],
            "evaluation_mean_lead_time_demand": inventory_result["mean_lead_time_demand"],
            "evaluation_safety_stock": inventory_result["safety_stock"],
            "achieved_csl": np.nan,
            "shortage_units": np.nan,
            "fill_rate": np.nan,
            "total_actual_lead_time_demand": np.nan,
        }

    shortage = np.maximum(0.0, actual_lead_time_demand - inventory_result["reorder_point"])
    total_actual = float(actual_lead_time_demand.sum())
    fill_rate = 1.0 if total_actual == 0 else float(1.0 - shortage.sum() / total_actual)

    return {
        "evaluation_status": "ok",
        "train_days": train_demand.size,
        "test_days": test_demand.size,
        "number_of_test_cases": int(actual_lead_time_demand.size),
        "evaluation_reorder_point": inventory_result["reorder_point"],
        "evaluation_mean_lead_time_demand": inventory_result["mean_lead_time_demand"],
        "evaluation_safety_stock": inventory_result["safety_stock"],
        "achieved_csl": float(np.mean(actual_lead_time_demand <= inventory_result["reorder_point"])),
        "shortage_units": float(shortage.sum()),
        "fill_rate": fill_rate,
        "total_actual_lead_time_demand": total_actual,
    }


def build_inventory_outputs(series_by_sku):
    minimum_inventory_rows = []
    evaluation_rows = []

    sorted_skus = sorted(series_by_sku)
    for sku_index, sku in enumerate(sorted_skus):
        sku_record = series_by_sku[sku]
        demand = sku_record["demand"]

        full_rng = np.random.default_rng(RANDOM_SEED + sku_index)
        inventory_result = calculate_inventory_parameters(
            demand=demand,
            lead_time_days=LEAD_TIME_DAYS,
            service_level=TARGET_SERVICE_LEVEL,
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
            resampling_method=RESAMPLING_METHOD,
            rng=full_rng,
        )

        minimum_inventory_rows.append(
            {
                "item_code": sku,
                "product_name": sku_record["product_name"],
                "first_observed_date": sku_record["first_observed_date"].date().isoformat(),
                "last_observed_date": sku_record["last_observed_date"].date().isoformat(),
                "history_days": int(demand.size),
                "nonzero_days": int(np.count_nonzero(demand)),
                "zero_demand_days": int(demand.size - np.count_nonzero(demand)),
                "total_demand": float(demand.sum()),
                "lead_time_days": LEAD_TIME_DAYS,
                "service_level": TARGET_SERVICE_LEVEL,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "resampling_method": RESAMPLING_METHOD,
                "random_seed": RANDOM_SEED + sku_index,
                "mean_lead_time_demand": inventory_result["mean_lead_time_demand"],
                "reorder_point": inventory_result["reorder_point"],
                "safety_stock": inventory_result["safety_stock"],
                "minimum_inventory": inventory_result["minimum_inventory"],
            }
        )

        evaluation_rng = np.random.default_rng(RANDOM_SEED + 100_000 + sku_index)
        evaluation_result = evaluate_holdout(
            demand=demand,
            lead_time_days=LEAD_TIME_DAYS,
            service_level=TARGET_SERVICE_LEVEL,
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
            resampling_method=RESAMPLING_METHOD,
            rng=evaluation_rng,
            test_fraction=TEST_FRACTION,
        )
        evaluation_rows.append(
            {
                "item_code": sku,
                "product_name": sku_record["product_name"],
                "history_days": int(demand.size),
                "lead_time_days": LEAD_TIME_DAYS,
                "service_level": TARGET_SERVICE_LEVEL,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "resampling_method": RESAMPLING_METHOD,
                **evaluation_result,
            }
        )

    minimum_inventory_df = pd.DataFrame(minimum_inventory_rows).sort_values("item_code")
    evaluation_df = pd.DataFrame(evaluation_rows).sort_values("item_code")
    return minimum_inventory_df, evaluation_df


def main():
    preprocessing_dir = find_preprocessing_dir()
    source_path = find_order_data_path(preprocessing_dir)
    df_order, columns = load_order_data(source_path)
    series_by_sku = build_daily_demand_series(df_order, columns)
    minimum_inventory_df, evaluation_df = build_inventory_outputs(series_by_sku)

    minimum_inventory_path = save_dataframe(
        minimum_inventory_df,
        MINIMUM_INVENTORY_OUTPUT_PATH,
    )
    evaluation_path = save_dataframe(
        evaluation_df,
        EVALUATION_OUTPUT_PATH,
    )

    print("Minimum inventory methodology:")
    print("- Daily demand is aggregated by SKU and calendar date.")
    print("- Zero-demand days are filled only between each SKU's first and last observed date.")
    print(
        f"- Lead time = {LEAD_TIME_DAYS} day(s), service level = {TARGET_SERVICE_LEVEL:.2%}, "
        f"resampling = {RESAMPLING_METHOD}, bootstrap repetitions = {BOOTSTRAP_REPETITIONS}."
    )
    print("- minimum_inventory is mapped directly to the reorder point (ROP).")
    print(f"Source file: {make_console_safe(source_path)}")
    print(f"Minimum inventory output saved to: {make_console_safe(minimum_inventory_path)}")
    print(f"Evaluation output saved to: {make_console_safe(evaluation_path)}")
    print(f"SKUs processed: {len(minimum_inventory_df)}")


if __name__ == "__main__":
    main()