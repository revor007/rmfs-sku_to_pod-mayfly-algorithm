from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PREPROCESSING_DIR = ROOT_DIR.parent / "Preprocessing"
DATASET_OUTPUT_DIR = BASE_DIR / "dataset"
RESULTS_OUTPUT_PATH = BASE_DIR / "bc-k-means-results.csv"

ORDER_SKU_CANDIDATES = ["\u5546\u54c1\u7f16\u7801", "item_code"]
MAX_CLUSTER_SIZE = 2


def find_product_data_path():
    path = PREPROCESSING_DIR / "preprocessed_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {path}")
    return path


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


def normalize_item_code(series):
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def build_feature_frame(df_product):
    df = df_product.copy()

    price_columns = ["price_per_piece", "original_price_per_piece"]
    for price_col in price_columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df[f"log_{price_col}"] = np.log1p(df[price_col].clip(lower=0))

    numeric_columns = [
        "estimation_discount",
        "log_price_per_piece",
        "log_original_price_per_piece",
        "shelf_life",
        "capacity",
        "Promo",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    scaler = StandardScaler()
    normalized_cols = []
    for col in numeric_columns:
        normalized_col = f"{col}_normalized"
        df[normalized_col] = scaler.fit_transform(df[[col]])
        normalized_cols.append(normalized_col)

    return df, normalized_cols


def pick_nearest_historical(df_cluster, new_product_code, feature_cols):
    candidate_hist = df_cluster[df_cluster["is_new_product"] == 0].copy()
    if candidate_hist.empty:
        return None

    new_row = df_cluster[df_cluster["item_code"] == new_product_code]
    x_new = new_row[feature_cols].to_numpy(dtype=float)
    x_hist = candidate_hist[feature_cols].to_numpy(dtype=float)
    distances = cdist(x_new, x_hist, metric="euclidean").flatten()
    return candidate_hist.iloc[int(distances.argmin())]["item_code"]


def run_bc_kmeans_for_product(df_product, new_product_code):
    current_df = df_product.copy()
    nearest_hist_code = None

    while True:
        if len(current_df) <= MAX_CLUSTER_SIZE:
            final_cluster = current_df.copy()
            break

        df_encoded, feature_cols = build_feature_frame(current_df)
        if df_encoded.empty:
            final_cluster = current_df.copy()
            break

        n_clusters = len(df_encoded) // 2 if len(df_encoded) % 2 == 0 else (len(df_encoded) + 1) // 2
        feature_values = df_encoded[feature_cols].astype(float)
        n_unique_points = np.unique(feature_values.to_numpy(), axis=0).shape[0]

        if n_unique_points < n_clusters:
            final_cluster = df_encoded.copy()
            nearest_hist_code = pick_nearest_historical(final_cluster, new_product_code, feature_cols)
            break

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_encoded["cluster"] = model.fit_predict(feature_values)
        new_product_cluster = df_encoded.loc[
            df_encoded["item_code"] == new_product_code,
            "cluster",
        ].iloc[0]
        problem_cluster = df_encoded[df_encoded["cluster"] == new_product_cluster].copy()

        if len(problem_cluster) <= MAX_CLUSTER_SIZE:
            final_cluster = problem_cluster.copy()
            break

        if len(problem_cluster) >= len(current_df):
            final_cluster = problem_cluster.copy()
            nearest_hist_code = pick_nearest_historical(problem_cluster, new_product_code, feature_cols)
            break

        current_df = problem_cluster.drop(columns=["cluster"]).copy()

    historical_products = final_cluster[final_cluster["is_new_product"] == 0]
    if len(final_cluster) < 2:
        allocation_type = "random_allocation"
        corresponding_historical_product = None
    elif len(final_cluster) == 2 and len(historical_products) == 1:
        allocation_type = "common_allocation"
        corresponding_historical_product = historical_products["item_code"].iloc[0]
    elif nearest_hist_code is not None:
        allocation_type = "common_allocation"
        corresponding_historical_product = nearest_hist_code
    else:
        allocation_type = "problematic_allocation"
        corresponding_historical_product = None

    return {
        "new_product_code": new_product_code,
        "allocation_type": allocation_type,
        "corresponding_historical_product": corresponding_historical_product,
    }


def main():
    product_path = find_product_data_path()
    order_path = find_order_data_path()

    df_product = pd.read_csv(product_path, sep=";", encoding="utf-8-sig", decimal=",")
    df_order = pd.read_csv(order_path, sep=";", encoding="utf-8-sig", decimal=",")

    sku_col = find_column(df_order.columns, ORDER_SKU_CANDIDATES)
    df_product["item_code"] = normalize_item_code(df_product["item_code"])
    df_order[sku_col] = normalize_item_code(df_order[sku_col])

    product_counts = df_order[sku_col].value_counts()
    new_products = set(product_counts[product_counts <= 1].index)
    df_product["is_new_product"] = df_product["item_code"].isin(new_products).astype(int)

    new_product_codes = sorted(
        set(df_product.loc[df_product["is_new_product"] == 1, "item_code"])
    )
    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for item_code in new_product_codes:
        new_product_row = df_product[df_product["item_code"] == item_code]
        if new_product_row.empty:
            continue

        product_category = new_product_row["category"].iloc[0]
        product_capacity = new_product_row["capacity_category"].iloc[0]
        selected_new_product = df_product[df_product["item_code"] == item_code]
        same_historical_product = df_product[
            (df_product["is_new_product"] == 0)
            & (df_product["category"] == product_category)
            & (df_product["capacity_category"] == product_capacity)
        ]
        dataset_df = pd.concat(
            [selected_new_product, same_historical_product],
            ignore_index=True,
        )
        dataset_df.to_csv(
            DATASET_OUTPUT_DIR / f"{item_code}_dataset.csv",
            index=False,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
        )
        results.append(run_bc_kmeans_for_product(dataset_df, item_code))

    results_df = pd.DataFrame(results).sort_values("new_product_code")
    results_df.to_csv(
        RESULTS_OUTPUT_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )

    print(
        f"BC-K-means results saved to: "
        f"{str(RESULTS_OUTPUT_PATH).encode('unicode_escape').decode('ascii')}"
    )
    print(f"New SKUs processed: {len(results_df):,}")


if __name__ == "__main__":
    main()
