from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PREPROCESSING_DIR = ROOT_DIR.parent / "Preprocessing"
OUTPUT_PATH = BASE_DIR / "time-series-clustering-results.csv"

SKU_CANDIDATES = ["\u5546\u54c1\u7f16\u7801", "item_code"]
QUANTITY_CANDIDATES = ["\u5546\u54c1\u6570\u91cf", "quantity"]
CREATED_TIME_CANDIDATES = ["\u521b\u5efa\u65f6\u95f4", "created_at"]


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


def wcsse_score(values, labels):
    total = 0.0
    for cluster_id in np.unique(labels):
        cluster_values = values[labels == cluster_id]
        if cluster_values.size == 0:
            continue
        centroid = cluster_values.mean(axis=0)
        total += ((cluster_values - centroid) ** 2).sum()
    return float(total)


def select_best_cluster_count(values):
    max_candidate = min(59, len(values) - 1)
    if max_candidate < 2:
        return 1

    linkage_matrix = linkage(values, method="ward")
    diagnostics = []
    for k in range(2, max_candidate + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        actual_clusters = len(np.unique(labels))
        if actual_clusters < 2 or actual_clusters >= len(values):
            continue

        diagnostics.append(
            {
                "k": actual_clusters,
                "wcsse": wcsse_score(values, labels),
                "silhouette": silhouette_score(values, labels, metric="euclidean"),
            }
        )

    if not diagnostics:
        return 1

    diagnostics_df = pd.DataFrame(diagnostics).drop_duplicates(subset=["k"]).sort_values("k")
    x = diagnostics_df["k"].to_numpy(dtype=float)
    y = diagnostics_df["wcsse"].to_numpy(dtype=float)

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    numerator = np.abs(
        (p2[1] - p1[1]) * x_norm
        - (p2[0] - p1[0]) * y_norm
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    )
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) + 1e-12
    distance_to_line = numerator / denominator
    best_idx = int(np.argmax(distance_to_line))
    return int(x[best_idx])


def main():
    order_path = find_order_data_path()
    df_order = pd.read_csv(order_path, sep=";", encoding="utf-8-sig", decimal=",")

    sku_col = find_column(df_order.columns, SKU_CANDIDATES)
    quantity_col = find_column(df_order.columns, QUANTITY_CANDIDATES)
    created_col = find_column(df_order.columns, CREATED_TIME_CANDIDATES)

    df_cluster = df_order[[sku_col, quantity_col, created_col]].copy()
    df_cluster[sku_col] = df_cluster[sku_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df_cluster[quantity_col] = pd.to_numeric(df_cluster[quantity_col], errors="coerce").fillna(0.0)
    df_cluster[created_col] = pd.to_datetime(
        df_cluster[created_col],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df_cluster = df_cluster.dropna(subset=[created_col]).copy()
    df_cluster["day"] = df_cluster[created_col].dt.normalize()

    daily = (
        df_cluster.groupby([sku_col, "day"], as_index=False)[quantity_col]
        .sum()
        .rename(columns={quantity_col: "daily_demand"})
    )

    pivot = daily.pivot_table(index=sku_col, columns="day", values="daily_demand", fill_value=0.0)
    values = np.log1p(pivot.to_numpy(dtype=np.float64))
    best_k = select_best_cluster_count(values)

    if best_k <= 1:
        labels = np.ones(len(pivot), dtype=int)
    else:
        linkage_matrix = linkage(values, method="ward")
        labels = fcluster(linkage_matrix, t=best_k, criterion="maxclust")

    cluster_labels = pd.DataFrame(
        {
            "item_code": pivot.index.astype(str),
            "cluster": labels.astype(int),
        }
    ).sort_values("item_code")

    cluster_labels.to_csv(
        OUTPUT_PATH,
        index=False,
        header=True,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )

    print(
        f"Time-series clustering saved to: "
        f"{str(OUTPUT_PATH).encode('unicode_escape').decode('ascii')}"
    )
    print(f"Order source: {str(order_path).encode('unicode_escape').decode('ascii')}")
    print(f"SKUs clustered: {len(cluster_labels):,}")
    print(f"Selected number of clusters: {best_k}")


if __name__ == "__main__":
    main()
