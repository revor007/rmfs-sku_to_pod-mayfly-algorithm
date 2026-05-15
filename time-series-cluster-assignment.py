from pathlib import Path

import numpy as np
import pandas as pd

from sku_alignment import load_aligned_sku_set, normalize_item_code


BASE_DIR = Path(__file__).resolve().parent
TIME_SERIES_CLUSTER_INPUT_PATH = BASE_DIR / "Clustering" / "time-series-clustering-results.csv"
SAME_CLUSTER_MATRIX_OUTPUT_PATH = BASE_DIR / "same_cluster_matrix.csv"
SKU_COLUMN_CANDIDATES = ["item_code", "\u5546\u54c1\u7f16\u7801"]


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


def main():
    df_ts = pd.read_csv(
        TIME_SERIES_CLUSTER_INPUT_PATH,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
    )
    sku_col = find_column(df_ts.columns, SKU_COLUMN_CANDIDATES)
    if "cluster" not in df_ts.columns:
        raise KeyError("Column 'cluster' was not found in time-series clustering output.")

    cluster_assignments = (
        df_ts[[sku_col, "cluster"]]
        .copy()
        .rename(columns={sku_col: "item_code"})
    )
    cluster_assignments["item_code"] = cluster_assignments["item_code"].map(normalize_item_code)
    aligned_skus = load_aligned_sku_set(BASE_DIR)
    cluster_assignments = cluster_assignments[cluster_assignments["item_code"].isin(aligned_skus)].copy()
    cluster_assignments = cluster_assignments.drop_duplicates(subset=["item_code"]).sort_values("item_code")

    labels = cluster_assignments["cluster"].to_numpy()
    same_cluster = (labels[:, None] == labels[None, :]).astype(np.int8)
    np.fill_diagonal(same_cluster, 0)

    same_cluster_matrix = pd.DataFrame(
        same_cluster,
        index=cluster_assignments["item_code"],
        columns=cluster_assignments["item_code"],
    )
    same_cluster_matrix.to_csv(
        SAME_CLUSTER_MATRIX_OUTPUT_PATH,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )

    print(
        f"Same-cluster matrix saved to: "
        f"{str(SAME_CLUSTER_MATRIX_OUTPUT_PATH).encode('unicode_escape').decode('ascii')}"
    )
    print(f"Aligned SKU universe: {len(aligned_skus):,}")
    print(f"SKUs covered: {len(cluster_assignments):,}")


if __name__ == "__main__":
    main()
