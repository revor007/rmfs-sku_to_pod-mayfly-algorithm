from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


BASE_DIR = Path(__file__).resolve().parent


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


PREPROCESSING_DIR = find_preprocessing_dir()

JACCARD_OUTPUT_PATH = BASE_DIR / "jaccard_similarity_matrix.csv"
LEGACY_OUTPUT_PATH = BASE_DIR / "association_matrix_normalized.csv"
TOP_PAIRS_OUTPUT_PATH = BASE_DIR / "top_jaccard_pairs.csv"
JACCARD_BUCKETS_OUTPUT_PATH = BASE_DIR / "jaccard_similarity_buckets.csv"
JACCARD_VISUALIZATION_PATH = BASE_DIR / "jaccard_similarity_distribution.png"

ORDER_ID_CANDIDATES = [
    "\u8ba2\u5355\u53f7",
    "order_id",
]
SKU_CANDIDATES = [
    "\u5546\u54c1\u7f16\u7801",
    "item_code",
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


def find_column(columns, candidates):
    normalized = {str(col).strip(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def build_jaccard_matrix(df_order, order_col, sku_col):
    pairs = df_order[[order_col, sku_col]].dropna().copy()
    pairs[order_col] = pairs[order_col].astype(str).str.strip()
    pairs[sku_col] = pairs[sku_col].astype(str).str.strip()
    pairs = pairs[(pairs[order_col] != "") & (pairs[sku_col] != "")]
    pairs = pairs.drop_duplicates(subset=[order_col, sku_col])

    if pairs.empty:
        raise ValueError("No valid order-SKU pairs remained after preprocessing.")

    order_codes = pairs[order_col].astype("category").cat.codes
    sku_cat = pairs[sku_col].astype("category")
    sku_codes = sku_cat.cat.codes

    n_orders = order_codes.nunique()
    n_skus = sku_codes.nunique()

    incidence_sparse = coo_matrix(
        (np.ones(len(pairs), dtype=np.int32), (order_codes, sku_codes)),
        shape=(n_orders, n_skus),
        dtype=np.int32,
    ).tocsr()

    co_occurrence = (incidence_sparse.T @ incidence_sparse).astype(np.int32).toarray()
    supports = np.diag(co_occurrence).astype(np.float32)

    denominators = supports[:, None] + supports[None, :] - co_occurrence
    jaccard = np.divide(
        co_occurrence,
        denominators,
        out=np.zeros_like(denominators, dtype=np.float32),
        where=denominators > 0,
    )
    np.fill_diagonal(jaccard, 0.0)

    sku_labels = sku_cat.cat.categories.astype(str)

    co_occurrence_matrix = pd.DataFrame(
        co_occurrence,
        index=sku_labels,
        columns=sku_labels,
    ).round(4)
    jaccard_matrix = pd.DataFrame(
        jaccard,
        index=sku_labels,
        columns=sku_labels,
    ).round(4)

    return pairs, co_occurrence_matrix, jaccard_matrix


def build_top_pairs(co_occurrence_matrix, jaccard_matrix, top_n=20):
    mask = np.triu(np.ones(jaccard_matrix.shape, dtype=bool), k=1)

    co_occurrence_pairs = (
        co_occurrence_matrix.where(mask)
        .stack()
        .reset_index(name="co_occurrence")
    )
    jaccard_pairs = (
        jaccard_matrix.where(mask)
        .stack()
        .reset_index(name="jaccard_similarity")
    )

    top_pairs = co_occurrence_pairs.merge(
        jaccard_pairs,
        on=["level_0", "level_1"],
        how="inner",
    )
    top_pairs = top_pairs.rename(
        columns={"level_0": "product_a", "level_1": "product_b"}
    )

    return top_pairs.sort_values(
        ["jaccard_similarity", "co_occurrence"],
        ascending=[False, False],
    ).head(top_n)


def summarize_jaccard_buckets(jaccard_matrix):
    mask = np.triu(np.ones(jaccard_matrix.shape, dtype=bool), k=1)
    pair_values = jaccard_matrix.where(mask).stack().to_numpy(dtype=float)

    bucket_summary = pd.DataFrame(
        {
            "bucket": ["Below 0.1", "0.1 to 0.5", "Above 0.5"],
            "pair_count": [
                int((pair_values < 0.1).sum()),
                int(((pair_values >= 0.1) & (pair_values <= 0.5)).sum()),
                int((pair_values > 0.5).sum()),
            ],
        }
    )
    bucket_summary["percentage"] = (
        bucket_summary["pair_count"] / max(len(pair_values), 1) * 100
    ).round(4)

    return bucket_summary


def plot_jaccard_buckets(bucket_summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c9d6df", "#f4b183", "#c55a11"]

    bars = ax.bar(
        bucket_summary["bucket"],
        bucket_summary["pair_count"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Distribution of Jaccard Similarity Across SKU Pairs")
    ax.set_xlabel("Jaccard similarity range")
    ax.set_ylabel("Number of unique SKU pairs")
    ax.grid(axis="y", alpha=0.3)

    for bar, percentage in zip(bars, bucket_summary["percentage"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = get_writable_output_path(JACCARD_VISUALIZATION_PATH)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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


def main():
    path_order = find_order_data_path()
    df_order = pd.read_csv(path_order, sep=";", encoding="utf-8-sig", decimal=",")

    order_col = find_column(df_order.columns, ORDER_ID_CANDIDATES)
    sku_col = find_column(df_order.columns, SKU_CANDIDATES)

    pairs, co_occurrence_matrix, jaccard_matrix = build_jaccard_matrix(
        df_order=df_order,
        order_col=order_col,
        sku_col=sku_col,
    )
    top_pairs = build_top_pairs(co_occurrence_matrix, jaccard_matrix)
    bucket_summary = summarize_jaccard_buckets(jaccard_matrix)

    jaccard_output_path = save_dataframe(
        jaccard_matrix,
        JACCARD_OUTPUT_PATH,
        encoding="utf-8-sig",
        sep=";",
        float_format="%.4f",
    )
    # Keep the previous filename as a compatibility export for downstream steps.
    legacy_output_path = save_dataframe(
        jaccard_matrix,
        LEGACY_OUTPUT_PATH,
        encoding="utf-8-sig",
        sep=";",
        float_format="%.4f",
    )
    top_pairs_output_path = save_dataframe(
        top_pairs,
        TOP_PAIRS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        float_format="%.4f",
    )
    bucket_output_path = save_dataframe(
        bucket_summary,
        JACCARD_BUCKETS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        float_format="%.4f",
    )
    visualization_output_path = plot_jaccard_buckets(bucket_summary)

    safe_path_order = str(path_order).encode("unicode_escape").decode("ascii")
    print(f"Order data: {safe_path_order}")
    print(f"Unique order-SKU pairs: {len(pairs):,}")
    print(f"Number of orders: {pairs[order_col].nunique():,}")
    print(f"Number of SKUs: {pairs[sku_col].nunique():,}")
    print(f"Jaccard matrix saved to: {jaccard_output_path}")
    print(f"Compatibility export saved to: {legacy_output_path}")
    print(f"Top pairs saved to: {top_pairs_output_path}")
    print(f"Bucket summary saved to: {bucket_output_path}")
    print(f"Visualization saved to: {visualization_output_path}")
    print(top_pairs.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(bucket_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    main()
