# Library
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
print('Running mixed_clustering.py')

# Static-only clustering for products that need a fallback grouping.
# This keeps new products and old products with no association in the same pipeline
# instead of relying on a dynamic time-series stage.
path_file_filter = ['D:/ITB/Thesis/Preprocessing/preprocessed_final.csv']

# try to load association matrix to detect items with zero association
association_path = os.path.join("D:/ITB/Thesis/fcgma/rmfs-sku_to_pod-mayfly-algorithm", "association_matrix.csv")
order_path = "D:/ITB/Thesis/Preprocessing/訂單資料_final.csv"
zero_assoc_set = set()
low_order_set = set()
try:
    if os.path.exists(association_path):
        assoc_df = pd.read_csv(association_path, sep=';', index_col=0, dtype=str)
        assoc_numeric = assoc_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        zero_assoc_idx = assoc_numeric.index[assoc_numeric.sum(axis=1) == 0].astype(str)
        zero_assoc_set = set(zero_assoc_idx)
    else:
        print(f'Association matrix not found at {association_path}; continuing without zero-association detection')
except Exception as e:
    print('Failed to load association matrix:', e)
    zero_assoc_set = set()

try:
    if os.path.exists(order_path):
        df_orders = pd.read_csv(order_path, sep=';', encoding='utf-8-sig', dtype=str)
        if '商品编码' not in df_orders.columns:
            print(f"Order file missing '商品编码' column: {order_path}")
        else:
            order_counts = df_orders['商品编码'].astype(str).str.strip().value_counts()
            low_order_set = set(order_counts[order_counts <= 1].index)
    else:
        print(f'Order history not found at {order_path}; continuing without order-based new detection')
except Exception as e:
    print('Failed to load order history:', e)
    low_order_set = set()

new_assoc_set = zero_assoc_set | low_order_set

# iterative KMeans parameters
THRESHOLD_NEW_CLUSTER_SIZE = 2
MAX_KMEANS_RETRIES = 10
MAX_NEW_CLUSTER_SPLITS = 20

# Function to calculate within-cluster SSE for elbow method
def within_cluster_sse(X, labels):
    sse = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0, keepdims=True)
        sse += ((Xc - mu) ** 2).sum()
    return sse


def split_new_clusters(labels, X, is_new_mask, max_size, max_iter=20):
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for _ in range(max_iter):
        changed = False
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            if len(idx) <= max_size:
                continue
            if not is_new_mask[idx].any():
                continue
            if X.iloc[idx].drop_duplicates().shape[0] < 2:
                continue
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            sub_labels = km.fit_predict(X.iloc[idx])
            if len(np.unique(sub_labels)) < 2:
                continue
            labels[idx[sub_labels == 1]] = next_label
            next_label += 1
            changed = True
        if not changed:
            break
    return labels

def assign_pair_ids(df_result, features_df):
    pair_id = pd.Series(index=df_result.index, dtype=object)
    for cluster_id, idx in df_result.groupby('cluster_global').groups.items():
        cluster = df_result.loc[idx]
        new_idx = cluster[cluster['is_new_association']].index
        old_idx = cluster[~cluster['is_new_association']].index
        if len(new_idx) == 0 or len(old_idx) == 0:
            continue

        new_idx = cluster.loc[new_idx].sort_values('item_code').index
        old_idx_list = list(old_idx)
        old_pos = {row_idx: pos for pos, row_idx in enumerate(old_idx_list)}

        new_feats = features_df.loc[new_idx].to_numpy()
        old_feats = features_df.loc[old_idx_list].to_numpy()
        dists = ((new_feats[:, None, :] - old_feats[None, :, :]) ** 2).sum(axis=2)

        available = set(old_idx_list)
        pair_counter = 1
        for i, new_row in enumerate(new_idx):
            if not available:
                break
            avail_pos = [old_pos[row_idx] for row_idx in available]
            best_pos = min(avail_pos, key=lambda p: (dists[i, p], p))
            old_row = old_idx_list[best_pos]
            new_code = str(df_result.at[new_row, 'item_code'])
            old_code = str(df_result.at[old_row, 'item_code'])
            pid = f"{new_code}__{old_code}"
            pair_id.at[new_row] = pid
            pair_id.at[old_row] = pid
            pair_counter += 1
            available.remove(old_row)

    # Fallback: pair unresolved new items with the closest item in the same cluster.
    for cluster_id, idx in df_result.groupby('cluster_global').groups.items():
        cluster = df_result.loc[idx]
        unresolved = cluster.index[cluster['is_new_association'] & pair_id.loc[cluster.index].isna()]
        if len(unresolved) == 0:
            continue
        cluster_idx = list(cluster.index)
        cluster_feats = features_df.loc[cluster_idx].to_numpy()
        pos_map = {row_idx: pos for pos, row_idx in enumerate(cluster_idx)}
        for new_row in unresolved:
            if pd.notna(pair_id.at[new_row]):
                continue
            candidates = [row_idx for row_idx in cluster_idx if row_idx != new_row]
            if not candidates:
                continue
            new_pos = pos_map[new_row]
            cand_pos = [pos_map[row_idx] for row_idx in candidates]
            diffs = cluster_feats[cand_pos] - cluster_feats[new_pos]
            dists = (diffs ** 2).sum(axis=1)
            best_pos = cand_pos[int(np.argmin(dists))]
            best_row = cluster_idx[best_pos]
            new_code = str(df_result.at[new_row, 'item_code'])
            best_code = str(df_result.at[best_row, 'item_code'])
            if pd.notna(pair_id.at[best_row]):
                pid = pair_id.at[best_row]
            else:
                pid = f"{new_code}__{best_code}"
                pair_id.at[best_row] = pid
            pair_id.at[new_row] = pid

    return pair_id

merged_results = []
global_cluster_offset = 0

for path in path_file_filter:
    df = pd.read_csv(path, sep=';', encoding='utf-8', decimal=',')
    print(f'Loaded {path} -> rows={len(df)} cols={list(df.columns)[:10]}')
    if df.empty:
        print(f"Skip {path}: empty input file.")
        continue

    if 'item_code' not in df.columns:
        print(f"Skip {path}: missing required column 'item_code'.")
        continue

    df['product_type'] = df['item_code'].astype(str).str[:4]
    for product_type, df_group in df.groupby('product_type'):
        df_encoded = df_group.copy()
        print(f"Processing product_type={product_type} rows={len(df_encoded)}")

        # Normalization of static features.
        price_columns = ['price_per_piece', 'original_price_per_piece']
        for price in price_columns:
            df_encoded[price] = pd.to_numeric(df_encoded[price], errors='coerce')
            df_encoded[f'log_{price}'] = np.log1p(df_encoded[price].clip(lower=0))

        cols_to_normalize = [
            'estimation_discount',
            'log_price_per_piece',
            'log_original_price_per_piece',
            'shelf_life',
            'capacity'
        ]

        required_cols = price_columns + cols_to_normalize + ['title', 'category', 'capacity_unit', 'item_code']
        missing_cols = [col for col in required_cols if col not in df_encoded.columns]
        if missing_cols:
            print(f"Skip product_type={product_type}: missing required columns {missing_cols}")
            continue

        for col in cols_to_normalize:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        df_encoded = df_encoded.dropna(subset=cols_to_normalize).copy()

        if len(df_encoded) < 2:
            print(f"Skip product_type={product_type}: not enough rows after preprocessing.")
            continue

        scaler = StandardScaler()
        for col in cols_to_normalize:
            df_encoded[f'{col}_normalized'] = scaler.fit_transform(df_encoded[[col]]).ravel()

        # K-means clustering on static product attributes only.
        features_final = df_encoded.filter(like='_normalized').columns
        X_static = df_encoded[features_final].astype(float)

        # Ensure valid k range for each group (cap by unique points to avoid KMeans warnings).
        unique_points = X_static.drop_duplicates().shape[0]
        k_limit = min(len(X_static) - 1, unique_points)
        if k_limit < 2:
            print(
                f"Skip product_type={product_type}: not enough unique samples for K-means "
                f"(unique_points={unique_points})."
            )
            continue

        # initial elbow selection (use up to 9 clusters or k_limit)
        k_max = min(9, k_limit)
        k_values = list(range(2, k_max + 1))
        elbow_rows = []
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            km.fit(X_static)
            elbow_rows.append({'k': k, 'sse': km.inertia_})

        df_elbow_static = pd.DataFrame(elbow_rows)
        x = df_elbow_static['k'].to_numpy(dtype=float)
        y = df_elbow_static['sse'].to_numpy(dtype=float)

        x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

        p1 = np.array([x_n[0], y_n[0]])
        p2 = np.array([x_n[-1], y_n[-1]])

        num = np.abs((p2[1] - p1[1]) * x_n - (p2[0] - p1[0]) * y_n + p2[0] * p1[1] - p2[1] * p1[0])
        den = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) + 1e-12
        dist_to_line = num / den

        best_idx = np.argmax(dist_to_line)
        best_k_static = int(x[best_idx])

        # Iteratively re-run KMeans increasing k until all new products are
        # in clusters of size <= THRESHOLD_NEW_CLUSTER_SIZE, or until retries exhausted.
        k = best_k_static
        labels_static = None
        for attempt in range(MAX_KMEANS_RETRIES):
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = km.fit_predict(X_static)

            # quick check: assign clusters and compute sizes
            temp = df_encoded[['title', 'category', 'capacity_unit', 'item_code']].copy()
            temp['cluster_local'] = labels
            temp['cluster_global'] = temp['cluster_local'] + global_cluster_offset
            temp['mixed_cluster'] = temp['cluster_global']
            sizes = temp['cluster_global'].value_counts()
            temp['cluster_size'] = temp['cluster_global'].map(sizes)
            temp['product_type'] = product_type
            temp['is_new_association'] = temp['item_code'].astype(str).isin(new_assoc_set)

            problematic = temp[(temp['is_new_association']) & (temp['cluster_size'] > THRESHOLD_NEW_CLUSTER_SIZE)]
            if problematic.empty:
                labels_static = labels
                best_k_static = k
                if attempt > 0:
                    print(f"Converged for product_type={product_type} with k={k} after {attempt+1} attempts")
                break

            # otherwise increase k if possible and retry
            if k < k_limit:
                k += 1
            else:
                print(f"Reached k limit for product_type={product_type}; stopping retries (k={k}).")
                labels_static = labels
                best_k_static = k
                break

        if labels_static is None:
            # fallback: use last labels
            labels_static = labels

        is_new_mask = df_encoded['item_code'].astype(str).isin(new_assoc_set).to_numpy()
        labels_static = split_new_clusters(
            labels_static,
            X_static,
            is_new_mask,
            THRESHOLD_NEW_CLUSTER_SIZE,
            max_iter=MAX_NEW_CLUSTER_SPLITS
        )

        file_result = df_encoded[['title', 'category', 'capacity_unit', 'item_code']].copy()
        file_result['cluster_local'] = labels_static
        file_result['cluster_global'] = file_result['cluster_local'] + global_cluster_offset

        # Mixed cluster id is global across groups.
        file_result['mixed_cluster'] = file_result['cluster_global']

        cluster_sizes_best = file_result['cluster_global'].value_counts()
        file_result['cluster_size'] = file_result['cluster_global'].map(cluster_sizes_best)
        # product type (first 4 digits of item_code)
        file_result['product_type'] = product_type
        # flag items with zero association (new products)
        file_result['is_new_association'] = file_result['item_code'].astype(str).isin(new_assoc_set)
        # pair each new product with nearest historical product inside the same cluster
        file_result['pair_id'] = assign_pair_ids(file_result, X_static)
        # default allocation type: common if cluster has multiple items, otherwise new/unassociated
        file_result['allocation_type'] = np.where(
            file_result['cluster_size'] > 1,
            'common allocation',
            'new/unassociated allocation'
        )
        # override: if new product (zero association) and singleton cluster, allocate randomly first
        mask_random = (file_result['is_new_association']) & (file_result['cluster_size'] == 1)
        file_result.loc[mask_random, 'allocation_type'] = 'random allocation'
        merged_results.append(file_result)

        group_cluster_count = int(np.max(labels_static)) + 1
        global_cluster_offset += group_cluster_count
        print(
            f"Processed product_type={product_type}: best_k_static={best_k_static}, "
            f"next_global_offset={global_cluster_offset}"
        )

if merged_results:
    df_mixed_merged = pd.concat(merged_results, ignore_index=True)
    output_path = "D:/ITB/Thesis/fcgma/rmfs-sku_to_pod-mayfly-algorithm/Clustering/mixed_cluster_results.csv"
    df_mixed_merged.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    print(f"Merged mixed cluster result saved to: {output_path}")
    print(f"Total rows: {len(df_mixed_merged)}, total global clusters: {df_mixed_merged['cluster_global'].nunique()}")
    # summarize random allocations (new unassociated singletons allocated randomly)
    try:
        rand_summary = df_mixed_merged[df_mixed_merged['allocation_type'] == 'random allocation']
        if not rand_summary.empty:
            rand_counts = rand_summary.groupby('product_type').size().reset_index(name='random_alloc_count')
            rand_out = "D:/ITB/Thesis/fcgma/rmfs-sku_to_pod-mayfly-algorithm/Clustering/random_allocation_summary.csv"
            rand_counts.to_csv(rand_out, index=False, encoding='utf-8-sig', sep=';', decimal=',')
            print(f'Random allocation summary saved to: {rand_out}')
        else:
            print('No random allocations were made.')
    except Exception as e:
        print('Failed to write random allocation summary:', e)
else:
    print('No file produced a valid mixed clustering result.')