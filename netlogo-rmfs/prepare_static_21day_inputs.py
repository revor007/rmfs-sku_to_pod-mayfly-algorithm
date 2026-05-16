from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


POD_THRESHOLD = 0.4
GLOBAL_THRESHOLD = 0.4
POD_TYPE = 4
SLOT_TYPE = 3
DEFAULT_SLOT_CAPACITY = 40


def normalize_item_code(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    text = text.replace(",", ".")
    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if np.isfinite(numeric_value):
        return str(int(numeric_value))
    return text


def find_column(columns, candidates):
    normalized = {str(col).replace("\ufeff", "").strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = candidate.lower()
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    raise KeyError(f"Could not find any of {candidates} in columns {list(columns)}")


def load_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", engine="python")


def find_existing_directory(candidates: list[Path], required_files: list[str]) -> Path:
    for candidate in candidates:
        if candidate.exists() and all((candidate / filename).exists() for filename in required_files):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    required = ", ".join(required_files)
    raise FileNotFoundError(
        f"Could not locate a directory containing {required}. Searched: {searched}"
    )


def assign_abc_classes(order_frequency: pd.Series) -> pd.Series:
    if order_frequency.empty or float(order_frequency.sum()) <= 0:
        return pd.Series("C", index=order_frequency.index, dtype=object)

    ranked = order_frequency.sort_values(ascending=False, kind="stable")
    share = ranked / ranked.sum()
    cumulative_share = share.cumsum()

    classes = pd.Series(index=ranked.index, dtype=object)
    classes.loc[cumulative_share <= 0.80] = "A"
    classes.loc[(cumulative_share > 0.80) & (cumulative_share <= 0.95)] = "B"
    classes.loc[cumulative_share > 0.95] = "C"

    if not classes.empty and classes.isna().all():
        classes.iloc[0] = "A"

    return classes.reindex(order_frequency.index).fillna("C")


def count_physical_pods(generated_pod_path: Path) -> int:
    layout = pd.read_csv(generated_pod_path, header=None)
    return int((layout == 1).sum().sum())


def backup_live_files(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / "backups" / f"static_21day_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("items.csv", "pods.csv"):
        source = output_dir / filename
        if source.exists():
            shutil.copy2(source, backup_dir / filename)

    return backup_dir


def prepare_inputs(
    scenario_root: Path,
    allocation_path: Path,
    order_path: Path,
    metadata_path: Path,
    translated_info_path: Path,
    max_comp_path: Path,
) -> dict:
    output_dir = scenario_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    orders = load_semicolon_csv(order_path)
    order_id_col = find_column(orders.columns, ["订单号", "order_id"])
    item_code_col = find_column(orders.columns, ["商品编码", "item_code"])
    quantity_col = find_column(orders.columns, ["商品数量", "quantity", "qty"])

    orders = orders[[order_id_col, item_code_col, quantity_col]].copy()
    orders.columns = ["order_id", "item_code", "quantity"]
    orders["item_code"] = orders["item_code"].map(normalize_item_code)
    orders["quantity"] = (
        pd.to_numeric(orders["quantity"], errors="coerce").fillna(0).round().astype(np.int32)
    )
    orders = orders[orders["item_code"] != ""].copy()

    metadata = load_semicolon_csv(metadata_path)
    metadata_code_col = find_column(metadata.columns, ["item_code"])
    metadata_title_col = find_column(metadata.columns, ["title"])
    metadata = metadata[[metadata_code_col, metadata_title_col]].copy()
    metadata.columns = ["item_code", "item_name_meta"]
    metadata["item_code"] = metadata["item_code"].map(normalize_item_code)

    translated = load_semicolon_csv(translated_info_path)
    translated_code_col = find_column(translated.columns, ["item code"])
    translated_name_col = find_column(translated.columns, ["notes_en", "notes"])
    translated_length_col = find_column(translated.columns, ["length carton", "(箱)長"])
    translated_width_col = find_column(translated.columns, ["width", "(箱)寬"])
    translated_height_col = find_column(translated.columns, ["heigth", "height", "(箱)高"])
    translated_weight_col = find_column(translated.columns, ["weigth", "weight", "(箱)重量"])
    translated_units_col = find_column(translated.columns, ["number of cartons", "箱入數"])

    translated = translated[
        [
            translated_code_col,
            translated_name_col,
            translated_length_col,
            translated_width_col,
            translated_height_col,
            translated_weight_col,
            translated_units_col,
        ]
    ].copy()
    translated.columns = [
        "item_code",
        "item_name_translated",
        "box_length",
        "box_width",
        "box_height",
        "box_weight",
        "number_of_item_in_a_box",
    ]
    translated["item_code"] = translated["item_code"].map(normalize_item_code)
    numeric_cols = [
        "box_length",
        "box_width",
        "box_height",
        "box_weight",
        "number_of_item_in_a_box",
    ]
    for column in numeric_cols:
        translated[column] = pd.to_numeric(translated[column], errors="coerce")

    max_comp = pd.read_csv(max_comp_path)
    max_comp_code_col = find_column(max_comp.columns, ["item_code"])
    max_fit_col = find_column(max_comp.columns, ["max_comp_number"])
    max_comp = max_comp[[max_comp_code_col, max_fit_col]].copy()
    max_comp.columns = ["item_code", "max_fit"]
    max_comp["item_code"] = max_comp["item_code"].map(normalize_item_code)
    max_comp["max_fit"] = pd.to_numeric(max_comp["max_fit"], errors="coerce")

    allocation = pd.read_csv(allocation_path)
    allocation_item_col = find_column(allocation.columns, ["item"])
    allocation_pod_col = find_column(allocation.columns, ["pod"])
    allocation_slot_col = find_column(allocation.columns, ["slot"])
    allocation_qty_col = find_column(allocation.columns, ["quantity_in_that_slot"])

    allocation = allocation[
        [allocation_pod_col, allocation_slot_col, allocation_item_col, allocation_qty_col]
    ].copy()
    allocation.columns = ["pod", "slot", "item_code", "qty"]
    allocation["item_code"] = allocation["item_code"].map(normalize_item_code)
    allocation["pod"] = pd.to_numeric(allocation["pod"], errors="coerce").fillna(0).astype(np.int32)
    allocation["slot"] = pd.to_numeric(allocation["slot"], errors="coerce").fillna(0).astype(np.int32)
    allocation["qty"] = pd.to_numeric(allocation["qty"], errors="coerce").fillna(0).round().astype(np.int32)
    allocation = allocation[
        (allocation["item_code"] != "")
        & (allocation["pod"] > 0)
        & (allocation["slot"] > 0)
        & (allocation["qty"] > 0)
    ].copy()

    raw_ordered_skus = set(orders["item_code"].unique())
    metadata_skus = set(metadata["item_code"].dropna().astype(str))
    translated_skus = set(translated["item_code"].dropna().astype(str))
    max_comp_skus = set(max_comp["item_code"].dropna().astype(str))
    eligible_ordered_skus = (
        raw_ordered_skus & metadata_skus & translated_skus & max_comp_skus
    )
    excluded_order_skus = sorted(raw_ordered_skus - eligible_ordered_skus)
    allocated_skus = set(allocation["item_code"].unique())
    missing_allocation = sorted(eligible_ordered_skus - allocated_skus)
    if missing_allocation:
        preview = ", ".join(missing_allocation[:10])
        raise ValueError(
            "The frozen allocation does not cover the eligible 21-day order horizon used by FCGMA. "
            f"Missing {len(missing_allocation)} eligible ordered SKUs, for example: {preview}. "
            "Rerun the FCGMA optimization with the updated aligned/capacity-feasible inputs first."
        )
    orders = orders[orders["item_code"].isin(eligible_ordered_skus)].copy()

    generated_pod_path = output_dir / "generated_pod.csv"
    physical_pod_count = count_physical_pods(generated_pod_path)
    max_pod_index = int(allocation["pod"].max())
    if max_pod_index > physical_pod_count:
        raise ValueError(
            f"Allocation uses pod {max_pod_index}, but the current RMFS geometry only has "
            f"{physical_pod_count} pod positions."
        )

    if int(allocation["slot"].max()) > DEFAULT_SLOT_CAPACITY:
        raise ValueError(
            f"Allocation uses slot {int(allocation['slot'].max())}, exceeding the expected "
            f"{DEFAULT_SLOT_CAPACITY} slots per pod."
        )

    order_frequency = orders.groupby("item_code")["order_id"].nunique().astype(np.int32)
    allocated_quantity = allocation.groupby("item_code")["qty"].sum().astype(np.int32)

    item_master = pd.DataFrame({"item_code": sorted(allocated_skus)})
    item_master["item_order_frequency"] = (
        item_master["item_code"].map(order_frequency).fillna(0).astype(np.int32)
    )
    item_master["item_initial_quantity_inventory"] = (
        item_master["item_code"].map(allocated_quantity).fillna(0).astype(np.int32)
    )
    item_master = item_master.merge(metadata, on="item_code", how="left")
    item_master = item_master.merge(translated, on="item_code", how="left")
    item_master = item_master.merge(max_comp, on="item_code", how="left")

    missing_dimensions = item_master[item_master[numeric_cols + ["max_fit"]].isna().any(axis=1)]
    if not missing_dimensions.empty:
        preview = ", ".join(missing_dimensions["item_code"].head(10))
        raise ValueError(
            "Some allocated SKUs are missing dimension/capacity data required by RMFS. "
            f"Examples: {preview}"
        )

    item_master["item_name"] = (
        item_master["item_name_translated"]
        .fillna(item_master["item_name_meta"])
        .fillna(item_master["item_code"])
    )
    item_master["item_class"] = assign_abc_classes(
        item_master.set_index("item_code")["item_order_frequency"]
    ).values
    item_master["box_volume"] = (
        item_master["box_length"] * item_master["box_width"] * item_master["box_height"]
    )
    item_master["item_volume"] = (
        item_master["box_volume"] / item_master["number_of_item_in_a_box"]
    )
    item_master["item_weight"] = (
        item_master["box_weight"] / item_master["number_of_item_in_a_box"]
    )
    item_master["item_unit"] = "PCS"
    item_master["item_pod_inventory_level"] = POD_THRESHOLD
    item_master["item_warehouse_inventory_level"] = GLOBAL_THRESHOLD
    item_master["max_fit"] = item_master["max_fit"].round().astype(np.int32)
    item_master["item_id"] = np.arange(len(item_master), dtype=np.int32)
    item_master["item_code_numeric"] = pd.to_numeric(item_master["item_code"], errors="raise").astype(np.int64)

    items_output = item_master[
        [
            "item_id",
            "item_code_numeric",
            "item_class",
            "item_order_frequency",
            "item_initial_quantity_inventory",
            "box_length",
            "box_width",
            "box_height",
            "box_volume",
            "box_weight",
            "number_of_item_in_a_box",
            "item_volume",
            "item_weight",
            "item_unit",
            "max_fit",
            "item_pod_inventory_level",
            "item_warehouse_inventory_level",
        ]
    ].copy()
    items_output.columns = [
        "item_id",
        "item_code",
        "item_class",
        "item_order_frequency",
        "item_initial_quantity_inventory",
        "box_length",
        "box_width",
        "box_height",
        "box_volume",
        "box_weight",
        "number_of_item_in_a_box",
        "item_volume",
        "item_weight",
        "item_unit",
        "max_fit",
        "item_pod_inventory_level",
        "item_warehouse_inventory_level",
    ]

    code_to_id = dict(zip(item_master["item_code"], item_master["item_id"]))
    code_to_weight = dict(zip(item_master["item_code"], item_master["item_weight"]))

    pods_output = allocation.copy()
    pods_output["pod_id"] = pods_output["pod"] - 1
    pods_output["slot_id"] = pods_output["slot"] - 1
    pods_output["item"] = pods_output["item_code"].map(code_to_id).astype(np.int32)
    pods_output["pod_type"] = POD_TYPE
    pods_output["slot_type"] = SLOT_TYPE
    pods_output["unusedColumn1"] = 0
    pods_output["unusedColumn2"] = 0
    pods_output["unusedColumn3"] = 0
    pods_output["max_qty"] = pods_output["qty"].astype(np.int32)
    pods_output["due_date"] = 99999
    pods_output["facing"] = 0
    pods_output["pick_ind"] = 0
    pods_output["item_weight"] = pods_output["item_code"].map(code_to_weight).astype(float).round(6)
    pods_output["total_item_weight"] = (pods_output["item_weight"] * pods_output["qty"]).round(6)
    pods_output["item_pod_inventory_level"] = POD_THRESHOLD
    pods_output["item_warehouse_inventory_level"] = GLOBAL_THRESHOLD

    pods_output = pods_output[
        [
            "pod_id",
            "pod_type",
            "slot_id",
            "slot_type",
            "item",
            "unusedColumn1",
            "unusedColumn2",
            "unusedColumn3",
            "qty",
            "max_qty",
            "due_date",
            "facing",
            "pick_ind",
            "item_weight",
            "total_item_weight",
            "item_pod_inventory_level",
            "item_warehouse_inventory_level",
        ]
    ].sort_values(["pod_id", "slot_id"], kind="stable")

    backup_dir = backup_live_files(output_dir)

    items_static_path = output_dir / "items_static_21day.csv"
    pods_static_path = output_dir / "pods_static_21day.csv"
    items_live_path = output_dir / "items.csv"
    pods_live_path = output_dir / "pods.csv"
    summary_path = output_dir / "static_21day_input_summary.csv"
    mapping_path = output_dir / "item_code_to_id_static_21day.csv"

    items_output.to_csv(items_static_path, index=False)
    pods_output.to_csv(pods_static_path, index=False)
    items_output.to_csv(items_live_path, index=False)
    pods_output.to_csv(pods_live_path, index=False)
    item_master[["item_id", "item_code", "item_name"]].to_csv(mapping_path, index=False)

    summary = pd.DataFrame(
        [
            {"metric": "ordered_skus_raw", "value": len(raw_ordered_skus)},
            {"metric": "ordered_skus_eligible", "value": len(eligible_ordered_skus)},
            {"metric": "ordered_skus_excluded_upstream", "value": len(excluded_order_skus)},
            {"metric": "allocated_skus", "value": len(allocated_skus)},
            {"metric": "order_lines", "value": len(orders)},
            {"metric": "unique_orders", "value": int(orders["order_id"].nunique())},
            {"metric": "occupied_slots", "value": len(pods_output)},
            {"metric": "pods_used", "value": int(pods_output["pod_id"].nunique())},
            {"metric": "physical_pods_available", "value": physical_pod_count},
        ]
    )
    summary.to_csv(summary_path, index=False)

    return {
        "items_static_path": items_static_path,
        "pods_static_path": pods_static_path,
        "items_live_path": items_live_path,
        "pods_live_path": pods_live_path,
        "summary_path": summary_path,
        "mapping_path": mapping_path,
        "backup_dir": backup_dir,
        "pods_used": int(pods_output["pod_id"].nunique()),
        "occupied_slots": int(len(pods_output)),
        "allocated_skus": int(len(allocated_skus)),
        "eligible_ordered_skus": int(len(eligible_ordered_skus)),
        "excluded_ordered_skus": int(len(excluded_order_skus)),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parents[1]
    fcgma_dir = find_existing_directory(
        [
            workspace_dir,
            workspace_dir / "fcgma",
            workspace_dir / "revision-fcgma - Copy" / "rmfs-sku-allocation",
            workspace_dir.parent / "fcgma",
            workspace_dir.parent / "revision-fcgma - Copy" / "rmfs-sku-allocation",
        ],
        required_files=["max_comp_number.csv"],
    )
    preprocessing_dir = find_existing_directory(
        [
            fcgma_dir / "Preprocessing",
            workspace_dir / "Preprocessing",
            workspace_dir.parent / "Preprocessing",
        ],
        required_files=["preprocessed_final.csv"],
    )

    parser = argparse.ArgumentParser(
        description="Convert a frozen FCGMA allocation into RMFS Scenario 3 inputs for the static 21-day experiment."
    )
    parser.add_argument(
        "--allocation",
        type=Path,
        default=fcgma_dir / "results_fcgma" / "Z_trial_1.csv",
        help="Path to the frozen Z allocation CSV.",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=preprocessing_dir / "訂單資料_final.csv",
        help="Path to the 21-day order horizon CSV.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=preprocessing_dir / "preprocessed_final.csv",
        help="Path to the aligned product metadata CSV.",
    )
    parser.add_argument(
        "--translated-info",
        type=Path,
        default=preprocessing_dir / "儲格設計_原檔(商品資訊)(Translated).csv",
        help="Path to the translated item dimension/weight master CSV.",
    )
    parser.add_argument(
        "--max-comp",
        type=Path,
        default=fcgma_dir / "max_comp_number.csv",
        help="Path to the slot-capacity-by-SKU CSV.",
    )
    args = parser.parse_args()

    result = prepare_inputs(
        scenario_root=script_dir,
        allocation_path=args.allocation,
        order_path=args.orders,
        metadata_path=args.metadata,
        translated_info_path=args.translated_info,
        max_comp_path=args.max_comp,
    )

    print("Static 21-day RMFS inputs prepared successfully.")
    print(f"Backup directory: {result['backup_dir']}")
    print(f"Live items.csv:   {result['items_live_path']}")
    print(f"Live pods.csv:    {result['pods_live_path']}")
    print(
        f"Coverage: {result['allocated_skus']} allocated SKUs, "
        f"{result['eligible_ordered_skus']} eligible ordered SKUs, "
        f"{result['excluded_ordered_skus']} raw ordered SKUs excluded upstream, "
        f"{result['pods_used']} pods used, "
        f"{result['occupied_slots']} occupied slots"
    )


if __name__ == "__main__":
    main()
