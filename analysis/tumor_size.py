# Stratifying segmentation results by tumor size (small, medium, large)

import json
import pandas as pd
from pathlib import Path

base_dir = Path("/your/segmentation/predictions/folder")


def size_bin(group):
    # size bins determined by quantiles
    q33 = group["total_size"].quantile(0.33)
    q66 = group["total_size"].quantile(0.66)
    group = group.copy()
    group["size_bin"] = pd.cut(
        group["total_size"],
        bins=[0, q33, q66, float("inf")],
        labels=["small", "medium", "large"],
    )
    return group


for folder in sorted(base_dir.iterdir()):
    if not folder.is_dir():
        continue

    # Your segmentation results should be renamed to dataset_summary.json (e.g., half_missing_T1c_summary.json)
    json_files = list(folder.glob("*_summary.json"))
    json_path = json_files[0]
    model_name = json_path.stem.replace(
        "_summary", ""
    )  # grab model name from the file itself

    if not json_path.exists():
        print(f"[SKIP] No JSON found for {folder.name} (expected {json_path.name})")
        continue

    with open(json_path) as f:
        data = json.load(f)

    # --- Build per-slice records ---
    records = []
    for case in data["metric_per_case"]:
        filename = Path(
            case["prediction_file"]
        ).stem  # BraTS-GLI-00002-000_053 --> ID + slice number
        patient_id = filename.rsplit("_", 1)[
            0
        ]  # BraTS-GLI-00002-000 --> only patient ID left

        for cls_str, metrics in case["metrics"].items():
            cls = int(cls_str)

            # class 0 is background
            if cls == 0:
                continue
            records.append(
                {
                    "patient_id": patient_id,
                    "class": cls,
                    "n_ref": metrics["n_ref"],
                    "Dice": metrics["Dice"],
                    "IoU": metrics["IoU"],
                }
            )

    df = pd.DataFrame(records)

    # n_ref of 0 means no tumor
    df_tumor = df[df["n_ref"] > 0].copy()

    # --- Aggregate to patient level ---
    patient_df = (
        df_tumor.groupby(["patient_id", "class"])
        .agg(
            total_size=("n_ref", "sum"),
            Dice=("Dice", "mean"),
            IoU=("IoU", "mean"),
        )
        .reset_index()
    )

    # --- Bin by size ---
    patient_df = patient_df.groupby("class", group_keys=False).apply(size_bin)

    # --- Summarise ---
    result = (
        patient_df.groupby(["class", "size_bin"])[["total_size", "Dice", "IoU"]]
        .mean()
        .round(4)
    )

    # --- Save ---
    seg_path = Path("/your/segmentation/results/folder")
    out_path = seg_path / f"{model_name}_size_stratified.csv"
    result.to_csv(out_path)
    print(f" Saved to {out_path.name}\n")

print("Done!")
