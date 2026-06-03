
'''
Creates a dataset from real BraTS 2023 data that fits the nnU-Net 2D segmentation model proposed by Isensee et al. (2021).
specific segmentation conditions are written in segmentation_conditions.py
run this first to get the 100% real T1c train and test sets

dataset structures are visualized in nnUNet_folder_structure.txt

See original nnU-Net repo for more details: https://github.com/MIC-DKFZ/nnUNet
'''

import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── Configure default paths ───────────────────────────────────────────────────
png_root = Path("/path/to/local_root_folder")
nn_unet_raw = Path("/path/to/nnUNet/nnUNet_raw")

train_patients_txt = "train_patients.txt"
validation_patients_txt = "validation_patients.txt"
test_patients_txt = "test_patients.txt"
split_directory = "pruned_data_folder"

# BraTS 2023 modality folder → nnU-Net channel index
modalities_map = {
    "t2f": "0000",  # FLAIR
    "t1n": "0001",  # T1 weighted
    "t1c": "0002",  # T1 contrast
    "t2w": "0003",  # T2 weighted
}
# ─────────────────────────────────────────────────────────────────────────────


def read_patient_list(txt_path: Path) -> list:
    if not txt_path.exists():
        raise FileNotFoundError(f"Patient list not found: {txt_path}")
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]


def get_slices_for_patient(split_dir: Path, patient_id: str) -> dict:
    """Returns { modality: [sorted png paths] } for a patient."""
    patient_dir = split_dir / patient_id
    if not patient_dir.exists():
        return {}
    result = {}
    for modality in list(modalities_map.keys()) + ["seg"]:
        mod_dir = patient_dir / modality
        if mod_dir.exists():
            slices = sorted(mod_dir.glob("*.png"))
            if slices:
                result[modality] = slices
    return result


def process_patients(
    patient_ids: list,
    split_dir: Path,
    images_dest: Path,
    labels_dest: Path,  # None for test when skipping labelsTs here
    include_labels: bool,
) -> tuple:
    # Copies into the nnU-Net folder structure.

    n_slices = 0
    skipped = []

    for patient_id in patient_ids:
        mods = get_slices_for_patient(split_dir, patient_id)

        required = list(modalities_map.keys())
        if include_labels:
            required.append("seg")

        missing = [m for m in required if m not in mods]
        if missing:
            print(f"  [WARN] Skipping {patient_id} — missing: {missing}")
            skipped.append(patient_id)
            continue

        for i, t2f_slice in enumerate(mods["t2f"]):
            case_id = f"{patient_id}_{i:04d}"

            for modality, channel_idx in modalities_map.items():
                dest = images_dest / f"{case_id}_{channel_idx}.png"

                if modality == "t1c":
                    shutil.copy2(mods["t1c"][i], dest)
                else:
                    shutil.copy2(mods[modality][i], dest)

            if include_labels and labels_dest is not None:
                shutil.copy2(mods["seg"][i], labels_dest / f"{case_id}.png")

            n_slices += 1

        print(f"  ✓ {patient_id}  ({len(mods['t2f'])} slices)")

    return n_slices, skipped


def build_dataset(
    png_root: Path,
    nnunet_raw: Path,
    dataset_id: int,
    dataset_name: str,
    seed: int = 42,
):
    random.seed(seed)

    split_root = png_root / split_directory

    # ── Read patient lists ────────────────────────────────────────────────────
    train_patients = read_patient_list(png_root / train_patients_txt)
    val_patients = read_patient_list(png_root / validation_patients_txt)
    test_patients = read_patient_list(png_root / test_patients_txt)
    trainval = train_patients + val_patients

    print(f"\n{'─' * 60}")
    print(
        f"Patients — train: {len(train_patients)}, "
        f"val: {len(val_patients)}, test: {len(test_patients)}"
    )
    print(f"imagesTr pool: {len(trainval)} patients")
    print(f"{'─' * 60}")

    # ── Create output dirs ────────────────────────────────────────────────────
    dataset_folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    labels_ts = dataset_folder / "labelsTs"

    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {dataset_folder}\n")

    # ── imagesTr + labelsTr (train + val) ─────────────────────────────────────
    print(f"=== imagesTr + labelsTr ({len(trainval)} patients) ===")
    n_train, skipped_train = process_patients(
        patient_ids=trainval,
        split_dir=split_root / "train",  # train and val share the same
        images_dest=images_tr,  # logic via patient txt lists
        labels_dest=labels_tr,
        include_labels=True,
    )

    # ── imagesTs + labelsTs (test) ─────────────────────────────────────────────
    print(f"\n=== imagesTs + labelsTs ({len(test_patients)} patients) ===")
    n_test, skipped_test = process_patients(
        patient_ids=test_patients,
        split_dir=split_root / "test",
        images_dest=images_ts,
        labels_dest=labels_ts,
        include_labels=True,  # labelsTs for evaluation
    )

    # ── dataset.json ──────────────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "T2f_FLAIR", "1": "T1n", "2": "T1c", "3": "T2w"},
        "labels": {
            "background": 0,
            "necrotic_tumor": 1,
            "edema": 2,
            "enhancing_tumor": 3,
        },
        "numTraining": n_train,
        "numTest": n_test,
        "file_ending": ".png",
        "name": dataset_name,
        "description": f"BraTS 2023 GLI — {dataset_name}",
        "reference": "BraTS 2023",
        "licence": "CC-BY-SA 4.0",
    }
    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Dataset ready: {dataset_folder.name}")
    print(
        f"  imagesTr : {n_train:,} slices  "
        f"({len(trainval) - len(skipped_train)} patients)"
    )
    print(f"  labelsTr : {n_train:,} masks")
    print(
        f"  imagesTs : {n_test:,} slices  "
        f"({len(test_patients) - len(skipped_test)} patients)"
    )
    print(f"  labelsTs : {n_test:,} masks")
    print(f"  dataset.json written ✓")

    if skipped_train:
        print(f"\n  [WARN] {len(skipped_train)} train/val patients skipped:")
        for p in skipped_train:
            print(f"    - {p}")
    if skipped_test:
        print(f"\n  [WARN] {len(skipped_test)} test patients skipped:")
        for p in skipped_test:
            print(f"    - {p}")

    print(f"\nNext step:")
    print(
        f"  nnUNetv2_plan_and_preprocess -d {dataset_id:03d} --verify_dataset_integrity"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BraTS 2023 PNGs to a full nnU-Net dataset"
    )
    parser.add_argument(
        "--png_root",
        type=Path,
        default=png_root,
        help="Path to png_data/ containing 256_pruned_split/ and txt files",
    )
    parser.add_argument(
        "--nnunet_raw", type=Path, default=nn_unet_raw, help="Path to nnUNet_raw/"
    )
    parser.add_argument(
        "--dataset_id", type=int, required=True, help="3-digit dataset ID e.g. 100"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name e.g. BraTS_RealT1c",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mixed condition patient sampling.",
    )

    args = parser.parse_args()

    build_dataset(
        png_root=args.png_root,
        nnunet_raw=args.nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        seed=args.seed,
    )
