"""
build_nnunet_dataset_external.py
=================================
Converts an external dataset (e.g. UCSF PDGM) into nnU-Net v2 imagesTs +
labelsTs format. No train/val split — all patients go into the test folders.

Remaps segmentation label 4 → 3 to match BraTS 2024 label conventions.

Source structure:
    png_root/
    └── data_pruned/
        └── BraTS-GLI-XXXXX-XXX/
            ├── FLAIR/              *.png
            ├── T1/                 *.png
            ├── T1c/                *.png
            ├── T2/                 *.png
            └── tumor_segmentation/ *.png

Output:
    nnUNet_raw/
    └── Dataset{ID}_{name}/
        ├── dataset.json
        ├── imagesTs/
        │   ├── BraTS-GLI-00000-000_0000_0000.png   ← FLAIR
        │   ├── BraTS-GLI-00000-000_0000_0001.png   ← T1
        │   ├── BraTS-GLI-00000-000_0000_0002.png   ← T1c
        │   ├── BraTS-GLI-00000-000_0000_0003.png   ← T2
        │   └── ...
        └── labelsTs/
            ├── BraTS-GLI-00000-000_0000.png
            └── ...

Usage:
    # Condition A — Real T1c (baseline)
    python build_nnunet_dataset_external.py --dataset_id 200 --dataset_name UCSF_RealT1c

    # Condition B — Synthetic T1c (all patients)
    python build_nnunet_dataset_external.py --dataset_id 201 --dataset_name UCSF_SyntheticT1c \\
        --synthetic_t1c_dir /path/to/synthetic/t1c

    # Condition C — Mixed T1c (50% synthetic)
    python build_nnunet_dataset_external.py --dataset_id 202 --dataset_name UCSF_MixedT1c \\
        --synthetic_t1c_dir /path/to/synthetic/t1c \\
        --mix_ratio 0.5

    # Condition D — Zeroed T1c (ablation)
    python build_nnunet_dataset_external.py --dataset_id 203 --dataset_name UCSF_ZeroedT1c \\
        --zero_t1c
"""

import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ── Configure default paths ───────────────────────────────────────────────────
DEFAULT_PNG_ROOT   = Path("/path/to/dataset_2")
DEFAULT_NNUNET_RAW = Path("/path/to/nnUNet_2/nnUNet_raw")
DATA_SUBDIR        = "data_pruned"    # subfolder inside png_root containing patients

# Modality folder name → nnU-Net channel index
MODALITY_MAP = {
    "FLAIR": "0000",
    "T1":    "0001",
    "T1c":   "0002",
    "T2":    "0003",
}
SEG_FOLDER = "tumor_segmentation"
# ─────────────────────────────────────────────────────────────────────────────


def read_patient_list(txt_path: Path) -> list:
    if not txt_path.exists():
        raise FileNotFoundError(f"Patient list not found: {txt_path}")
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]


def discover_patients(data_dir: Path) -> list:
    """Return sorted list of all patient IDs found in data_dir."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    patients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not patients:
        raise RuntimeError(f"No patient folders found in {data_dir}")
    return patients


def get_slices_for_patient(data_dir: Path, patient_id: str) -> dict:
    """Returns { modality: [sorted png paths] } including 'seg' if present."""
    patient_dir = data_dir / patient_id
    if not patient_dir.exists():
        return {}

    result = {}
    for modality in MODALITY_MAP:
        mod_dir = patient_dir / modality
        if mod_dir.exists():
            slices = sorted(mod_dir.glob("*.png"))
            if slices:
                result[modality] = slices

    seg_dir = patient_dir / SEG_FOLDER
    if seg_dir.exists():
        slices = sorted(seg_dir.glob("*.png"))
        if slices:
            result["seg"] = slices

    return result


def make_zero_slice(reference_path: Path, dest_path: Path):
    """Write a black PNG matching the dimensions of the reference."""
    ref  = Image.open(reference_path)
    zero = Image.fromarray(np.zeros(ref.size[::-1], dtype=np.uint8))
    zero.save(dest_path)


def copy_seg_slice(src: Path, dest_path: Path):
    """Copy a segmentation mask, remapping label 4 → 3."""
    arr = np.array(Image.open(src))
    arr[arr == 4] = 3
    Image.fromarray(arr).save(dest_path)


def process_patients(
    patient_ids: list,
    data_dir: Path,
    images_dest: Path,
    labels_dest: Path,
    synthetic_t1c_dir: Path = None,
    synthetic_patient_ids: set = None,
    zero_t1c: bool = False,
) -> tuple:
    """
    Copies all slices for every patient into imagesTs / labelsTs.
    Returns (n_slices_total, skipped_patients).
    """
    n_slices_total = 0
    skipped        = []

    for patient_id in patient_ids:
        mods = get_slices_for_patient(data_dir, patient_id)

        missing = [m for m in list(MODALITY_MAP.keys()) + ["seg"] if m not in mods]
        if missing:
            print(f"  [WARN] Skipping {patient_id} — missing: {missing}")
            skipped.append(patient_id)
            continue

        use_synthetic = (
            synthetic_t1c_dir is not None
            and synthetic_patient_ids is not None
            and patient_id in synthetic_patient_ids
        )

        for i, flair_slice in enumerate(mods["FLAIR"]):
            case_id = f"{patient_id}_{i:04d}"

            for modality, channel_idx in MODALITY_MAP.items():
                dest = images_dest / f"{case_id}_{channel_idx}.png"

                if modality == "T1c":
                    if zero_t1c:
                        make_zero_slice(mods["T1c"][i], dest)
                    elif use_synthetic:
                        syn = synthetic_t1c_dir / flair_slice.name
                        if syn.exists():
                            shutil.copy2(syn, dest)
                        else:
                            print(f"  [WARN] Synthetic T1c missing for "
                                  f"{patient_id}/{flair_slice.name} — falling back to real")
                            shutil.copy2(mods["T1c"][i], dest)
                    else:
                        shutil.copy2(mods["T1c"][i], dest)
                else:
                    shutil.copy2(mods[modality][i], dest)

            # Copy seg with label 4 → 3 remapping
            copy_seg_slice(mods["seg"][i], labels_dest / f"{case_id}.png")

            n_slices_total += 1

        syn_tag = " [synthetic T1c]" if use_synthetic else (
                  " [zeroed T1c]"    if zero_t1c      else "")
        print(f"  ✓ {patient_id}  ({len(mods['FLAIR'])} slices){syn_tag}")

    return n_slices_total, skipped


def build_dataset(
    png_root: Path,
    nnunet_raw: Path,
    dataset_id: int,
    dataset_name: str,
    synthetic_t1c_dir: Path = None,
    mix_ratio: float = None,
    zero_t1c: bool = False,
    seed: int = 42,
):
    random.seed(seed)

    data_dir     = png_root / DATA_SUBDIR
    all_patients = discover_patients(data_dir)

    print(f"\n{'─'*60}")
    print(f"Found {len(all_patients)} patients in {data_dir}")
    print(f"{'─'*60}")

    # ── Determine synthetic patient set ───────────────────────────────────────
    if synthetic_t1c_dir and mix_ratio is not None:
        n_syn = int(len(all_patients) * mix_ratio)
        synthetic_patients = set(random.sample(all_patients, n_syn))
        print(f"Mixed mode: {n_syn}/{len(all_patients)} patients will use synthetic T1c")
    elif synthetic_t1c_dir:
        synthetic_patients = set(all_patients)
        print(f"Synthetic mode: all {len(all_patients)} patients use synthetic T1c")
    else:
        synthetic_patients = set()
        print(f"Real mode: all {len(all_patients)} patients use real T1c")

    # ── Create output dirs ────────────────────────────────────────────────────
    dataset_folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_ts      = dataset_folder / "imagesTs"
    labels_ts      = dataset_folder / "labelsTs"

    for d in [images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {dataset_folder}\n")

    # ── Process all patients ──────────────────────────────────────────────────
    print(f"Processing {len(all_patients)} patients...")
    n_total, skipped = process_patients(
        patient_ids           = all_patients,
        data_dir              = data_dir,
        images_dest           = images_ts,
        labels_dest           = labels_ts,
        synthetic_t1c_dir     = synthetic_t1c_dir,
        synthetic_patient_ids = synthetic_patients,
        zero_t1c              = zero_t1c,
    )

    # ── dataset.json ──────────────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "FLAIR", "1": "T1", "2": "T1c", "3": "T2"},
        "labels": {
            "background":      0,
            "necrotic_tumor":  1,
            "edema":           2,
            "enhancing_tumor": 3,
        },
        "numTraining": 0,
        "numTest":     n_total,
        "file_ending": ".png",
        "name":        dataset_name,
        "description": f"UCSF PDGM — {dataset_name}",
        "reference":   "UCSF PDGM",
        "licence":     "CC-BY-SA 4.0",
    }
    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok = len(all_patients) - len(skipped)
    print(f"\n{'─'*60}")
    print(f"Dataset ready: {dataset_folder.name}")
    print(f"  imagesTs : {n_total:,} slices  ({n_ok} patients)")
    print(f"  labelsTs : {n_total:,} masks")
    print(f"  dataset.json written ✓")

    if skipped:
        print(f"\n  [WARN] {len(skipped)} patients skipped:")
        for p in skipped:
            print(f"    - {p}")

    print(f"\nNext step on the cluster:")
    print(f"  nnUNetv2_predict \\")
    print(f"    -d {dataset_id:03d} \\")
    print(f"    -i $nnUNet_raw/Dataset{dataset_id:03d}_{dataset_name}/imagesTs \\")
    print(f"    -o $nnUNet_results/predictions/{dataset_name} \\")
    print(f"    -c 2d -f 0")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert external dataset PNGs to nnU-Net v2 imagesTs + labelsTs"
    )
    parser.add_argument("--png_root",         type=Path,  default=DEFAULT_PNG_ROOT,
                        help="Path to dataset_2/ containing data_pruned/")
    parser.add_argument("--nnunet_raw",        type=Path,  default=DEFAULT_NNUNET_RAW,
                        help="Path to nnUNet_raw/")
    parser.add_argument("--dataset_id",        type=int,   required=True,
                        help="3-digit dataset ID e.g. 200")
    parser.add_argument("--dataset_name",      type=str,   required=True,
                        help="Dataset name e.g. UCSF_RealT1c")
    parser.add_argument("--synthetic_t1c_dir", type=Path,  default=None,
                        help="Flat dir of synthetic T1c PNGs — filenames must match real slices.")
    parser.add_argument("--mix_ratio",         type=float, default=None,
                        help="Fraction of patients using synthetic T1c (e.g. 0.5). "
                             "Only used with --synthetic_t1c_dir.")
    parser.add_argument("--zero_t1c",          action="store_true",
                        help="Replace T1c channel with zeros (ablation condition).")
    parser.add_argument("--seed",              type=int,   default=42,
                        help="Random seed for mixed condition patient sampling.")

    args = parser.parse_args()

    if args.zero_t1c and args.synthetic_t1c_dir:
        parser.error("Use either --zero_t1c or --synthetic_t1c_dir, not both.")

    build_dataset(
        png_root          = args.png_root,
        nnunet_raw        = args.nnunet_raw,
        dataset_id        = args.dataset_id,
        dataset_name      = args.dataset_name,
        synthetic_t1c_dir = args.synthetic_t1c_dir,
        mix_ratio         = args.mix_ratio,
        zero_t1c          = args.zero_t1c,
        seed              = args.seed,
    )