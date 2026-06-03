"""
Conditions:
    A  — 0% missing T1c                        (complete real T1c)
    B  — 50% real / 50% synthetic T1c
    C  — 50% real / 50% missing T1c            (half missingness)
    D  — 100% missing T1c                     (total missingness)

100% Synthetic is not included, as it is trivial to make by replacing *_0002.png images
in the real T1c test dataset with corresponding generated T1c from the synhtetic models.

Simply duplicate the real T1c test set, copy over the synthetic T1c, and rename the dataset according to the
generative model name.

"""

# Non-T1c channels (_0000, _0001, _0003) are always copied from the real T1c folder.
# T1c channel (_0002) varies by condition.


# creates dataset.json per nmnUNet requirements

import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

#
nn_unet_path = Path("/home/u893875/nnUNet/nnUNet_raw")

# Non-T1c channels always sourced from here
real_root = nn_unet_path / "Dataset000_BraTS_RealT1c_complete" / "imagesTs"

# Synthetic T1c sources — only _0002.png files are used from these
pix2pix_root = nn_unet_path / "your_pix2pix_full_synthetic_dataset" / "imagesTs"
custom_root = nn_unet_path / "your_custom_full_synthetic_dataset" / "imagesTs"
syndiff_root = nn_unet_path / "your_syndiff_full_synthetic_dataset" / "imagesTs"


test_patients_txt = Path("/home/u893875/nnUNet/nnUNet_raw/test_patients.txt")

t1c_channel = "0002"
other_channels = ["0000", "0001", "0003"]


def read_patient_list(txt_path: Path) -> list:
    if not txt_path.exists():
        raise FileNotFoundError(f"Patient list not found: {txt_path}")
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]


def get_case_ids_for_patient(real_root: Path, patient_id: str) -> list:
    # Returns sorted list of case_id stems
    return sorted(
        f.stem.rsplit("_", 1)[0]
        for f in real_root.glob(f"{patient_id}_*_{t1c_channel}.png")
    )


def make_zero_slice(reference_path: Path, dest_path: Path):
    # create a zeroed image
    ref = Image.open(reference_path)
    zero = Image.fromarray(np.zeros(ref.size[::-1], dtype=np.uint8))
    zero.save(dest_path)


def split_patients(patients: list, seed: int) -> tuple:
    # Randomly split patient list (50/50).

    random.seed(seed)
    shuffled = patients[:]
    random.shuffle(shuffled)
    mid = len(shuffled) // 2
    return set(shuffled[:mid]), set(shuffled[mid:])


def build_dataset(
    condition: str,
    dataset_id: int,
    dataset_name: str,
    model: str = None,
    seed: int = 42,
):
    if condition in "B":
        if model == "pix2pix":
            synth_root = pix2pix_root
        elif model == "custom":
            synth_root = custom_root
        elif model == "syndiff":
            synth_root = syndiff_root
        else:
            raise ValueError("Condition B requires --model [pix2pix|syndiff|custom]")
    else:
        synth_root = None

    test_patients = read_patient_list(test_patients_txt)

    # ── Per-patient T1c conditions─────────────────────────────────────────────
    if condition == "A":
        t1c_treatment = {p: "real" for p in test_patients}

    elif condition == "B":
        real_set, synth_set = split_patients(test_patients, seed)
        t1c_treatment = {
            p: ("real" if p in real_set else "synthetic") for p in test_patients
        }

    elif condition == "C":
        real_set, zero_set = split_patients(test_patients, seed)
        t1c_treatment = {
            p: ("real" if p in real_set else "missing") for p in test_patients
        }

    elif condition == "D":
        t1c_treatment = {p: "missing" for p in test_patients}

    counts = Counter(t1c_treatment.values())

    # ── Output dir ────────────────────────────────────────────────────────────
    dataset_folder = nn_unet_path / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_ts = dataset_folder / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Condition  : {condition}  model={model or 'n/a'}")
    print(f"Output : {dataset_folder}")
    print(
        f"Patients : {len(test_patients)}  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    print(f"Seed : {seed}")
    print(f"{'─' * 60}\n")

    n_slices_total = 0
    skipped = []

    for patient_id in test_patients:
        case_ids = get_case_ids_for_patient(real_root, patient_id)

        if not case_ids:
            print(f" {patient_id} — no cases found in real root")
            skipped.append(patient_id)
            continue

        treatment = t1c_treatment[patient_id]

        for case_id in case_ids:
            for chan in other_channels:
                src = real_root / f"{case_id}_{chan}.png"
                dest = images_ts / f"{case_id}_{chan}.png"
                if not src.exists():
                    print(f" Missing {src.name} in real root — skipping case")
                    break
                shutil.copy2(src, dest)

            # ── T1c channel (_0002) ─────────────────────
            dest_t1c = images_ts / f"{case_id}_{t1c_channel}.png"
            real_t1c = real_root / f"{case_id}_{t1c_channel}.png"

            if treatment == "real":
                shutil.copy2(real_t1c, dest_t1c)

            elif treatment == "synthetic":
                syn_src = synth_root / f"{case_id}_{t1c_channel}.png"
                if syn_src.exists():
                    shutil.copy2(syn_src, dest_t1c)
                else:
                    print(
                        f"  [WARN] Synthetic T1c not found for {case_id} "
                        f"— falling back to real"
                    )
                    shutil.copy2(real_t1c, dest_t1c)

            elif treatment == "missing":
                make_zero_slice(real_t1c, dest_t1c)

            n_slices_total += 1

        print(f"  ✓ {patient_id}  ({len(case_ids)} slices)  [{treatment} T1c]")

    # ── dataset.json ──────────────────────────────────────────────────────────
    # Created according to the nnU-Net requirements
    dataset_json = {
        "channel_names": {"0": "T2f_FLAIR", "1": "T1n", "2": "T1c", "3": "T2w"},
        "labels": {
            "background": 0,
            "necrotic_tumor": 1,
            "edema": 2,
            "enhancing_tumor": 3,
        },
        "numTest": n_slices_total,
        "file_ending": ".png",
        "name": dataset_name,
        "description": f"BraTS 2023 GLI — 15 pruned axial slices — {dataset_name}",
        "reference": "BraTS 2023",
        "licence": "CC-BY-SA 4.0",
    }
    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(
        f"imagesTs : {n_slices_total:,} slices  "
        f"({len(test_patients) - len(skipped)} patients)"
    )
    if skipped:
        print(f" {len(skipped)} patients skipped:")
        for p in skipped:
            print(f" {p}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create nnU-Net imagesTs")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["A", "B", "C", "D"],
        help=(
            "A=0%% missing T1c | "
            "B=50%% real + 50%% synthetic T1c| "
            "C=50%% real + 50%% missing T1c| "
            "D=100%% missing T1c"
        ),
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["pix2pix", "syndiff", "custom"],
        help="Synthetic model to use for T1c (required for B and C)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 50/50 patient split (B and D)",
    )

    args = parser.parse_args()

    build_dataset(
        condition=args.condition,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        model=args.model,
        seed=args.seed,
    )
