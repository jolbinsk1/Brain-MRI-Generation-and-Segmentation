import re
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from typing import Optional

# Matches only the base patient ID: UCSF-PDGM-XXXX (4-digit number)
# may be necessary due to the alternate naming convention for follow-ups (e.g., *_FU001d)
PATIENT_RE = re.compile(r"^(UCSF-PDGM-\d{4})")

# ── Cluster paths ─────────────────────────────────────────────────────────────
nn_unet_raw_2 = Path("/path/to/nnUNet/nnUNet_raw_2")

real_root = nn_unet_raw_2 / "your_real_UCSF" / "imagesTs"

synth_roots = {
    "pix2pix": nn_unet_raw_2 / "your/pix2pix/100%/synthetic/folder" / "imagesTs",
    "custom": nn_unet_raw_2 / "your/custom/100%/synthetic/folder" / "imagesTs",
    "syndiff": nn_unet_raw_2 / "your/syndiff/100%/synthetic/folder" / "imagesTs",
}

# Where the shared split is saved (assuming you created one)
split_file = nn_unet_raw_2 / "split_file_you_generated_earlier.json"
# ─────────────────────────────────────────────────────────────────────────────

t1c_channel = "0002"
other_chans = ["0000", "0001", "0003"]

PATIENT_PATTERN = "UCSF-PDGM-*"  # glob prefix used to discover patients


# ── Patient / split helpers ───────────────────────────────────────────────────


def discover_patients(real_root: Path) -> list:
    """
    e.g. UCSF-PDGM-0409_0002.png    -> "UCSF-PDGM-0409"
         UCSF-PDGM-0409_FU001d_0002.png -> "UCSF-PDGM-0409_FU001d"
    """
    ids = set()
    suffix = f"_{t1c_channel}.png"
    for f in real_root.glob(f"*_{t1c_channel}.png"):
        if f.name.endswith(suffix):
            # strip the _<index>_0002.png tail to get the scan prefix
            # e.g. UCSF-PDGM-0409_0007_0002.png -> UCSF-PDGM-0409
            #      UCSF-PDGM-0409_FU001d_0007_0002.png -> UCSF-PDGM-0409_FU001d
            stem = f.name[: -len(suffix)]  # strip _0002.png
            scan_id = stem.rsplit("_", 1)[0]  # strip _<slice_index>
            ids.add(scan_id)
    return sorted(ids)


def load_or_create_split(split_file: Path, patients: list, seed: int = 42) -> dict:
    """
    Load an existing split from split_file, or create one and save it.
    Patients are randomly shuffled with the given seed and split ~50/50.
    The same split file is reused across all conditions for comparability.
    """
    if split_file.exists():
        print(f"[split] Loading existing split from {split_file}")
        with open(split_file) as f:
            split = json.load(f)
        saved = set(split["real"]) | set(split["synthetic_or_zeroed"])
        current = set(patients)
        if saved != current:
            added = current - saved
            removed = saved - current
            print(f"  [WARN] Patient list has changed since split was created.")
            if added:
                print(f"    Added  : {sorted(added)}")
            if removed:
                print(f"    Removed: {sorted(removed)}")
        return split

    print(f"[split] No split file found — generating random split (seed={seed})")
    rng = random.Random(seed)
    shuffled = sorted(patients)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    split = {
        "method": "random",
        "seed": seed,
        "n_total": len(patients),
        "real": sorted(shuffled[:mid]),
        "synthetic_or_zeroed": sorted(shuffled[mid:]),
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f, indent=4)
    print(f"  Saved split -> {split_file}")
    print(
        f"  real={len(split['real'])}  synthetic_or_zeroed={len(split['synthetic_or_zeroed'])}"
    )
    return split


def print_split_summary(split: dict):
    print("\n── Shared patient split ──────────────────────────────────────")
    print(f"  Total patients : {split['n_total']}")
    method = split.get("method", "random")
    print(f"  Method         : {method}")
    print(f"  Real group     : {len(split['real'])} patients")
    print(f"  Synth/Zero grp : {len(split['synthetic_or_zeroed'])} patients")


# ── Slice helpers ─────────────────────────────────────────────────────────────


def get_case_ids_for_patient(real_root: Path, patient_id: str) -> list:
    """
    Returns sorted list of case stems for a given scan_id prefix.
    e.g. scan_id "UCSF-PDGM-0409" returns ["UCSF-PDGM-0409_0000", ..., "UCSF-PDGM-0409_0014"]
         scan_id "UCSF-PDGM-0409_FU001d" returns ["UCSF-PDGM-0409_FU001d_0000", ...]

    Anchors the glob so that UCSF-PDGM-0409 does NOT match UCSF-PDGM-0409_FU001d slices.
    """
    stems = []
    t1c_suffix = f"_{t1c_channel}.png"
    for f in real_root.glob(f"{patient_id}_*_{t1c_channel}.png"):
        # The character immediately after patient_id must be _ then a digit
        # (slice index), not _ then a letter (which would be a follow-up or
        # other scan belonging to a different scan_id entry)
        remainder = f.name[len(patient_id) :]  # e.g. "_0007_0002.png"
        if len(remainder) < 2 or remainder[0] != "_" or not remainder[1].isdigit():
            continue
        stem = f.name[: -len(t1c_suffix)]  # strip _0002.png
        stems.append(stem.rsplit("_", 1)[0])  # strip _<slice_index>
    return sorted(stems)


def make_zero_slice(reference_path: Path, dest_path: Path):
    """Write a black PNG matching the dimensions of reference_path."""
    ref = Image.open(reference_path)
    zero = Image.fromarray(np.zeros(ref.size[::-1], dtype=np.uint8))
    zero.save(dest_path)


# ── Main builder ──────────────────────────────────────────────────────────────


def build_dataset(
    condition: str,
    dataset_id: int,
    dataset_name: str,
    split: dict,
    model: Optional[str] = None,
    dry_run: bool = False,
):
    # ── Validate model requirement ─────────────────────────────────────────
    if condition in ("B", "C"):
        if model not in SYNTH_ROOTS:
            raise ValueError(
                f"Conditions B and C require --model [{'/'.join(SYNTH_ROOTS)}]"
            )
        synth_root = SYNTH_ROOTS[model]
    else:
        synth_root = None

    # ── Per-patient T1c treatment ──────────────────────────────────────────
    real_set = set(split["real"])
    synth_set = set(split["synthetic_or_zeroed"])

    all_patients = sorted(real_set | synth_set)

    if condition == "A":
        t1c_treatment = {p: "real" for p in all_patients}

    elif condition == "B":
        t1c_treatment = {
            p: ("real" if p in real_set else "synthetic") for p in all_patients
        }

    elif condition == "C":
        t1c_treatment = {p: "synthetic" for p in all_patients}

    elif condition == "D":
        t1c_treatment = {
            p: ("real" if p in real_set else "zeroed") for p in all_patients
        }

    elif condition == "E":
        t1c_treatment = {p: "zeroed" for p in all_patients}

    counts = Counter(t1c_treatment.values())

    # ── Output dir ─────────────────────────────────────────────────────────
    dataset_folder = NNUNET_RAW / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_ts = dataset_folder / "imagesTs"

    if not dry_run:
        images_ts.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 65}")
    print(f"  Condition    : {condition}  |  model = {model or 'n/a'}")
    print(f"  Output       : {dataset_folder}")
    print(
        f"  Patients     : {len(all_patients)}  |  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    if dry_run:
        print("  *** DRY RUN — no files will be written ***")
    print(f"{'─' * 65}\n")

    n_slices_total = 0
    skipped = []
    synth_fallback = 0

    for patient_id in all_patients:
        case_ids = get_case_ids_for_patient(REAL_ROOT, patient_id)
        treatment = t1c_treatment[patient_id]

        if not case_ids:
            print(f"  [WARN] {patient_id} — no cases in real root, skipping")
            skipped.append(patient_id)
            continue

        for case_id in case_ids:
            # ── Non-T1c channels — always real ──────────────────────────────
            def safe_copy(src: Path, dest: Path, label: str = "") -> bool:
                """Copy src to dest. Returns False and warns on broken symlinks or missing files."""
                if not src.exists():
                    tag = f" [{label}]" if label else ""
                    is_symlink = src.is_symlink()
                    reason = "broken symlink" if is_symlink else "file not found"
                    print(
                        f"  [WARN] {reason}: {src.name}{tag} — skipping case {case_id}"
                    )
                    return False
                shutil.copy(
                    str(src), str(dest)
                )  # copy() skips metadata, avoids symlink copystat bug
                return True

            all_ok = True
            for ch in other_chans:
                src = REAL_ROOT / f"{case_id}_{ch}.png"
                dest = images_ts / f"{case_id}_{ch}.png"
                if not dry_run:
                    if not safe_copy(src, dest, ch):
                        all_ok = False
                        break
                elif not src.exists():
                    print(f"  [WARN] Missing {src.name} — would skip case {case_id}")
                    all_ok = False
                    break
            if not all_ok:
                continue

            # ── T1c channel — varies by treatment ───────────────────────────
            real_t1c = real_root / f"{case_id}_{t1c_channel}.png"
            dest_t1c = images_ts / f"{case_id}_{t1c_channel}.png"

            if treatment == "real":
                if not dry_run:
                    if not safe_copy(real_t1c, dest_t1c, "real T1c"):
                        continue

            elif treatment == "synthetic":
                syn_src = synth_root / f"{case_id}_{t1c_channel}.png"
                if syn_src.exists():
                    if not dry_run:
                        safe_copy(syn_src, dest_t1c, "synthetic T1c")
                else:
                    is_followup = "_FU" in case_id
                    if not is_followup:
                        print(
                            f"  [WARN] Synthetic T1c missing for {case_id} (baseline!) — falling back to real"
                        )
                    synth_fallback += 1
                    if not dry_run:
                        if not safe_copy(real_t1c, dest_t1c, "real T1c fallback"):
                            continue

            elif treatment == "zeroed":
                if not dry_run:
                    make_zero_slice(real_t1c, dest_t1c)

            n_slices_total += 1

        n_followup = sum(1 for c in case_ids if "_FU" in c)
        n_baseline = len(case_ids) - n_followup
        fu_note = f" + {n_followup} follow-up" if n_followup else ""
        print(
            f"  {'(dry) ' if dry_run else ''}✓ {patient_id}  "
            f"({n_baseline} baseline{fu_note} slices)  [{treatment} T1c]"
        )

    # ── dataset.json ───────────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {
            "0": "T2f_FLAIR",
            "1": "T1n",
            "2": "T1c",
            "3": "T2w",
        },
        "labels": {
            "background": 0,
            "necrotic_tumor": 1,
            "edema": 2,
            "enhancing_tumor": 3,
        },
        "numTest": n_slices_total,
        "file_ending": ".png",
        "name": dataset_name,
        "description": (
            f"UCSF-PDGM — 15 pruned axial slices — "
            f"Condition {condition}" + (f" [{model}]" if model else "")
        ),
        "reference": "UCSF-PDGM",
        "licence": "CC-BY-SA 4.0",
    }

    if not dry_run:
        with open(dataset_folder / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(
        f"  imagesTs     : {n_slices_total:,} slices  "
        f"({len(all_patients) - len(skipped)} patients processed)"
    )
    print(f"  dataset.json : {'written ✓' if not dry_run else 'skipped (dry run)'}")
    if synth_fallback:
        print(f"  [WARN] {synth_fallback} slices fell back to real (synthetic missing)")
    if skipped:
        print(f"  [WARN] {len(skipped)} patients skipped (not in real root):")
        for p in skipped:
            print(f"    - {p}")
    print(f"\n  Run inference (model trained on Dataset101):")
    print(f"    nnUNetv2_predict \\")
    print(f"      -d 101 \\")
    print(f"      -i $nnUNet_raw_2/Dataset{dataset_id:03d}_{dataset_name}/imagesTs \\")
    print(f"      -o $nnUNet_results/predictions/{dataset_name} \\")
    print(f"      -c 2d -f 0")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build nnU-Net imagesTs for UCSF-PDGM experimental conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        help=(
            "A=100%% real | "
            "B=50%% real + 50%% synthetic | "
            "C=100%% synthetic | "
            "D=50%% real + 50%% zeroed | "
            "E=100%% zeroed"
        ),
    )
    parser.add_argument("--dataset_id", type=int, help="Numeric dataset ID (e.g. 107)")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name string (e.g. UCSF_PDGM_pix2pix_50Synth)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(synth_roots.keys()),
        help="Synthetic model for conditions B and C",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation (ignored if split already exists)",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        default=split_file,
        help=f"Path to shared patient split JSON (default: {split_file})",
    )
    parser.add_argument(
        "--generate_split_only",
        action="store_true",
        help="Discover patients, generate/show the split, then exit",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be built without writing any files",
    )

    args = parser.parse_args()

    # ── Discover patients from real root ──────────────────────────────────
    if not real_root.exists():
        raise SystemExit(f"ERROR: real root not found:\n  {real_root}")

    patients = discover_patients(real_root)
    if not patients:
        raise SystemExit(f"ERROR: no UCSF-PDGM patients found in {real_root}")

    print(f"[patients] Discovered {len(patients)} patients in real root.")

    # ── Load or create the shared split ───────────────────────────────────
    split = load_or_create_split(args.split_file, patients, seed=args.seed)
    print_split_summary(split)

    if args.generate_split_only:
        print("Split file written. Exiting (--generate_split_only).")
        raise SystemExit(0)

    # ── Validate remaining args ───────────────────────────────────────────
    if not args.condition:
        parser.error("--condition is required unless using --generate_split_only")
    if not args.dataset_id:
        parser.error("--dataset_id is required")
    if not args.dataset_name:
        parser.error("--dataset_name is required")

    build_dataset(
        condition=args.condition,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        split=split,
        model=args.model,
        dry_run=args.dry_run,
    )
