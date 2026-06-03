import os
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# =========================================================================== #
#  CONFIGS
# ===========================================================================  #

# JSONS for 0% missing T1c BraTS test, UCSF test, and BraTS training (cross-validation) conditions (respectively)
TEST_JSON = "/path/to/brats_test/summary.json"
UCSF_JSON = "/path/to/ucsf_test/summary.json"
VAL_JSON_GLOB = "/path/to/brats_training_cv/summary.json"

# redex necessary for stripping slice nums from patient IDs
SLICE_RE = re.compile(r"_\d+$")

LABEL_MAP = {1: "NCR", 2: "Edema", 3: "ET"}

OUTPUT_DIR = "path/to/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


SIZE_THRESHOLDS: dict = {}

# click to adjust color
SIZE_COLORS = {"small": "#d64e3f", "medium": "#DD8C0A", "large": "#3185be"}

# click to adjust color
SPLITS = [
    ("Training", "#55A868"),
    ("Test", "#DD8452"),
    ("UCSF", "#8E44AD"),
]


# --------------------------------------------------------------------------- #
#  Adjust layout for 3 plots
# --------------------------------------------------------------------------- #


def axes_layout(figsize=(14, 10)):
    # use gridspec to create a layout with one figure in middle top panel, and two
    # figures on the bottom

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax_top = fig.add_subplot(gs[0, 1:3])
    ax_bl = fig.add_subplot(gs[1, 0:2])
    ax_br = fig.add_subplot(gs[1, 2:4])
    return fig, [ax_top, ax_bl, ax_br]


# --------------------------------------------------------------------------- #
#  parse thru JSON files
# --------------------------------------------------------------------------- #


def _strip_slice_suffix(name, pattern=SLICE_RE):
    return pattern.sub("", name)


def patient_id_from_path(filepath, slice_re=SLICE_RE):
    name = os.path.splitext(os.path.basename(filepath))[0]
    return slice_re.sub("", name)


def aggregate_cases(json_path, slice_re=SLICE_RE):
    with open(json_path) as f:
        raw = json.load(f)

    accum = defaultdict(
        lambda: defaultdict(
            lambda: {"TP": 0.0, "FP": 0.0, "FN": 0.0, "n_ref": 0.0, "n_pred": 0.0}
        )
    )

    for case in raw["metric_per_case"]:
        pid = patient_id_from_path(case["reference_file"], slice_re)
        for label_str, m in case["metrics"].items():
            lb = int(label_str)
            accum[pid][lb]["TP"] += m["TP"]
            accum[pid][lb]["FP"] += m["FP"]
            accum[pid][lb]["FN"] += m["FN"]
            accum[pid][lb]["n_ref"] += m["n_ref"]
            accum[pid][lb]["n_pred"] += m["n_pred"]

    results = {}
    for pid, labels in accum.items():
        row = {}
        dice_vals = []
        total_tumor = 0.0

        for label, name in LABEL_MAP.items():
            tp = labels[label]["TP"]
            fp = labels[label]["FP"]
            fn = labels[label]["FN"]
            nr = labels[label]["n_ref"]
            np_ = labels[label]["n_pred"]
            dice = labels[label]["Dice"]

            row[f"Dice_{name}"] = dice
            row[f"tumor_px_{name}"] = nr
            row[f"TP_{name}"] = tp
            row[f"FP_{name}"] = fp
            row[f"FN_{name}"] = fn
            row[f"n_ref_{name}"] = nr
            row[f"n_pred_{name}"] = np_
            row[f"FN_rate_{name}"] = (fn / nr) if nr > 0 else np.nan
            row[f"FP_rate_{name}"] = (fp / np_) if np_ > 0 else np.nan
            row[f"complete_miss_{name}"] = int(nr > 0 and tp == 0)

            total_tumor += nr
            if not np.isnan(dice):
                dice_vals.append(dice)

        row["Dice_mean"] = float(np.mean(dice_vals)) if dice_vals else np.nan
        row["total_tumor_px"] = total_tumor
        results[pid] = row

    print(f"  {len(results)} patients parsed from {os.path.basename(json_path)}")
    return results


# Done because results are in 5 separate validation folders
def load_val_cases(glob_pattern):
    val_paths = sorted(glob.glob(glob_pattern))
    if not val_paths:
        raise FileNotFoundError(f"No validation JSONs found: {glob_pattern}")
    merged = {}
    for path in val_paths:
        merged.update(aggregate_cases(path))
    print(f"  {len(merged)} total validation patients across {len(val_paths)} folds")
    return merged


# --------------------------------------------------------------------------- #
#  PER-LABEL QUANTILE BUCKETING
# --------------------------------------------------------------------------- #


# compute size thresholds for ease of plotting
def compute_size_thresholds(*dicts):
    # so it can be used outside this function
    global SIZE_THRESHOLDS
    combined = {}
    for d in dicts:
        combined.update(d)

    print("\nSize thresholds (33rd / 66th percentile, Training only):")
    for name in LABEL_MAP.values():
        sizes = [
            v[f"tumor_px_{name}"] for v in combined.values() if v[f"n_ref_{name}"] > 0
        ]
        q33 = float(np.percentile(sizes, 33))
        q66 = float(np.percentile(sizes, 66))
        SIZE_THRESHOLDS[name] = (q33, q66)
        print(
            f"  {name:6s}  small < {q33:.0f}px  |  medium < {q66:.0f}px  |  large >= {q66:.0f}px"
        )


def size_bucket_for_label(patient_row, label_name):
    nr = patient_row.get(f"n_ref_{label_name}", 0)
    if nr == 0:
        return None
    q33, q66 = SIZE_THRESHOLDS[label_name]
    if nr < q33:
        return "small"
    elif nr < q66:
        return "medium"
    else:
        return "large"


# --------------------------------------------------------------------------- #
#  HELPERS
# --------------------------------------------------------------------------- #


def _collect_per_label(cases_dict, field, label_name, bucket=None):
    out = []
    for v in cases_dict.values():
        if bucket and size_bucket_for_label(v, label_name) != bucket:
            continue
        if v.get(f"n_ref_{label_name}", 0) == 0:
            continue
        val = v.get(field, np.nan)
        if not np.isnan(val):
            out.append(val)
    return out


def _boxplot_group(ax, data_lists, labels, colors):
    bp = ax.boxplot(
        data_lists,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, (data, c) in enumerate(zip(data_lists, colors), start=1):
        jitter = np.random.normal(0, 0.06, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, alpha=0.35, s=12, color=c)


# --------------------------------------------------------------------------- #
#  DICE PLOTS
# --------------------------------------------------------------------------- #


def scatter_dice_vs_size(splits, label_name="ET"):
    # x = pixel count, y = Dice.

    fig, axes = axes_layout(figsize=(14, 10))
    q33, q66 = SIZE_THRESHOLDS[label_name]

    fig.suptitle(
        f"Dice vs {label_name} Tumour Size — " + " vs ".join(s[0] for s in splits),
        fontsize=12,
        fontweight="bold",
    )

    for ax, (split_name, cases_dict, _) in zip(axes, splits):
        sizes, dice, buckets = [], [], []
        for v in cases_dict.values():
            nr = v.get(f"n_ref_{label_name}", 0)
            d = v.get(f"Dice_{label_name}", np.nan)
            bkt = size_bucket_for_label(v, label_name)
            if nr == 0 or np.isnan(d) or bkt is None:
                continue
            sizes.append(nr)
            dice.append(d)
            buckets.append(bkt)

        sizes = np.array(sizes)
        dice = np.array(dice)

        for bkt, color in SIZE_COLORS.items():
            mask = np.array([b == bkt for b in buckets])
            ax.scatter(
                sizes[mask],
                dice[mask],
                c=color,
                alpha=0.6,
                s=30,
                label=f"{bkt} (n={mask.sum()})",
            )

        miss = dice == 0.0
        if miss.sum():
            ax.scatter(
                sizes[miss],
                dice[miss],
                edgecolors="black",
                facecolors="none",
                s=80,
                linewidths=1.5,
                label=f"Complete miss (n={miss.sum()})",
            )

        ax.axvline(q33, color=SIZE_COLORS["small"], linestyle=":", linewidth=1)
        ax.axvline(q66, color=SIZE_COLORS["medium"], linestyle=":", linewidth=1)
        ax.axhline(
            np.nanmedian(dice),
            color="grey",
            linestyle="--",
            linewidth=1,
            label=f"Median={np.nanmedian(dice):.3f}",
        )

        ax.set_xscale("log")
        ax.set_xlabel(f"{label_name} region pixels (n_ref, log scale)")
        ax.set_ylabel("Dice" if ax is axes[0] else "")
        ax.set_title(split_name)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"scatter_dice_vs_size_{label_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- #
#  FP/FN summary table
# --------------------------------------------------------------------------- #


def fp_fn_summary(splits):
    print("\n" + "=" * 50)
    print("FP / FN SUMMARY BY SUB-REGION AND SIZE")
    print("=" * 50)
    for name in LABEL_MAP.values():
        print(f"\n-- {name} --")
        for bucket in ["small", "medium", "large"]:
            print(f"  {bucket}:")
            for split_name, cases_dict, _ in splits:
                fn_rates = _collect_per_label(
                    cases_dict, f"FN_rate_{name}", name, bucket
                )
                fp_rates = _collect_per_label(
                    cases_dict, f"FP_rate_{name}", name, bucket
                )
                present = [
                    v
                    for v in cases_dict.values()
                    if size_bucket_for_label(v, name) == bucket
                    and v.get(f"n_ref_{name}", 0) > 0
                ]
                misses = sum(v.get(f"complete_miss_{name}", 0) for v in present)
                miss_pct = 100 * misses / len(present) if present else 0
                print(
                    f"    {split_name:5s}  "
                    f"FN={np.mean(fn_rates):.3f}  "
                    f"FP={np.mean(fp_rates):.3f}  "
                    f"complete_miss={miss_pct:.1f}%  "
                    f"(n={len(present)})"
                )
    print("=" * 50 + "\n")


# --------------------------------------------------------------------------- #
#  PATIENT COUNT HISTOGRAMS
# --------------------------------------------------------------------------- #


def plot_patient_counts_by_size(splits):
    sub_names = list(LABEL_MAP.values())  # [NCR, Edema, ET]
    buckets = ["small", "medium", "large"]
    n_splits = len(splits)
    bar_width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * bar_width

    for chart, ylabel, normalise in [
        ("counts", "Number of patients", False),
        ("proportions", "% of patients in split", True),
    ]:
        fig, axes = axes_layout(figsize=(14, 10))
        fig.suptitle(
            "Patients per Size Bucket — "
            + " vs ".join(s[0] for s in splits)
            + f"  ({chart})",
            fontsize=13,
            fontweight="bold",
        )

        for ax, name in zip(axes, sub_names):
            x = np.arange(len(buckets))

            for offset, (split_name, cases_dict, color) in zip(offsets, splits):
                counts = np.array(
                    [
                        sum(
                            1
                            for v in cases_dict.values()
                            if size_bucket_for_label(v, name) == bucket
                            and v.get(f"n_ref_{name}", 0) > 0
                        )
                        for bucket in buckets
                    ],
                    dtype=float,
                )
                values = (counts / counts.sum() * 100) if normalise else counts

                bars = ax.bar(
                    x + offset,
                    values,
                    bar_width,
                    label=split_name,
                    color=color,
                    alpha=0.8,
                )
                for bar, val in zip(bars, values):
                    label_str = f"{val:.1f}%" if normalise else f"{int(val)}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.3 if not normalise else 0.5),
                        label_str,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_title(name)
            ax.set_xticks(x)
            ax.set_xticklabels([b.capitalize() for b in buckets])
            ax.set_xlabel("Tumor Size")
            ax.set_ylabel(ylabel if ax is axes[0] else "")
            ax.legend(fontsize=9)

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"patient_counts_by_size_{chart}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


# --------------------------------------------------------------------------- #
#  COMPLETE MISS RATE
# --------------------------------------------------------------------------- #


def plot_complete_miss_rate(splits):
    sub_names = list(LABEL_MAP.values())
    buckets = ["small", "medium", "large"]
    n_splits = len(splits)
    bar_width = 0.8 / n_splits
    offsets = np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits) * bar_width
    x_pos = np.arange(len(buckets))

    fig, axes = axes_layout(figsize=(14, 10))
    fig.suptitle(
        "Complete Miss Rate by Sub-region and Tumour Size",
        fontsize=13,
        fontweight="bold",
    )

    for ax, name in zip(axes, sub_names):
        for offset, (split_name, cases_dict, color) in zip(offsets, splits):
            heights = []
            for bkt in buckets:
                patients = [
                    v
                    for v in cases_dict.values()
                    if size_bucket_for_label(v, name) == bkt
                    and v.get(f"n_ref_{name}", 0) > 0
                ]
                n = len(patients)
                miss_rate = (
                    (sum(v[f"complete_miss_{name}"] for v in patients) / n * 100)
                    if n > 0
                    else 0.0
                )
                heights.append(miss_rate)

            bars = ax.bar(
                x_pos + offset,
                heights,
                bar_width,
                label=split_name,
                color=color,
                alpha=0.8,
            )
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{h:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([b.capitalize() for b in buckets])
        ax.set_title(name)
        ax.set_xlabel("Size bucket")
        ax.set_ylabel(
            "% of patients with complete miss (TP = 0)" if ax is axes[0] else ""
        )
        ax.set_ylim(0, 25)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "complete_miss_rate.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_cases = aggregate_cases(TEST_JSON)
    val_cases = load_val_cases(VAL_JSON_GLOB)
    ucsf_cases = aggregate_cases(UCSF_JSON, slice_re=SLICE_RE)

    compute_size_thresholds(val_cases)

    splits = [
        ("Training", val_cases, "#55A868"),
        ("Test", test_cases, "#DD8452"),
        ("UCSF", ucsf_cases, "#8E44AD"),
    ]

    fp_fn_summary(splits)

    print("\n-- Scatter: Dice vs tumour size --")
    for label in ["NCR", "Edema", "ET"]:
        scatter_dice_vs_size(splits, label_name=label)

    print("\n-- Patient counts by size bucket --")
    plot_patient_counts_by_size(splits)

    print("\n-- Complete miss rate --")
    plot_complete_miss_rate(splits)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
