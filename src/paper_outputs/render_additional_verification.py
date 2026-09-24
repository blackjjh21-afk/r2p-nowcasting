"""Render rainfall-episode and CNN patch-size results from aggregate CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEADS = (60, 90, 120, 150, 180)
THRESHOLDS = (1, 5, 10, 20)
ROUTES = {
    "pysteps_cnn": ("pySTEPS + CNN", "#2C7FB8"),
    "exprecast_cnn": ("exPreCast + CNN", "#009E73"),
    "direct_r2p": ("Direct R2P", "#D62728"),
}


def validate_data(data: Path) -> None:
    manifest = json.loads((data / "manifest.json").read_text())
    for entry in manifest["records"]:
        path = data / entry["path"]
        if not path.resolve().is_relative_to(data.resolve()):
            raise ValueError("Aggregate path must remain inside data root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Aggregate checksum mismatch: {path.name}")
        frame = pd.read_csv(path)
        if len(frame) != entry["rows"] or list(frame.columns) != entry["columns"]:
            raise ValueError(f"Aggregate schema mismatch: {path.name}")


def save(fig, output: Path, name: str) -> list[Path]:
    paths = []
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def render_episode(data: Path, output: Path) -> list[Path]:
    frame = pd.read_csv(data / "episode_route_metrics.csv").query("period == 'pooled'")
    expected = {(route, lead) for route in ROUTES for lead in LEADS}
    if len(frame) != len(expected) or set(zip(frame.route, frame.lead_min)) != expected:
        raise ValueError("Expected all three routes at all five leads")
    for column in ("n_episodes", "n_timing_episodes"):
        if frame[column].nunique() != 1:
            raise ValueError("Routes and leads must use common evaluation samples")
    metrics = ("observed_peak_mae_mm", "peak_mae_mm", "timing_mae_min")
    titles = ("Amount at observed peak", "Episode maximum amount", "Episode maximum timing")
    amount_top = float(np.ceil(frame[list(metrics[:2])].max().max() / 5) * 5 + 5)
    time_top = float(np.ceil(frame[metrics[2]].max() / 5) * 5 + 5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for idx, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        for route, (label, color) in ROUTES.items():
            part = frame.loc[frame.route.eq(route)].sort_values("lead_min")
            ax.plot(part.lead_min, part[metric], "o-", label=label, color=color)
        ax.set_title(f"({chr(97 + idx)}) {title}", loc="left", fontsize=11, pad=12)
        ax.set(xlabel="Lead time (min)", ylabel="MAE (min)" if idx == 2 else "MAE (mm)", xticks=LEADS, ylim=(0, time_top if idx == 2 else amount_top))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    return save(fig, output, "episode_peak_mae")


def render_duration(data: Path, output: Path) -> list[Path]:
    frame = pd.read_csv(data / "episode_duration_histogram.csv")
    if not np.all(frame.right_min - frame.left_min == 10) or not np.array_equal(frame.right_min[:-1], frame.left_min[1:]):
        raise ValueError("Duration bins must be contiguous 10-minute intervals")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    edges_hours = np.r_[frame.left_min.to_numpy(), frame.right_min.iloc[-1]] / 60
    ax.stairs(frame.n_episodes, edges_hours, fill=True, facecolor="#BBCBD7", edgecolor="#204F78", linewidth=1.3)
    ax.text(.98, .96, f"n = {int(frame.n_episodes.sum()):,} station-episodes", transform=ax.transAxes, ha="right", va="top")
    end = int(np.ceil(frame.right_min.max() / 60))
    ax.set(xlabel="Episode duration (h; 10-min bins)", ylabel="Number of station-episodes", title="Observed RN60 ≥5 mm episodes with maximum ≥20 mm", xticks=np.arange(0, end + 1), xlim=(0, end), ylim=(0, None))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    return save(fig, output, "episode_duration_histogram")


def render_patch(data: Path, output: Path, *, computed_results: bool = False) -> list[Path]:
    prefix = "" if computed_results else "patch_"
    mean = pd.read_csv(data / f"{prefix}metrics_mean_of_members.csv")
    contrasts = pd.read_csv(data / f"{prefix}paired_CSI_intervals.csv")
    styles = (("exPreCast_CNN3", "exPreCast + CNN3", "#0072B2", "o"),
              ("exPreCast_CNN5", "exPreCast + CNN5", "#009E73", "s"),
              ("Direct_R2P", "Direct R2P", "#D55E00", "^"))
    expected = {(route, lead, threshold) for route, *_ in styles for lead in LEADS for threshold in THRESHOLDS}
    actual = set(zip(mean.route, mean.lead_min, mean.threshold_mm))
    if len(mean) != len(expected) or actual != expected:
        raise ValueError("Patch plots require exPreCast_CNN3, exPreCast_CNN5 and Direct_R2P at the five leads and four thresholds")
    for contrast in ("CNN5_minus_CNN3", "Direct_minus_CNN5"):
        rows = contrasts.loc[contrasts.contrast.eq(contrast)]
        if len(rows) != 20 or set(zip(rows.lead_min, rows.threshold_mm)) != {(lead, threshold) for lead in LEADS for threshold in THRESHOLDS}:
            raise ValueError(f"Expected all 20 comparisons for {contrast}")
    files = []
    for mode in ("skill", "contrasts"):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
        for ax, threshold, letter in zip(axes.flat, THRESHOLDS, "abcd"):
            if mode == "skill":
                for route, label, color, marker in styles:
                    part = mean.loc[mean.route.eq(route) & mean.threshold_mm.eq(threshold)].sort_values("lead_min")
                    ax.plot(part.lead_min, part.CSI, color=color, marker=marker, linewidth=1.6, markersize=4, label=label)
                ax.set(ylabel="CSI", ylim=(0, None))
            else:
                for name, label, color in (("CNN5_minus_CNN3", "CNN5 − CNN3", "#009E73"), ("Direct_minus_CNN5", "Direct − CNN5", "#D55E00")):
                    part = contrasts.loc[contrasts.contrast.eq(name) & contrasts.threshold_mm.eq(threshold)].sort_values("lead_min")
                    ax.plot(part.lead_min, part.CSI_difference, color=color, marker="o", markersize=4, label=label)
                    ax.fill_between(part.lead_min.to_numpy(), part.lower95.to_numpy(), part.upper95.to_numpy(), color=color, alpha=.16)
                ax.axhline(0, color=".4", linewidth=.8, linestyle="--")
                ax.set_ylabel("CSI difference")
            ax.set_title(f"({letter}) RN60 ≥ {threshold:g} mm", loc="left")
            ax.set_xticks(LEADS)
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("Lead time (min)")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3 if mode == "skill" else 2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .91))
        files.extend(save(fig, output, f"patch_spatial_support_{mode}"))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/additional_verification"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/additional_verification"))
    parser.add_argument("--item", choices=("all", "episode", "duration", "patch"), default="all")
    parser.add_argument("--patch-results", type=Path, help="Render a newly computed patch-sensitivity comparison directory; requires --item patch")
    args = parser.parse_args(argv)
    if args.patch_results:
        if args.item != "patch":
            parser.error("--patch-results requires --item patch")
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
            for path in render_patch(args.patch_results, args.output_dir, computed_results=True):
                print(path)
        return
    validate_data(args.data_root)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        for name, renderer in (("episode", render_episode), ("duration", render_duration), ("patch", render_patch)):
            if args.item in ("all", name):
                for path in renderer(args.data_root, args.output_dir):
                    print(path)


if __name__ == "__main__":
    main()
