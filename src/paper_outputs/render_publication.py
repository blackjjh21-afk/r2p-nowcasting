#!/usr/bin/env python3
"""Render the aggregate manuscript figures and Table 1 from public sources.

The bundled tables are sufficient for Figs. 2--6, Supplementary Figs. S1/S4
and Table 1.  Figure 4 uses the two separately audited aggregate files under
``data/examples``.  Figure 1 and Supplementary Figs. S2/S3 require authorized
station-resolved inputs and are handled by ``paper_outputs.render_restricted``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


LEADS = (60, 90, 120, 150, 180)
THRESHOLDS = (1.0, 5.0, 10.0, 20.0)
ROUTES = ("pysteps", "exprecast", "r2p")
ROUTE_LABELS = {
    "pysteps": "pySTEPS + Patch MLP",
    "exprecast": "exPreCast + Patch MLP",
    "r2p": "Direct R2P",
}
ROUTE_COLORS = {"pysteps": "#2C7FB8", "exprecast": "#009E73", "r2p": "#D62728"}
THRESHOLD_COLORS = {1.0: "#366DB2", 5.0: "#EE8420", 10.0: "#C63E4B", 20.0: "#7750AE"}


class PaperOutputError(RuntimeError):
    """Raised when a compact source table violates the frozen display grid."""


def configure(
    *,
    font_size: float = 10.0,
    axes_labelsize: float = 11.0,
    axes_titlesize: float = 11.0,
    legend_fontsize: float = 9.0,
    tick_labelsize: float = 9.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": axes_labelsize,
            "axes.titlesize": axes_titlesize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save(figure: plt.Figure, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    stem = figure.get_label()
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "direct-r2p-4km10min"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return png, pdf


def audit_grid(frame: pd.DataFrame, *, routes: set[str] | None = None) -> None:
    if set(frame.lead_min.astype(int)) != set(LEADS):
        raise PaperOutputError("lead grid changed")
    if set(frame.threshold_mm.astype(float)) != set(THRESHOLDS):
        raise PaperOutputError("threshold grid changed")
    if routes is not None and set(frame.route_key.astype(str)) != routes:
        raise PaperOutputError("route grid changed")


def render_figure2(data: Path, output: Path) -> tuple[Path, Path]:
    configure(
        font_size=14.0,
        axes_labelsize=15.0,
        axes_titlesize=15.0,
        legend_fontsize=14.0,
        tick_labelsize=14.0,
    )
    density = read(data, "Fig2a_density")
    summary = read(data, "Fig2a_summary").iloc[0]
    amount = read(data, "Fig2b_amount_bins")
    fixed = read(data, "Fig2c_fixed_metrics")
    csi = read(data, "Fig2d_CSI_summary")
    fb = read(data, "Fig2d_FB_summary")
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.2), label="figure2_valid_time_readout")

    ax = axes[0, 0]
    positive = density.pair_count.to_numpy(float) > 0
    artist = ax.scatter(
        density.gauge_log1p[positive], density.fixed_readout_log1p[positive],
        c=density.pair_count[positive], marker="s", s=9, linewidths=0,
        cmap="cividis", norm=LogNorm(vmin=1, vmax=float(density.pair_count.max())),
    )
    ticks_mm = np.asarray([0, 1, 5, 10, 20, 50, 100], float)
    ticks = np.log1p(ticks_mm)
    limit = float(np.log1p(115))
    ax.plot([0, limit], [0, limit], "--", color="#E74C3C", lw=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xticks=ticks, yticks=ticks,
           xticklabels=[f"{v:g}" for v in ticks_mm], yticklabels=[f"{v:g}" for v in ticks_mm],
           xlabel="Gauge RN60 (mm)", ylabel="4-km HSR fixed-readout RN60 (mm)")
    ax.text(0.04, 0.95, f"Pearson $r$ = {float(summary.pearson_r):.3f}",
            transform=ax.transAxes, ha="left", va="top")
    figure.colorbar(artist, ax=ax, label="Pair count (log scale)")
    ax.set_title("(a) Gauge–fixed-readout 2-D-binned density", x=0.58)

    ax = axes[0, 1]
    x = np.arange(len(amount))
    ratio = amount.fixed_readout_to_gauge_amount_ratio.to_numpy(float)
    bias = amount.gauge_conditioned_amount_bias_mm.to_numpy(float)
    ax2 = ax.twinx()
    ax2.bar(x, bias, width=0.68, color="#A8C4D9", alpha=0.85, label="Bias")
    ax.plot(x, ratio, color="#D81B3A", marker="o", lw=2, label="Amount ratio")
    ax.axhline(1, color="0.25", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=amount.gauge_bin, ylim=(0, 1.05),
           xlabel="Gauge accumulation bin (mm)", ylabel="Amount ratio")
    ax2.set(ylim=(-24, 1.2), ylabel="Bias (mm)")
    handles = [Line2D([], [], color="#D81B3A", marker="o", label="Amount ratio"),
               Patch(facecolor="#A8C4D9", label="Bias")]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    ax.set_title("(b) Amount ratio and bias by gauge bin", loc="left")

    ax = axes[1, 0]
    ax.plot(fixed.threshold_mm, fixed.frequency_bias, marker="o", lw=2, color="#2C7FB8")
    ax.axhline(1, color="0.25", ls="--", lw=1)
    for row in fixed.itertuples():
        ax.text(row.threshold_mm, row.frequency_bias + 0.025, f"{row.frequency_bias:.2f}", ha="center")
    ax.set(xticks=THRESHOLDS, ylim=(0, 1.05), xlabel="Threshold (mm)", ylabel="Frequency bias")
    ax.set_title("(c) Event-frequency attenuation", loc="left")

    ax = axes[1, 1]
    route_order = ("fixed_mp", "center_mlp", "patch_mlp")
    labels = {
        "fixed_mp": "Observed HSR + Fixed M–P",
        "center_mlp": "Observed HSR + Center MLP",
        "patch_mlp": "Observed HSR + Patch MLP",
    }
    colors = {"fixed_mp": "#606A73", "center_mlp": "#DE7A60", "patch_mlp": "#2384A6"}
    offsets = (-0.24, 0, 0.24)
    for offset, route in zip(offsets, route_order, strict=True):
        block = csi[csi.route.eq(route)].sort_values("threshold_mm")
        ax.bar(np.arange(4) + offset, block.csi_mean, width=0.22, color=colors[route], alpha=0.55,
               yerr=block.csi_sd, capsize=2)
    ax2 = ax.twinx()
    for route, marker in zip(route_order, ("o", "D", "s"), strict=True):
        if route == "fixed_mp":
            block = csi[csi.route.eq(route)].sort_values("threshold_mm")
            values = block.merge(fixed[["threshold_mm", "frequency_bias"]], on="threshold_mm").frequency_bias
            errors = np.zeros(4)
        else:
            block = fb[fb.route.eq(route)].sort_values("threshold_mm")
            values, errors = block.frequency_bias_mean, block.frequency_bias_sd
        ax2.errorbar(np.arange(4), values, yerr=errors, color=colors[route], marker=marker, lw=1.8,
                     label=labels[route])
    ax2.axhline(1, color="0.25", ls="--", lw=1)
    ax.set(xticks=np.arange(4), xticklabels=["1", "5", "10", "20"], ylim=(0, 1.3),
           xlabel="Threshold (mm)", ylabel="CSI (bars)")
    ax2.set(
        ylim=(0, 1.8),
        yticks=np.arange(0.0, 1.41, 0.2),
        ylabel="Frequency bias (lines)",
    )
    ax2.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fontsize=13.5,
    )
    ax.set_title("(d) Held-out readout CSI and frequency bias", loc="left")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top"]].set_visible(False)
    figure.subplots_adjust(wspace=0.32, hspace=0.36)
    outputs = save(figure, output)
    configure()
    return outputs


def render_figure3(data: Path, output: Path) -> tuple[Path, Path]:
    configure(
        font_size=15.0,
        axes_labelsize=16.0,
        axes_titlesize=15.0,
        legend_fontsize=15.0,
        tick_labelsize=15.0,
    )
    absolute = read(data, "Categorical_mean_sd")
    absolute = absolute[absolute.route_key.isin(ROUTES)].copy()
    audit_grid(absolute, routes=set(ROUTES))
    contrasts = read(data, "Fig3ef_route_contrasts")
    fixed = read(data, "Fig2d_CSI_summary")
    fixed = fixed[fixed.route.eq("fixed_mp")].set_index("threshold_mm")
    figure = plt.figure(figsize=(11.7, 10.3), label="figure3_route_skill_and_contrasts")
    grid = figure.add_gridspec(3, 2, height_ratios=(1, 1, 0.82), hspace=0.62, wspace=0.19)
    axes = [figure.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    contrast_axes = [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1], sharey=None)]
    x = np.arange(5)
    width = 0.22
    for panel, threshold in enumerate(THRESHOLDS):
        ax = axes[panel]
        for idx, route in enumerate(ROUTES):
            block = absolute[np.isclose(absolute.threshold_mm, threshold) & absolute.route_key.eq(route)].set_index("lead_min").loc[list(LEADS)]
            ax.bar(x + (idx - 1) * width, block.csi_mean, width=width, color=ROUTE_COLORS[route],
                   yerr=block.csi_sd, capsize=1.5, alpha=0.96)
        ax.axhline(float(fixed.loc[threshold, "csi_mean"]), color="0.25", ls="--", lw=1.5)
        ax.set(xticks=x, xticklabels=LEADS, xlabel="Lead time (min)", ylabel="CSI" if panel % 2 == 0 else None)
        ax.text(-0.04, 1.06, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        ax.text(0.98, 1.02, f"RN60 ≥ {threshold:g} mm", transform=ax.transAxes, ha="right",
                bbox={"facecolor": "white", "edgecolor": "0.65", "pad": 2})
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    contrast_names = ("exprecast_minus_pysteps", "r2p_minus_exprecast")
    titles = ("exPreCast + Patch MLP\n− pySTEPS + Patch MLP", "Direct R2P\n− exPreCast + Patch MLP")
    t_width = 0.18
    for panel, (ax, name, title) in enumerate(zip(contrast_axes, contrast_names, titles, strict=True), start=4):
        for idx, threshold in enumerate(THRESHOLDS):
            block = contrasts[contrasts.contrast.eq(name) & np.isclose(contrasts.threshold_mm, threshold)].set_index("lead_min").loc[list(LEADS)]
            y = block.delta_csi.to_numpy(float)
            low, high = block.ci_low.to_numpy(float), block.ci_high.to_numpy(float)
            ax.bar(x + (idx - 1.5) * t_width, y, width=t_width, color=THRESHOLD_COLORS[threshold],
                   yerr=np.vstack((y - low, high - y)), error_kw={"ecolor": "0.65", "capsize": 2})
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set(xticks=x, xticklabels=LEADS, xlabel="Lead time (min)", ylabel="ΔCSI" if panel == 4 else None)
        ax.set_title(title, fontweight="bold", fontsize=15.0)
        ax.text(-0.04, 1.08, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    route_handles = [Patch(facecolor=ROUTE_COLORS[r], label=ROUTE_LABELS[r]) for r in ROUTES]
    route_handles.append(Line2D([], [], color="0.25", ls="--", label="Observed HSR + Fixed M–P (valid-time reference)"))
    figure.legend(
        handles=route_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=15.0,
    )
    threshold_handles = [Patch(facecolor=THRESHOLD_COLORS[t], label=f"≥{t:g} mm") for t in THRESHOLDS]
    figure.legend(
        handles=threshold_handles,
        title="RN60 threshold",
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=15.0,
        title_fontsize=15.5,
    )
    figure.subplots_adjust(top=0.85, bottom=0.13)
    outputs = save(figure, output)
    configure()
    return outputs


def render_figure4(data: Path, examples: Path, output: Path) -> tuple[Path, Path]:
    configure(
        font_size=10.5,
        axes_labelsize=11.5,
        axes_titlesize=12.5,
        legend_fontsize=10.5,
        tick_labelsize=10.5,
    )
    radar = pd.read_csv(examples / "radar_only_csi_by_lead_threshold.csv")
    dropout = pd.read_csv(examples / "station_dropout_csi_by_lead_threshold.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=False, label="figure4_r2p_ablations")
    specifications = (
        (radar, "delta_csi_radar_only_minus_standard", "Radar-only − Direct R2P"),
        (dropout, "delta_csi_station_dropout_minus_no_station_dropout", "Direct R2P − no-station-dropout"),
    )
    x = np.arange(5)
    width = 0.18
    for panel, (ax, (frame, value, label)) in enumerate(zip(axes, specifications, strict=True)):
        for idx, threshold in enumerate(THRESHOLDS):
            block = frame[np.isclose(frame.threshold_mm, threshold)].set_index("lead_min").loc[list(LEADS)]
            y = block[value].to_numpy(float)
            low, high = block.ci_low.to_numpy(float), block.ci_high.to_numpy(float)
            ax.bar(x + (idx - 1.5) * width, y, width=width, color=THRESHOLD_COLORS[threshold],
                   yerr=np.vstack((y - low, high - y)), error_kw={"ecolor": "0.65", "capsize": 2})
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set(xticks=x, xticklabels=LEADS, xlabel="Lead time (min)", ylabel="ΔCSI")
        ax.set_title(f"({chr(97 + panel)}) {label}", loc="left")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=THRESHOLD_COLORS[t], label=f"≥{t:g} mm") for t in THRESHOLDS]
    figure.legend(
        handles=handles,
        title="RN60 threshold",
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=10.5,
        title_fontsize=11.0,
    )
    figure.subplots_adjust(top=0.78, wspace=0.22)
    outputs = save(figure, output)
    configure()
    return outputs


def render_figure5(data: Path, output: Path) -> tuple[Path, Path]:
    configure(
        font_size=14.0,
        axes_labelsize=15.0,
        axes_titlesize=16.0,
        legend_fontsize=14.0,
        tick_labelsize=13.5,
    )
    frame = read(data, "Fig5_intensity_point_CI")
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.5), sharey="row", label="figure5_heavy_rain_attenuation")
    route_order = ("observed_hsr", "pysteps_mlp", "exprecast_mlp", "direct_r2p")
    labels = dict(frame[["route", "route_label"]].drop_duplicates().itertuples(index=False, name=None))
    colors = {"observed_hsr": "#3C3C3C", "pysteps_mlp": "#2C7FB8", "exprecast_mlp": "#009E73", "direct_r2p": "#D62728"}
    markers = {"observed_hsr": "D", "pysteps_mlp": "^", "exprecast_mlp": "s", "direct_r2p": "o"}
    for col, lead in enumerate((60, 120, 180)):
        for row, metric in enumerate(("conditional_mean_error", "frequency_bias")):
            ax = axes[row, col]
            for route in route_order:
                block = frame[frame.route.eq(route) & frame.metric.eq(metric) & frame.lead_min.eq(lead)].sort_values("x_index")
                ax.plot(np.arange(4), block.estimate, color=colors[route], marker=markers[route], lw=1.8,
                        ls=":" if route == "observed_hsr" else "-", label=labels[route])
            ax.axhline(0 if row == 0 else 1, color="0.35", ls="--", lw=0.8)
            ax.set(xticks=np.arange(4), xticklabels=block.x_label, xlabel="Gauge amount bin (mm)" if row == 0 else "Threshold (mm)")
            ax.set_ylabel("Gauge-conditioned amount bias (mm)" if row == 0 and col == 0 else "Frequency bias" if row == 1 and col == 0 else None)
            if row == 0:
                ax.set_title(f"({chr(97 + col)}) {lead}-min lead")
            else:
                ax.text(0.0, 1.04, f"({chr(100 + col)})", transform=ax.transAxes, fontweight="bold")
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels_out, loc="upper center", ncol=4, frameon=False)
    figure.subplots_adjust(top=0.86, hspace=0.40, wspace=0.16)
    outputs = save(figure, output)
    configure()
    return outputs


def render_figure_s1(data: Path, output: Path) -> tuple[Path, Path]:
    cells = read(data, "FigS1_distance_cells")
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), sharey=True, label="figureS1_allstation_distance_diagnostic")
    leads = LEADS
    colors = {60: "#447CBD", 90: "#36A497", 120: "#F18424", 150: "#CA4654", 180: "#8059B3"}
    width = 0.145
    groups = sorted(cells["Distance group"].astype(int).unique())
    labels = []
    for group in groups:
        distance_range = cells[cells["Distance group"].eq(group)][
            "Distance to nearest fitting station (km)"
        ].iloc[0]
        lower, upper = distance_range.split("–", maxsplit=1)
        labels.append(f"{float(lower):.1f}–{float(upper):.1f}")
    x = np.arange(4)
    for panel, (ax, threshold) in enumerate(zip(axes.flat, THRESHOLDS, strict=True)):
        for idx, lead in enumerate(leads):
            block = cells[np.isclose(cells["RN60 threshold (mm)"], threshold) & cells["Lead (min)"].eq(lead)].set_index("Distance group").loc[groups]
            y = block["ΔCSI (All-station − Direct)"].to_numpy(float)
            low, high = block["95% CI low"].to_numpy(float), block["95% CI high"].to_numpy(float)
            ax.bar(x + (idx - 2) * width, y, width=width, color=colors[lead],
                   yerr=np.vstack((y - low, high - y)), error_kw={"ecolor": "0.65", "capsize": 2})
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set(xticks=x, xticklabels=labels, xlabel="Distance to nearest fitting station (km)",
               ylabel="ΔCSI" if panel % 2 == 0 else None, ylim=(-0.065, 0.065))
        ax.set_title(
            f"({chr(97 + panel)}) RN60 ≥ {threshold:g} mm",
            loc="left",
            fontweight="bold",
            fontsize=17,
        )
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis="both", labelsize=14.5)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=colors[l], label=f"+{l} min") for l in leads]
    figure.legend(
        handles=handles,
        title="Lead time",
        title_fontsize=15.5,
        fontsize=14.5,
        loc="upper center",
        ncol=5,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.090,
        right=0.980,
        bottom=0.100,
        top=0.835,
        hspace=0.40,
        wspace=0.20,
    )
    return save(figure, output)


def render_figure_s4(data: Path, output: Path) -> tuple[Path, Path]:
    instant_m = read(data, "Field_instant_CSIM")
    instant = read(data, "Field_instant_CSI")
    rn_m = read(data, "Field_RN60_CSIM")
    rn = read(data, "Field_RN60_CSI")
    delta = read(data, "Field_RN60_CSI_delta_CI")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), label="figureS4_native_field_verification")
    ax = axes[0, 0]
    styles = {
        "Public exPreCast": ("#7B52A1", "o", "--"),
        "exPreCast": ("#009E73", "s", "-"),
        "pySTEPS": ("#2C7FB8", "^", "-."),
    }
    for model, block in instant_m.groupby("model", sort=False):
        color, marker, ls = styles[model]
        ax.plot(block.lead_min, block.csi_m, color=color, marker=marker, ls=ls, label=model)
    ax.axvline(60, color="0.55", ls=":")
    ax.set(xlabel="Forecast lead (min)", ylabel="CSI-M", ylim=(0, 0.5))
    ax.set_title("(a) Native-grid rain-rate CSI-M", fontweight="bold", fontsize=15, pad=9)
    ax.legend(frameon=False, fontsize=12.5)

    ax = axes[0, 1]
    palette = {1.0: "#366DB2", 5.0: "#EE8420", 10.0: "#C63E4B", 20.0: "#7750AE"}
    for threshold in THRESHOLDS:
        long = instant[instant.model.eq("exPreCast") & np.isclose(instant.threshold_mm_h, threshold)].set_index("lead_min")
        py = instant[instant.model.eq("pySTEPS") & np.isclose(instant.threshold_mm_h, threshold)].set_index("lead_min")
        common = long.index.intersection(py.index)
        ax.plot(common, long.loc[common].csi - py.loc[common].csi, color=palette[threshold], marker="o", label=f"{threshold:g} mm h$^{{-1}}$")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set(xlabel="Forecast lead (min)", ylabel="ΔCSI (exPreCast − pySTEPS)")
    ax.set_title("(b) Instantaneous CSI difference", fontweight="bold", fontsize=15, pad=9)
    ax.legend(ncol=2, frameon=False, fontsize=12.5)

    ax = axes[1, 0]
    for model, block in rn_m.groupby("model", sort=False):
        color, marker, ls = styles[model]
        ax.plot(block.lead_min, block.csi_m, color=color, marker=marker, ls=ls, label=model)
    ax.set(xlabel="Forecast lead (min)", ylabel="CSI-M", ylim=(0, 0.5))
    ax.set_title("(c) Native-grid RN60 CSI-M", fontweight="bold", fontsize=15, pad=9)
    ax.legend(frameon=False, fontsize=12.5)

    ax = axes[1, 1]
    for threshold in THRESHOLDS:
        block = delta[np.isclose(delta.threshold_mm, threshold)].sort_values("lead_min")
        y = block.delta_long_minus_pysteps.to_numpy(float)
        low, high = block.ci95_lower.to_numpy(float), block.ci95_upper.to_numpy(float)
        ax.errorbar(block.lead_min, y, yerr=np.vstack((y - low, high - y)), color=palette[threshold], marker="o", capsize=2, label=f"{threshold:g} mm")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set(xlabel="Forecast lead (min)", ylabel="ΔCSI (exPreCast − pySTEPS)")
    ax.set_title("(d) Native-grid RN60 CSI difference", fontweight="bold", fontsize=15, pad=9)
    ax.set_ylim(-0.04, 0.17)
    ax.legend(ncol=4, loc="upper center", frameon=False, fontsize=12.5, handlelength=1.8)
    for ax in axes.flat:
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.090,
        top=0.955,
        hspace=0.40,
        wspace=0.30,
    )
    return save(figure, output)


def render_figure6(output: Path) -> tuple[Path, Path]:
    """Run the editable-source-bound renderer used for the final manuscript."""

    release_root = next(
        (
            candidate
            for candidate in (Path.cwd(), *Path.cwd().parents)
            if (candidate / "scripts/finalize_fig6_architecture.py").is_file()
            and (candidate / "figures/main/fig6_architecture_source.pptx").is_file()
        ),
        None,
    )
    if release_root is None:
        raise PaperOutputError(
            "Figure 6 requires a source checkout with scripts/"
            "finalize_fig6_architecture.py and figures/main/fig6_architecture_source.pptx"
        )
    subprocess.run(
        [sys.executable, str(release_root / "scripts/finalize_fig6_architecture.py")],
        cwd=release_root,
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    png = output / "figure6_route_architecture.png"
    pdf = output / "figure6_route_architecture.pdf"
    shutil.copy2(release_root / "figures/main/fig6_architecture.png", png)
    shutil.copy2(release_root / "figures/main/fig6_architecture.pdf", pdf)
    return png, pdf


def render_table1(data: Path, output: Path) -> tuple[Path, Path]:
    frame = read(data, "Table1_main").astype(str)
    figure, ax = plt.subplots(figsize=(14.5, 5.0), label="table1_heldout_point_skill")
    ax.axis("off")
    table = ax.table(cellText=frame.values, colLabels=frame.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.4); cell.set_edgecolor("#AAB2BA")
        if row == 0:
            cell.set_facecolor("#E7EBEF"); cell.set_text_props(fontweight="bold")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "table1_heldout_point_skill.csv", index=False)
    (output / "table1_heldout_point_skill.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    return save(figure, output)


RENDERERS = {
    "2": render_figure2,
    "3": render_figure3,
    "5": render_figure5,
    "s1": render_figure_s1,
    "s4": render_figure_s4,
    "table1": render_table1,
}


def parser() -> argparse.ArgumentParser:
    # Aggregate files are checkout artifacts, not wheel package data.  The
    # installed command defaults to the current checkout; callers elsewhere
    # pass --data-root and --examples-root explicitly.
    package_root = Path.cwd()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--item", choices=["all", "2", "3", "4", "5", "6", "s1", "s4", "table1"], default="all")
    result.add_argument("--data-root", type=Path, default=package_root / "data" / "paper_aggregates")
    result.add_argument("--examples-root", type=Path, default=package_root / "data" / "examples")
    result.add_argument("--output-dir", type=Path, default=package_root / "figures" / "regenerated")
    return result


def main() -> None:
    args = parser().parse_args()
    configure()
    selected = list(RENDERERS) + ["4", "6"] if args.item == "all" else [args.item]
    for item in selected:
        if item == "4":
            outputs = render_figure4(args.data_root, args.examples_root, args.output_dir)
        elif item == "6":
            outputs = render_figure6(args.output_dir)
        else:
            outputs = RENDERERS[item](args.data_root, args.output_dir)
        print("rendered", item, *outputs)


if __name__ == "__main__":
    main()
