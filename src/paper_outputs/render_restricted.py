#!/usr/bin/env python3
"""Render station-resolved paper figures from author-supplied inputs.

No station-resolved inputs are bundled.  This module documents and implements
the final display transformations for Fig. 1 and Supplementary Figs. S2/S3 so
authorized users can reproduce them after supplying the corresponding files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROUTE_ORDER = ("truth", "pysteps_patch_cnn", "exprecast_patch_cnn", "direct_r2p")
ROUTE_COLORS = {
    "truth": "#303030",
    "pysteps_patch_cnn": "#2C7FB8",
    "exprecast_patch_cnn": "#009E73",
    "direct_r2p": "#D62728",
}
ROUTE_LABELS = {
    "truth": "Gauge truth",
    "pysteps_patch_cnn": "pySTEPS + CNN",
    "exprecast_patch_cnn": "exPreCast + CNN",
    "direct_r2p": "Direct R2P",
    "pysteps": "pySTEPS + CNN",
    "exprecast": "exPreCast + CNN",
    "r2p": "Direct R2P",
}

THRESHOLDS = (1.0, 5.0, 10.0, 20.0)


def configure_cartopy(data_dir: Path):
    """Use a caller-supplied Natural Earth cache without network downloads."""

    try:
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - dependency message
        raise SystemExit("install the paper-output optional dependency cartopy") from exc
    data_dir = Path(data_dir).expanduser().resolve()
    required = (
        "shapefiles/natural_earth/physical/ne_10m_land.shp",
        "shapefiles/natural_earth/physical/ne_10m_ocean.shp",
        "shapefiles/natural_earth/physical/ne_10m_coastline.shp",
        "shapefiles/natural_earth/cultural/ne_10m_admin_0_boundary_lines_land.shp",
    )
    missing = [relative for relative in required if not (data_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"--cartopy-data-dir lacks the required Natural Earth 10m files: {missing}"
        )
    cartopy.config["data_dir"] = str(data_dir)
    return ccrs, cfeature


def add_base_map(
    ax,
    ccrs,
    cfeature,
    *,
    extent: tuple[float, float, float, float],
    grid_label_size: float = 8.0,
) -> None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("#EAF3F7")
    background = (
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#EAF3F7", zorder=0),
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#F5F3EC", zorder=0),
        ax.add_feature(
            cfeature.BORDERS.with_scale("10m"), edgecolor="#8A8F94",
            linewidth=0.50, zorder=3,
        ),
        ax.coastlines(resolution="10m", color="#515A61", linewidth=0.72, zorder=4),
    )
    # Repeating the detailed Natural Earth polygons in every S3 panel would
    # otherwise produce a very large vector PDF.  The point symbols and text
    # remain vector while the geographic context layer is rasterized.
    for artist in background:
        artist.set_rasterized(True)
    grid = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4,
        color="#9099A1", alpha=0.45, linestyle=":"
    )
    grid.top_labels = False
    grid.right_labels = False
    grid.xlabel_style = {"size": grid_label_size, "color": "#66727D"}
    grid.ylabel_style = {"size": grid_label_size, "color": "#66727D"}


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white",
        metadata={"Software": "direct-r2p-4km10min"},
    )
    fig.savefig(
        output / f"{stem}.pdf", bbox_inches="tight", facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def render_figure1(
    coverage_npz: Path,
    stations_csv: Path,
    cartopy_data_dir: Path,
    output: Path,
) -> None:
    """Render the domain map from an authorized footprint and station split.

    ``coverage_npz`` must contain 2-D ``lon``, ``lat`` and boolean ``footprint``
    arrays.  The station CSV must contain ``lon``, ``lat`` and ``split`` with
    514 ``train`` and 128 ``test`` rows.
    """

    archive = np.load(coverage_npz, allow_pickle=False)
    if set(("lon", "lat", "footprint")) - set(archive.files):
        raise ValueError("coverage archive must contain lon, lat and footprint")
    lon, lat, footprint = archive["lon"], archive["lat"], archive["footprint"].astype(bool)
    if lon.shape != lat.shape or lon.shape != footprint.shape:
        raise ValueError("coverage arrays must share one 2-D shape")
    stations = pd.read_csv(stations_csv, encoding="utf-8-sig")
    required = {"lon", "lat", "split"}
    if not required.issubset(stations):
        raise ValueError(f"station split lacks {sorted(required - set(stations))}")
    fitting, heldout = stations[stations.split.eq("train")], stations[stations.split.eq("test")]
    if (len(fitting), len(heldout)) != (514, 128):
        raise ValueError("expected the frozen 514/128 split")
    ccrs, cfeature = configure_cartopy(cartopy_data_dir)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.1, 8.0), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    covered_lon, covered_lat = lon[footprint], lat[footprint]
    extent = (
        float(np.nanmin(covered_lon)) - 0.25,
        float(np.nanmax(covered_lon)) + 0.25,
        float(np.nanmin(covered_lat)) - 0.25,
        float(np.nanmax(covered_lat)) + 0.25,
    )
    add_base_map(ax, ccrs, cfeature, extent=extent)
    stride = max(1, int(np.ceil(max(lon.shape) / 600)))
    sl = (slice(None, None, stride), slice(None, None, stride))
    ax.contourf(
        lon[sl], lat[sl], footprint[sl].astype(int), levels=[0.5, 1.5],
        colors=["#55A8C7"], alpha=0.22, transform=projection, zorder=1,
    )
    ax.contour(
        lon[sl], lat[sl], footprint[sl].astype(int), levels=[0.5],
        colors=["#147A9B"], linewidths=0.95, transform=projection, zorder=2,
    )
    ax.scatter(
        fitting.lon, fitting.lat, s=8, c="#5F666B", edgecolors="none",
        alpha=0.82, transform=projection, zorder=6,
    )
    ax.scatter(
        heldout.lon, heldout.lat, s=36, c="#D81B3A", marker="^",
        edgecolors="white", linewidths=0.40, alpha=0.96,
        transform=projection, zorder=7,
    )
    handles = [
        Patch(fc="#55A8C7", ec="#147A9B", alpha=0.35, label="Observed HSR composite coverage"),
        Line2D([], [], marker="o", ls="", color="#5F666B", label="Fitting stations (n=514)"),
        Line2D([], [], marker="^", ls="", color="#D81B3A", label="Held-out stations (n=128)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.985)
    save(fig, output, "figure1_study_design")


def read_excel(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install the paper-output optional dependency openpyxl") from exc


def render_s2(workbook: Path, output: Path) -> None:
    series = read_excel(workbook, "FigS2_station_timeseries")
    metrics = read_excel(workbook, "Case_mean_sd")
    required_series = {"case_rank", "valid_time_kst", "lead_min", "route", "value_mean_mm", "value_min_mm", "value_max_mm"}
    required_metrics = {"case_rank", "lead_min", "threshold_mm", "route", "csi_mean", "csi_sd"}
    if not required_series.issubset(series) or not required_metrics.issubset(metrics):
        raise ValueError("Supplementary Data 1 station/case sheets do not match the frozen schema")
    series["valid_time_kst"] = pd.to_datetime(series.valid_time_kst)
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.8), label="figureS2_case_analysis")
    for row, rank in enumerate((1, 2)):
        ax = axes[row, 0]
        for route in ROUTE_ORDER:
            block = series[series.case_rank.eq(rank) & series.route.eq(route) & series.lead_min.eq(60)].sort_values("valid_time_kst")
            if block.empty:
                raise ValueError(f"missing case {rank}, route {route}")
            ax.plot(block.valid_time_kst, block.value_mean_mm, color=ROUTE_COLORS[route], lw=1.5,
                    ls=":" if route == "truth" else "-", label=ROUTE_LABELS[route])
            if route != "truth":
                ax.fill_between(block.valid_time_kst, block.value_min_mm, block.value_max_mm,
                                color=ROUTE_COLORS[route], alpha=0.12, linewidth=0)
        ax.set(ylabel="RN60 (mm)", title=f"({chr(97 + 2 * row)}) +60-min station time series")
        ax.title.set_fontsize(15)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=13)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)

        ax = axes[row, 1]
        x = np.arange(5); width = 0.22
        for idx, route in enumerate(ROUTE_ORDER[1:]):
            block = metrics[metrics.case_rank.eq(rank) & metrics.route.eq(route) & np.isclose(metrics.threshold_mm, 10)].set_index("lead_min").loc[[60, 90, 120, 150, 180]]
            ax.bar(x + (idx - 1) * width, block.csi_mean, width=width, yerr=block.csi_sd,
                   color=ROUTE_COLORS[route], capsize=2, label=ROUTE_LABELS[route])
        ax.set(xticks=x, xticklabels=[60, 90, 120, 150, 180], xlabel="Lead time (min)", ylabel="10-mm event CSI",
               title=f"({chr(98 + 2 * row)}) Event CSI across held-out stations")
        ax.title.set_fontsize(15)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.090,
        right=0.985,
        bottom=0.100,
        top=0.835,
        hspace=0.50,
        wspace=0.28,
    )
    save(figure, output, "figureS2_case_analysis")


def render_s3(
    workbook: Path,
    stations_csv: Path,
    cartopy_data_dir: Path,
    output: Path,
) -> None:
    station = read_excel(workbook, "FigS3_station_CSI")
    required = {"station_id", "lat", "lon", "route", "threshold_mm", "csi_mean"}
    if not required.issubset(station):
        raise ValueError("FigS3_station_CSI does not match the frozen schema")
    all_stations = pd.read_csv(stations_csv, encoding="utf-8-sig")
    required_stations = {"station_id", "lat", "lon", "split"}
    if not required_stations.issubset(all_stations):
        raise ValueError(f"station split lacks {sorted(required_stations - set(all_stations))}")
    fitting = all_stations[all_stations.split.eq("train")]
    heldout = all_stations[all_stations.split.eq("test")]
    if (len(fitting), len(heldout)) != (514, 128):
        raise ValueError("expected the frozen 514/128 split")
    expected_ids = set(station.station_id.astype(int).unique())
    if expected_ids != set(heldout.station_id.astype(int)):
        raise ValueError("FigS3 station IDs differ from the held-out station split")

    ccrs, cfeature = configure_cartopy(cartopy_data_dir)
    routes = ROUTE_ORDER[1:]
    figure, axes = plt.subplots(
        4, 3, figsize=(12.8, 17.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        label="figureS3_stationwise_CSI",
    )
    row_limits: dict[float, tuple[float, float]] = {}
    for threshold in THRESHOLDS:
        values = station.loc[np.isclose(station.threshold_mm, threshold), "csi_mean"]
        step = 0.05 if float(values.max()) >= 0.2 else 0.025
        lower = max(0.0, float(np.floor(values.min() / step) * step))
        upper = min(1.0, float(np.ceil(values.max() / step) * step))
        if threshold == 5.0:
            upper = 0.50
        row_limits[threshold] = (lower, upper)
    means = station.groupby(["threshold_mm", "route"]).csi_mean.mean()
    letters = iter("abcdefghijkl")
    for row, threshold in enumerate(THRESHOLDS):
        vmin, vmax = row_limits[threshold]
        artist = None
        for col, route in enumerate(routes):
            ax = axes[row, col]
            add_base_map(
                ax,
                ccrs,
                cfeature,
                extent=(124.0, 131.15, 32.8, 38.75),
            )
            # S3 is printed as a tall 4×3 plate, so its geographic labels
            # need to be larger than the single-map Figure 1 defaults.
            # Apply this after map construction to keep the public renderer's
            # base-map call contract simple for authorized-input adapters.
            for gridliner in getattr(ax, "_gridliners", []):
                gridliner.xlabel_style = {"size": 14.0, "color": "#66727D"}
                gridliner.ylabel_style = {"size": 14.0, "color": "#66727D"}
            ax.scatter(
                fitting.lon, fitting.lat, s=3.8, color="#AEB3B7", alpha=0.44,
                linewidths=0, transform=ccrs.PlateCarree(), zorder=1,
                rasterized=True,
            )
            block = station[station.route.eq(route) & np.isclose(station.threshold_mm, threshold)]
            if len(block) != 128:
                raise ValueError(f"expected 128 stations for {route}/{threshold:g} mm")
            artist = ax.scatter(
                block.lon, block.lat, c=block.csi_mean, s=29, cmap="plasma",
                vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.32,
                transform=ccrs.PlateCarree(), zorder=3, rasterized=True,
            )
            ax.text(
                0.018, 0.982, f"({next(letters)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=15.0, fontweight="normal",
            )
            ax.set_title(
                f"{ROUTE_LABELS[route]}\nmean station CSI = "
                f"{means.loc[(threshold, route)]:.3f}",
                fontsize=16.0, fontweight="normal", pad=8, linespacing=1.05,
            )
            if col == 0:
                ax.text(
                    -0.225, 0.5, f"RN60 ≥ {threshold:g} mm", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=16.0,
                    fontweight="bold",
                )
        assert artist is not None
        colorbar_axis = axes[row, 2].inset_axes([1.055, 0.015, 0.035, 0.970])
        colorbar = figure.colorbar(artist, cax=colorbar_axis, orientation="vertical")
        colorbar.set_label("Station-wise CSI", fontsize=14.5)
        colorbar.ax.tick_params(labelsize=14.0)
    figure.subplots_adjust(
        left=0.100,
        right=0.870,
        bottom=0.035,
        top=0.965,
        hspace=0.46,
        wspace=0.20,
    )
    save(figure, output, "figureS3_stationwise_CSI")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="item", required=True)
    p1 = sub.add_parser("figure1")
    p1.add_argument("--coverage-npz", type=Path, required=True)
    p1.add_argument("--stations-csv", type=Path, required=True)
    p1.add_argument("--cartopy-data-dir", type=Path, required=True)
    p1.add_argument("--output-dir", type=Path, required=True)
    p2 = sub.add_parser("s2")
    p2.add_argument("--supplementary-data", type=Path, required=True)
    p2.add_argument("--output-dir", type=Path, required=True)
    p3 = sub.add_parser("s3")
    p3.add_argument("--supplementary-data", type=Path, required=True)
    p3.add_argument("--stations-csv", type=Path, required=True)
    p3.add_argument("--cartopy-data-dir", type=Path, required=True)
    p3.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.item == "figure1":
        render_figure1(
            args.coverage_npz, args.stations_csv, args.cartopy_data_dir,
            args.output_dir,
        )
    elif args.item == "s2":
        render_s2(args.supplementary_data, args.output_dir)
    else:
        render_s3(
            args.supplementary_data, args.stations_csv,
            args.cartopy_data_dir, args.output_dir,
        )


if __name__ == "__main__":
    main()
