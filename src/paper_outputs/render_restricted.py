#!/usr/bin/env python3
"""Render station-resolved paper figures from author-supplied inputs.

No station-resolved inputs are bundled.  This module documents and implements
the final display transformations for Figures 1, 7 and 8 so
authorized users can reproduce them after supplying the corresponding files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
MAP_NORTH_LAT = 40.4
ELEVATION_LIMITS_M = (100, 1000)
ELEVATION_TICKS_M = tuple(range(100, 1001, 100))
ELEVATION_PALETTE = (
    (0.00, "#FFFFFF"), (0.08, "#F1E5CF"), (0.20, "#DFC49C"),
    (0.40, "#C49C6A"), (0.60, "#A4794E"), (0.80, "#855B39"),
    (1.00, "#664229"),
)
CASE_FONT = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.labelsize": 16, "axes.titlesize": 16, "axes.titleweight": "bold",
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}
CSI_COLORBAR_TICKS = (
    (.3, .4, .5, .6), (.2, .3, .4, .5), (.1, .2, .3, .4),
    (0., .05, .10, .15, .20, .25),
)


def elevation_colour_scale() -> tuple[LinearSegmentedColormap, Normalize]:
    """Use actual elevation: white through 100 m, saturated brown from 1000 m."""
    cmap = LinearSegmentedColormap.from_list("brown_elevation", ELEVATION_PALETTE)
    cmap.set_under("#FFFFFF")
    cmap.set_over(ELEVATION_PALETTE[-1][1])
    return cmap, Normalize(*ELEVATION_LIMITS_M)


def load_terrain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read caller-supplied elevation in metres, without downloading anything.

    NPZ uses 1-D ``lon``/``lat`` and 2-D ``elevation_m``; ETOPO NetCDF uses
    ``longitude``/``latitude`` and ``z``. Elevation dimensions must be lat × lon.
    No interpolation, hillshading, or transformation of elevation is applied.
    """
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            required = {"lon", "lat", "elevation_m"}
            if not required.issubset(archive.files):
                raise ValueError("terrain NPZ must contain lon, lat and elevation_m")
            lon, lat, elevation = (np.asarray(archive[key]) for key in ("lon", "lat", "elevation_m"))
    elif Path(path).suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            if not {"longitude", "latitude", "z"}.issubset(dataset.variables):
                raise ValueError("terrain NetCDF must contain longitude, latitude and z")
            lon, lat = dataset.longitude.values, dataset.latitude.values
            elevation = dataset.z.transpose("latitude", "longitude").values
    else:
        raise ValueError("terrain file must be NPZ or ETOPO-compatible NetCDF")
    if (lon.ndim != 1 or lat.ndim != 1 or min(lon.size, lat.size) < 2
            or elevation.shape != (lat.size, lon.size)):
        raise ValueError("terrain must have 1-D coordinates and lat × lon elevation")
    if not np.isfinite(lon).all() or not np.isfinite(lat).all() or not np.isfinite(elevation).all():
        raise ValueError("terrain coordinates and elevation must be finite")
    for coordinate in (lon, lat):
        difference = np.diff(coordinate)
        if not ((difference > 0).all() or (difference < 0).all()):
            raise ValueError("terrain coordinates must be strictly monotonic")
    return lon, lat, elevation


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
) -> None:
    """Geographic context used by the latest compact stationwise CSI plate."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("#eaf3f8")
    background = (
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#eaf3f8", zorder=0),
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f5f5f2", zorder=0),
        ax.add_feature(
            cfeature.BORDERS.with_scale("10m"), edgecolor="#888888",
            linewidth=0.40, zorder=1.5,
        ),
        ax.coastlines(resolution="10m", color="#555555", linewidth=0.48, zorder=1.5),
    )
    # Repeating the detailed Natural Earth polygons in every Figure 8 panel would
    # otherwise produce a very large vector PDF.  The point symbols and text
    # remain vector while the geographic context layer is rasterized.
    for artist in background:
        artist.set_rasterized(True)


def add_terrain_map(ax, ccrs, cfeature, *, extent, lon, lat, elevation):
    """Draw Figure 1 geographic layers and return its elevation artist."""
    projection = ccrs.PlateCarree()
    ax.set_facecolor("#F0F6F9")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", zorder=0)
    cmap, norm = elevation_colour_scale()
    terrain = ax.pcolormesh(
        lon, lat, np.ma.masked_less(elevation, 0), cmap=cmap, norm=norm,
        shading="auto", transform=projection, rasterized=True, zorder=1,
    )
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#F0F6F9", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#85817A", linewidth=.45, zorder=4)
    ax.coastlines(resolution="10m", color="#53534F", linewidth=.60, zorder=5)
    ax.set_extent(extent, crs=projection)
    grid = ax.gridlines(crs=projection, draw_labels=True, linewidth=.4,
                       color="#929A9E", alpha=.5, linestyle=":", zorder=5)
    grid.top_labels = grid.right_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 11.5, "color": "#4B5358"}
    return terrain


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
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white",
                metadata={"Date": None})
    plt.close(fig)


def render_figure1(
    coverage_npz: Path,
    stations_csv: Path,
    cartopy_data_dir: Path,
    output: Path,
    *,
    terrain_file: Path,
) -> None:
    """Render the domain map from an authorized footprint and station split.

    ``coverage_npz`` must contain 2-D ``lon``, ``lat`` and boolean ``footprint``
    arrays and may retain an ``extent`` computed before display-grid decimation.
    The station CSV must contain ``lon``, ``lat`` and ``split`` with 514 ``train``
    and 128 ``test`` rows. ``terrain_file`` is required; see :func:`load_terrain`.
    """

    with np.load(coverage_npz, allow_pickle=False) as archive:
        if set(("lon", "lat", "footprint")) - set(archive.files):
            raise ValueError("coverage archive must contain lon, lat and footprint")
        lon, lat, footprint = archive["lon"], archive["lat"], archive["footprint"].astype(bool)
        saved_extent = archive["extent"] if "extent" in archive.files else None
    if lon.ndim != 2 or lon.shape != lat.shape or lon.shape != footprint.shape:
        raise ValueError("coverage arrays must share one 2-D shape")
    if not footprint.any() or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("coverage requires finite coordinates and a nonempty footprint")
    stations = pd.read_csv(stations_csv, encoding="utf-8-sig")
    required = {"lon", "lat", "split"}
    if not required.issubset(stations):
        raise ValueError(f"station split lacks {sorted(required - set(stations))}")
    fitting, heldout = stations[stations.split.eq("train")], stations[stations.split.eq("test")]
    if (len(fitting), len(heldout)) != (514, 128):
        raise ValueError("expected the frozen 514/128 split")
    covered_lon, covered_lat = lon[footprint], lat[footprint]
    extent = np.array(saved_extent if saved_extent is not None else (
        float(np.nanmin(covered_lon)) - 0.25,
        float(np.nanmax(covered_lon)) + 0.25,
        float(np.nanmin(covered_lat)) - 0.25,
        float(np.nanmax(covered_lat)) + 0.25,
    ), dtype=float)
    if extent.shape != (4,) or not np.isfinite(extent).all():
        raise ValueError("coverage extent must contain four finite limits")
    extent[3] = MAP_NORTH_LAT
    if extent[0] >= extent[1] or extent[2] >= extent[3]:
        raise ValueError("coverage extent must have increasing longitude/latitude bounds")
    terrain_lon, terrain_lat, elevation = load_terrain(terrain_file)
    if not (terrain_lon.min() < extent[0] < extent[1] < terrain_lon.max()
            and terrain_lat.min() < extent[2] < extent[3] < terrain_lat.max()):
        raise ValueError("terrain must cover the complete displayed domain")
    if not np.isfinite(stations[["lon", "lat"]].to_numpy()).all():
        raise ValueError("every station must have finite map coordinates")
    if not (stations.lon.between(extent[0], extent[1]).all()
            and stations.lat.between(extent[2], extent[3]).all()):
        raise ValueError("displayed map must retain all fitting and held-out stations")
    ccrs, cfeature = configure_cartopy(cartopy_data_dir)
    projection = ccrs.PlateCarree()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5,
                         "axes.linewidth": .8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8.8, 8.0), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    terrain = add_terrain_map(ax, ccrs, cfeature, extent=extent,
                              lon=terrain_lon, lat=terrain_lat, elevation=elevation)
    stride = max(1, int(np.ceil(max(lon.shape) / 600)))
    sl = (slice(None, None, stride), slice(None, None, stride))
    ax.contour(
        lon[sl], lat[sl], footprint[sl].astype(int), levels=[0.5],
        colors=["#2C728A"], linewidths=1.0, transform=projection, zorder=3,
    )
    ax.scatter(
        fitting.lon, fitting.lat, s=6.5, c="#1984CF", edgecolors="black",
        linewidths=.22, transform=projection, zorder=6,
    )
    ax.scatter(
        heldout.lon, heldout.lat, s=36, c="#D62728", marker="^",
        edgecolors="white", linewidths=0.40,
        transform=projection, zorder=7,
    )
    handles = [
        Line2D([], [], color="#2C728A", linewidth=1.0, label="HSR composite coverage"),
        Line2D([], [], marker="o", ls="", markerfacecolor="#1984CF", markeredgecolor="black",
               markeredgewidth=.22, markersize=4, label="Fitting stations (n=514)"),
        Line2D([], [], marker="^", ls="", markerfacecolor="#D62728", markeredgecolor="white",
               markeredgewidth=.40, markersize=7, label="Held-out stations (n=128)"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=.97, facecolor="white",
              edgecolor="#C7CBCE", fontsize=11.5, borderpad=.65, handlelength=2.0)
    colorbar = fig.colorbar(terrain, ax=ax, orientation="vertical", pad=.035,
                           fraction=.037, shrink=.82, extend="both", ticks=ELEVATION_TICKS_M)
    colorbar.set_label("Elevation (m)", labelpad=9, fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    colorbar.outline.set_linewidth(.6)
    fig.subplots_adjust(left=.075, right=.915, bottom=.055, top=.985)
    save(fig, output, "figure1_study_domain")


def read_source_csv(source_data: Path, name: str, *legacy_names: str) -> pd.DataFrame:
    """Prefer the current table name, accepting explicitly named legacy exports."""
    for candidate in (name, *legacy_names):
        path = source_data / f"{candidate}.csv"
        if path.is_file():
            return pd.read_csv(path, encoding="utf-8-sig")
    raise FileNotFoundError(f"caller-supplied source table is missing: {source_data / (name + '.csv')}")


def render_figure7(source_data: Path, output: Path) -> None:
    """Render four +60-min station examples, without CSI panels or shading.

    Case definitions and station observations must be supplied by the caller;
    the public repository does not bundle station-resolved case metadata.
    """
    series = read_source_csv(source_data, "Fig7_station_timeseries", "FigS2_station_timeseries")
    cases = read_source_csv(source_data, "Case_definitions")
    required_series = {"case_rank", "station_id", "valid_time_kst", "lead_min", "route", "value_mean_mm"}
    required_cases = {"panel", "case_rank", "station_id", "window_start_exclusive", "window_end_inclusive", "lead_min"}
    if not required_series.issubset(series) or not required_cases.issubset(cases):
        raise ValueError("Figure 7 source CSVs lack the four-case station-time schema")
    cases = cases.sort_values("panel").reset_index(drop=True)
    if list(cases.panel) != list("abcd") or not cases.case_rank.is_unique:
        raise ValueError("Figure 7 requires four distinct cases with panels a, b, c and d")
    if not cases.lead_min.eq(60).all() or not series.lead_min.eq(60).all():
        raise ValueError("Figure 7 time series must use +60-min forecasts")
    if set(series.case_rank) != set(cases.case_rank):
        raise ValueError("Figure 7 case definitions and time series do not match")
    series["valid_time_kst"] = pd.to_datetime(series.valid_time_kst)
    validated = []
    for case in cases.itertuples(index=False):
        start, end = pd.Timestamp(case.window_start_exclusive), pd.Timestamp(case.window_end_inclusive)
        if end - start != pd.Timedelta(hours=24):
            raise ValueError("Each Figure 7 case must span exactly 24 h")
        expected_times = pd.date_range(start + pd.Timedelta(minutes=10), end, freq="10min")
        blocks = []
        case_rows = series[series.case_rank.eq(case.case_rank)]
        if set(case_rows.route) != set(ROUTE_ORDER):
            raise ValueError(f"case {case.case_rank} lacks the exact four plotted series")
        for route in ROUTE_ORDER:
            block = case_rows[case_rows.route.eq(route)].sort_values("valid_time_kst")
            if not block.station_id.eq(case.station_id).all():
                raise ValueError(f"case {case.case_rank} contains the wrong display station")
            if not pd.DatetimeIndex(block.valid_time_kst).equals(expected_times):
                raise ValueError(f"case {case.case_rank}, route {route} lacks exact common valid times")
            values = block.value_mean_mm.to_numpy(dtype=float)
            if np.isinf(values).any() or not np.isfinite(values).any():
                raise ValueError(f"case {case.case_rank}, route {route} has invalid plotted values")
            blocks.append((route, block))
        validated.append((case, start, end, blocks))

    plt.rcParams.update(CASE_FONT)
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), label="figure7_case_timeseries")
    for ax, (case, start, end, blocks) in zip(axes.flat, validated):
        for route, block in blocks:
            ax.plot(block.valid_time_kst, block.value_mean_mm,
                    color="black" if route == "truth" else ROUTE_COLORS[route],
                    lw=2.0 if route in ("truth", "direct_r2p") else 1.55,
                    ls="-", label=ROUTE_LABELS[route])
        if (start.year, start.month) == (end.year, end.month):
            date_label = f"{start.day}–{end.day} {start:%B %Y}"
        else:
            date_label = f"{start:%d %b %Y}–{end:%d %b %Y}"
        event_label = getattr(case, "event_label", None)
        if isinstance(event_label, str) and event_label.strip():
            date_label = event_label
        ax.set_ylabel("RN60 (mm)")
        ax.set_title(f"({case.panel}) {date_label}: station {int(case.station_id)}",
                     fontsize=16, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.0),
                  frameon=False, fontsize=15)
    figure.tight_layout(h_pad=2.4, w_pad=1.5, rect=(0, 0, 1, .945))
    save(figure, output, "figure7_case_timeseries")


def render_figure8(
    source_data: Path,
    stations_csv: Path,
    cartopy_data_dir: Path,
    output: Path,
) -> None:
    station = read_source_csv(source_data, "Fig8_station_CSI", "FigS3_station_CSI")
    required = {"station_id", "lat", "lon", "route", "threshold_mm", "csi_mean"}
    if not required.issubset(station):
        raise ValueError("Fig8_station_CSI does not match the frozen schema")
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
        raise ValueError("Figure 8 station IDs differ from the held-out station split")

    if (set(station.route) != set(ROUTE_ORDER[1:])
            or set(station.threshold_mm) != set(THRESHOLDS)
            or station.duplicated(["station_id", "route", "threshold_mm"]).any()):
        raise ValueError("Figure 8 requires one row per held-out station, route and threshold")
    if (not np.isfinite(station[["lon", "lat", "csi_mean"]].to_numpy()).all()
            or not station.csi_mean.between(0, 1).all()):
        raise ValueError("Figure 8 requires finite coordinates and CSI values within [0, 1]")
    coordinates = station.merge(heldout[["station_id", "lon", "lat"]], on="station_id",
                                validate="many_to_one", suffixes=("", "_split"))
    if not np.allclose(coordinates[["lon", "lat"]], coordinates[["lon_split", "lat_split"]],
                       rtol=0, atol=1e-6):
        raise ValueError("Figure 8 coordinates disagree with the held-out station split")

    ccrs, cfeature = configure_cartopy(cartopy_data_dir)
    routes = ROUTE_ORDER[1:]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 10, "pdf.fonttype": 42,
                         "axes.titleweight": "normal", "svg.fonttype": "none"})
    figure, axes = plt.subplots(
        4, 3, figsize=(12.8, 13.2),
        subplot_kw={"projection": ccrs.PlateCarree()},
        label="figure8_stationwise_csi",
    )
    row_limits: dict[float, tuple[float, float]] = {}
    for threshold in THRESHOLDS:
        values = station.loc[np.isclose(station.threshold_mm, threshold), "csi_mean"]
        step = 0.025
        lower = max(0.0, float(np.floor(values.min() / step) * step))
        upper = min(1.0, float(np.ceil(values.max() / step) * step))
        if upper <= lower:
            upper = min(1.0, lower + step)
            lower = min(lower, upper - step)
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
            ax.set_title(
                f"({next(letters)}) {ROUTE_LABELS[route]}",
                fontsize=14.5, fontweight="bold",
            )
            ax.text(.5, -.055, f"mean station CSI = {means.loc[(threshold, route)]:.3f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=14.5,
                    fontweight="normal", clip_on=False)
            if col == 0:
                ax.text(
                    -0.20, 0.5, f"RN60 ≥ {threshold:g} mm", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=15.5,
                    fontweight="bold",
                )
        assert artist is not None
        colorbar_axis = axes[row, 2].inset_axes([1.055, 0.015, 0.035, 0.970])
        colorbar = figure.colorbar(artist, cax=colorbar_axis, orientation="vertical")
        colorbar.set_ticks(CSI_COLORBAR_TICKS[row])
        colorbar.set_label("CSI", fontsize=16)
        colorbar.ax.tick_params(labelsize=14.5)
    figure.subplots_adjust(
        left=0.100,
        right=0.870,
        bottom=0.045,
        top=0.965,
        hspace=0.20,
        wspace=0.20,
    )
    save(figure, output, "figure8_stationwise_csi")


# Legacy call names accept earlier exports, but always emit current figure names.
render_s2 = render_figure7
render_s3 = render_figure8


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="item", required=True)
    p1 = sub.add_parser("figure1")
    p1.add_argument("--coverage-npz", type=Path, required=True)
    p1.add_argument("--stations-csv", type=Path, required=True)
    p1.add_argument("--cartopy-data-dir", type=Path, required=True)
    p1.add_argument("--terrain-file", "--terrain-npz", "--terrain-nc", type=Path, required=True,
                    help="caller-supplied elevation NPZ or ETOPO-compatible NetCDF")
    p1.add_argument("--output-dir", type=Path, required=True)
    p2 = sub.add_parser("figure7", aliases=["s2"])
    p2.add_argument("--source-data", type=Path, required=True,
                    help="directory containing authorized Figure 7 source CSVs")
    p2.add_argument("--output-dir", type=Path, required=True)
    p3 = sub.add_parser("figure8", aliases=["s3"])
    p3.add_argument("--source-data", type=Path, required=True,
                    help="directory containing authorized Figure 8 source CSVs")
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
            terrain_file=args.terrain_file,
        )
    elif args.item in {"figure7", "s2"}:
        render_figure7(args.source_data, args.output_dir)
    else:
        render_figure8(
            args.source_data, args.stations_csv,
            args.cartopy_data_dir, args.output_dir,
        )


if __name__ == "__main__":
    main()
