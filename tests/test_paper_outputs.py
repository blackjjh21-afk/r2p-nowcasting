from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_outputs.render_publication import (
    render_figure2,
    render_figure3,
    render_figure4,
    render_figure5,
    render_figure6,
    render_figure_s1,
    render_figure_s4,
    render_table1,
)
from paper_outputs import render_publication, render_restricted


ROOT = Path(__file__).resolve().parents[1]


def test_public_aggregate_manifest_has_no_station_resolved_sheets() -> None:
    manifest = json.loads((ROOT / "data/paper_aggregates/manifest.json").read_text(encoding="utf-8"))
    assert manifest["station_resolved_content_included"] is False
    names = {record["name"] for record in manifest["records"] if record["kind"] == "aggregate_sheet"}
    assert "FigS2_station_timeseries" not in names
    assert "FigS3_station_CSI" not in names
    assert "Distance_station_groups" not in names
    excluded_cases = {"TableS2_cases", "Case_definitions", "Case_per_rep", "Case_mean_sd", "FigS2_timeseries_per_member", "FigS2_timeseries_summary"}
    assert names.isdisjoint(excluded_cases)
    for name in excluded_cases:
        assert not (ROOT / "data/paper_aggregates" / f"{name}.csv").exists()
    forbidden = {"station_id", "held-out_station", "held_out_station", "display_station_id", "name", "lat", "lon", "valid_time_kst"}
    for name in names:
        frame = pd.read_csv(ROOT / "data/paper_aggregates" / f"{name}.csv", nrows=0)
        normalized = {column.strip().lower().replace(" ", "_") for column in frame.columns}
        assert normalized.isdisjoint(forbidden)


def test_code_native_figure6_smoke(tmp_path: Path) -> None:
    png, pdf = render_figure6(tmp_path)
    assert png.is_file() and png.stat().st_size > 10_000
    assert pdf.is_file() and pdf.stat().st_size > 1_000


def test_current_cnn_sources_and_validtime_support() -> None:
    data = ROOT / "data/paper_aggregates"
    table = pd.read_csv(data / "Table1_main.csv")
    assert set(table.Route) == {"pySTEPS + CNN", "exPreCast + CNN", "Direct R2P"}
    assert not (data / "TableS1b_readout.csv").exists()
    fb = pd.read_csv(data / "Fig2d_FB_summary.csv")
    assert set(fb.route) == {"fixed_mp", "center_mlp", "patch_cnn"}
    # Panel d uses held-out 2024–2025 support, not panel c's all-station support.
    fixed_d = fb[fb.route.eq("fixed_mp")].sort_values("threshold_mm")
    fixed_c = pd.read_csv(data / "Fig2c_fixed_metrics.csv").sort_values("threshold_mm")
    assert not np.allclose(fixed_d.frequency_bias_mean, fixed_c.frequency_bias)


def test_all_public_aggregate_renderers(tmp_path: Path) -> None:
    data = ROOT / "data/paper_aggregates"
    examples = ROOT / "data/examples"
    renderers = (
        lambda: render_figure2(data, tmp_path),
        lambda: render_figure3(data, tmp_path),
        lambda: render_figure4(data, examples, tmp_path),
        lambda: render_figure5(data, tmp_path),
        lambda: render_figure_s1(data, tmp_path),
        lambda: render_figure_s4(data, tmp_path),
        lambda: render_table1(data, tmp_path),
    )
    for renderer in renderers:
        png, pdf = renderer()
        assert png.is_file() and png.stat().st_size > 5_000
        assert pdf.is_file() and pdf.stat().st_size > 1_000


def test_figure2_panel_d_axis_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inspect_and_close(figure, _output):
        axis = next(
            axis for axis in figure.axes
            if axis.get_ylabel() == "Frequency bias (lines)"
        )
        assert axis.get_ylim() == (0.0, 1.875)
        np.testing.assert_allclose(axis.get_yticks(), np.arange(0.0, 1.21, 0.2))
        # Only headroom changes: value 1 occupies the former value 0.8 height.
        assert 1.0 / axis.get_ylim()[1] == pytest.approx(0.8 / 1.5)
        csi_axis = next(ax for ax in figure.axes if ax.get_ylabel() == "CSI (bars)")
        assert csi_axis.get_ylim() == (0.0, 1.625)
        assert 1.0 / csi_axis.get_ylim()[1] == pytest.approx(0.8 / 1.3)
        render_publication.plt.close(figure)
        return tmp_path / "figure2.png", tmp_path / "figure2.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure2(ROOT / "data/paper_aggregates", tmp_path)


def test_figure_s4_compact_threshold_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inspect_and_close(figure, _output):
        axis = figure.axes[3]
        legend = axis.get_legend()
        assert legend.columnspacing == 1.0
        assert len(legend.get_texts()) == 4
        figure.canvas.draw()
        assert legend.get_window_extent().x0 > axis.get_window_extent().x0
        render_publication.plt.close(figure)
        return tmp_path / "figureS4.png", tmp_path / "figureS4.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure_s4(ROOT / "data/paper_aggregates", tmp_path)


def _station_split() -> pd.DataFrame:
    fitting_id = np.arange(1, 515)
    heldout_id = np.arange(515, 643)
    station_id = np.concatenate((fitting_id, heldout_id))
    return pd.DataFrame(
        {
            "station_id": station_id,
            "lat": 33.2 + (station_id % 35) * 0.15,
            "lon": 124.4 + (station_id % 45) * 0.14,
            "split": ["train"] * 514 + ["test"] * 128,
        }
    )


def test_restricted_figure1_display_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cartopy = pytest.importorskip("cartopy")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # The test exercises the complete footprint/station display transform but
    # replaces the Natural Earth layer, whose files are intentionally not
    # bundled, with a network-free extent-only base map.
    monkeypatch.setattr(render_restricted, "configure_cartopy", lambda _: (ccrs, cfeature))
    monkeypatch.setattr(
        render_restricted,
        "add_base_map",
        lambda ax, _ccrs, _cfeature, *, extent: ax.set_extent(extent, crs=ccrs.PlateCarree()),
    )
    lon_axis = np.linspace(124.0, 131.0, 50)
    lat_axis = np.linspace(33.0, 39.0, 45)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    footprint = ((lon - 127.5) ** 2 / 10 + (lat - 36.0) ** 2 / 7) <= 1
    archive = tmp_path / "coverage.npz"
    np.savez_compressed(archive, lon=lon, lat=lat, footprint=footprint)
    stations = tmp_path / "stations.csv"
    _station_split().to_csv(stations, index=False)
    render_restricted.render_figure1(archive, stations, tmp_path, tmp_path)
    assert (tmp_path / "figure1_study_design.png").stat().st_size > 5_000


def _synthetic_s2_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = tuple(render_restricted.ROUTE_ORDER)
    series_rows = []
    case_rows = []
    for panel, rank in zip("abcd", (1, 4, 7, 8)):
        start = pd.Timestamp("2025-07-01") + pd.Timedelta(days=rank)
        end = start + pd.Timedelta(hours=24)
        times = pd.date_range(start + pd.Timedelta(minutes=10), end, freq="10min")
        case_rows.append({"panel": panel, "case_rank": rank, "station_id": 500 + rank,
                          "window_start_exclusive": start, "window_end_inclusive": end, "lead_min": 60})
        for route_index, route in enumerate(routes):
            for index, valid_time in enumerate(times):
                value = float(rank + route_index + index)
                series_rows.append(
                    {
                        "case_rank": rank,
                        "station_id": 500 + rank,
                        "valid_time_kst": valid_time,
                        "lead_min": 60,
                        "route": route,
                        "value_mean_mm": value,
                        "value_min_mm": max(0.0, value - 0.2),
                        "value_max_mm": value + 0.2,
                    }
                )
    return pd.DataFrame(series_rows), pd.DataFrame(case_rows)


def _write_s2_workbook(path: Path, series: pd.DataFrame, cases: pd.DataFrame) -> None:
    with pd.ExcelWriter(path) as writer:
        series.to_excel(writer, sheet_name="FigS2_station_timeseries", index=False)
        cases.to_excel(writer, sheet_name="Case_definitions", index=False)


def test_restricted_s2_display_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series, cases = _synthetic_s2_frames()
    # A missing observation remains a gap, not an interpolated or zero-filled point.
    series.loc[2, "value_mean_mm"] = np.nan
    workbook = tmp_path / "supplementary.xlsx"
    _write_s2_workbook(workbook, series, cases)
    original_save = render_restricted.save

    def inspect_and_save(figure, *args):
        assert len(figure.axes) == 4
        for axis, case in zip(figure.axes, cases.itertuples(index=False)):
            assert len(axis.lines) == 4
            assert not axis.collections and not axis.patches  # no shading or CSI bars
            assert axis.get_ylabel() == "RN60 (mm)"
            assert axis.get_title().startswith(f"({case.panel})")
            assert axis.lines[0].get_color() == "black"
            for route, line in zip(render_restricted.ROUTE_ORDER, axis.lines):
                expected = series.loc[series.case_rank.eq(case.case_rank) & series.route.eq(route), "value_mean_mm"]
                np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
                assert line.get_linestyle() == "-"
        return original_save(figure, *args)

    monkeypatch.setattr(render_restricted, "save", inspect_and_save)
    render_restricted.render_s2(workbook, tmp_path)
    assert (tmp_path / "figureS2_case_analysis.png").stat().st_size > 5_000
    assert (tmp_path / "figureS2_case_analysis.pdf").stat().st_size > 1_000


@pytest.mark.parametrize("defect", ("missing_time", "wrong_station", "wrong_lead", "wrong_panel", "missing_schema", "infinite_value"))
def test_restricted_s2_rejects_misaligned_cases(tmp_path: Path, defect: str) -> None:
    series, cases = _synthetic_s2_frames()
    if defect == "missing_time":
        series = series.iloc[1:]
    elif defect == "wrong_station":
        series.loc[0, "station_id"] = -1
    elif defect == "wrong_lead":
        series.loc[0, "lead_min"] = 90
    elif defect == "wrong_panel":
        cases.loc[0, "panel"] = "d"
    elif defect == "missing_schema":
        cases = cases.drop(columns="station_id")
    elif defect == "infinite_value":
        series.loc[0, "value_mean_mm"] = np.inf
    workbook = tmp_path / "invalid.xlsx"
    _write_s2_workbook(workbook, series, cases)
    with pytest.raises(ValueError):
        render_restricted.render_s2(workbook, tmp_path)
    assert not (tmp_path / "figureS2_case_analysis.png").exists()


def test_restricted_s3_display_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    monkeypatch.setattr(render_restricted, "configure_cartopy", lambda _: (ccrs, cfeature))
    monkeypatch.setattr(
        render_restricted,
        "add_base_map",
        lambda ax, _ccrs, _cfeature, *, extent: ax.set_extent(extent, crs=ccrs.PlateCarree()),
    )
    split = _station_split()
    stations = tmp_path / "stations.csv"
    split.to_csv(stations, index=False)
    heldout = split[split.split.eq("test")]
    rows = []
    for route_index, route in enumerate(render_restricted.ROUTE_ORDER[1:]):
        for threshold in (1.0, 5.0, 10.0, 20.0):
            for record in heldout.itertuples(index=False):
                rows.append(
                    {
                        "station_id": record.station_id,
                        "lat": record.lat,
                        "lon": record.lon,
                        "route": route,
                        "threshold_mm": threshold,
                        "csi_mean": min(0.75, 0.05 + route_index * 0.03 + threshold / 45 + (record.station_id % 11) / 100),
                    }
                )
    workbook = tmp_path / "supplementary.xlsx"
    pd.DataFrame(rows).to_excel(workbook, sheet_name="FigS3_station_CSI", index=False)
    original_save = render_restricted.save

    def inspect_labels_and_save(figure, *args, **kwargs):
        expected = {f"({letter})" for letter in "abcdefghijkl"}
        labels = [text for axis in figure.axes for text in axis.texts if text.get_text() in expected]
        assert {text.get_text() for text in labels} == expected
        assert len(labels) == 12
        for text in labels:
            x, y = text.get_position()
            assert x < 0 and y > 1
            assert not text.get_clip_on()
            assert text.get_fontsize() >= 15
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(render_restricted, "save", inspect_labels_and_save)
    render_restricted.render_s3(workbook, stations, tmp_path, tmp_path)
    assert (tmp_path / "figureS3_stationwise_CSI.png").stat().st_size > 5_000
