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
    render_figure9,
    render_figure10,
    render_figure_s1,
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
    superseded = {"Table1_numeric", "Fig2d_CSI_summary", "Fig2d_FB_summary"}
    assert names.isdisjoint(superseded)
    assert {"Table1_main", "Fig2d_readout_summary"}.issubset(names)
    for name in superseded:
        assert not (ROOT / "data/paper_aggregates" / f"{name}.csv").exists()
    forbidden = {"station_id", "held-out_station", "held_out_station", "display_station_id", "name", "lat", "lon", "valid_time_kst"}
    for name in names:
        frame = pd.read_csv(ROOT / "data/paper_aggregates" / f"{name}.csv", nrows=0)
        normalized = {column.strip().lower().replace(" ", "_") for column in frame.columns}
        assert normalized.isdisjoint(forbidden)


def test_architecture_figure2_smoke(tmp_path: Path) -> None:
    png, pdf = render_figure2(tmp_path)
    assert png.is_file() and png.stat().st_size > 10_000
    assert pdf.is_file() and pdf.stat().st_size > 1_000


def test_current_cnn_sources_and_validtime_support() -> None:
    data = ROOT / "data/paper_aggregates"
    table = pd.read_csv(data / "Table1_main.csv")
    assert set(table.Route) == {"pySTEPS + CNN", "exPreCast + CNN", "Direct R2P"}
    assert all(pd.api.types.is_numeric_dtype(table[column]) for column in table.columns[1:])
    # The only Table 1 source keeps full precision; rounding belongs to display code.
    assert table["RN60 CSI-M"].ne(table["RN60 CSI-M"].round(4)).any()
    assert not (data / "TableS1b_readout.csv").exists()
    fb = pd.read_csv(data / "Fig2d_readout_summary.csv")
    assert list(fb.columns) == [
        "route", "threshold_mm", "csi_mean", "csi_sd",
        "frequency_bias_mean", "frequency_bias_sd", "n_members",
    ]
    assert set(fb.route) == {"fixed_mp", "center_mlp", "patch_cnn"}
    assert len(fb) == 12 and not fb.duplicated(["route", "threshold_mm"]).any()
    for route, block in fb.groupby("route"):
        assert set(block.threshold_mm) == {1, 5, 10, 20}
        assert set(block.n_members) == ({1} if route == "fixed_mp" else {3})
    # Panel d uses held-out 2024–2025 support, not panel c's all-station support.
    fixed_d = fb[fb.route.eq("fixed_mp")].sort_values("threshold_mm")
    fixed_c = pd.read_csv(data / "Fig2c_fixed_metrics.csv").sort_values("threshold_mm")
    assert not np.allclose(fixed_d.frequency_bias_mean, fixed_c.frequency_bias)


def test_table1_numeric_source_preserves_display_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = [
        "Route", "Lead (min)", "RN60 CSI-M", "CSI ≥1 mm", "CSI ≥5 mm",
        "CSI ≥10 mm", "CSI ≥20 mm", "RMSE (mm)", "Mean bias (mm)", "Correlation",
    ]
    source = pd.DataFrame([[
        "pySTEPS + CNN", 60.0, 0.1, 0.987654321, 0.0000499,
        0.123456, 0.5, 1.23456, -0.00567, 0.7,
    ]], columns=columns)
    expected = [
        "pySTEPS + CNN", "60", "0.1000", "0.9877", "0.0000",
        "0.1235", "0.5000", "1.235", "-0.006", "0.700",
    ]
    data = tmp_path / "data"
    data.mkdir()
    source_path = data / "Table1_main.csv"
    source.to_csv(source_path, index=False)
    source_bytes = source_path.read_bytes()
    output = tmp_path / "output"

    def inspect_and_close(figure, _output):
        table = figure.axes[0].tables[0]
        assert [table[(1, column)].get_text().get_text() for column in range(10)] == expected
        assert [table[(0, column)].get_text().get_text() for column in range(10)] == [*columns[:-1], "Pearson r"]
        render_publication.plt.close(figure)
        return output / "table1.png", output / "table1.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_table1(data, output)
    display = pd.read_csv(output / "table1_heldout_point_skill.csv", dtype=str)
    assert display.columns[-1] == "Pearson r"
    assert display.iloc[0].tolist() == expected
    markdown_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in (output / "table1_heldout_point_skill.md").read_text().splitlines()
    ]
    assert markdown_rows[2] == expected
    assert source_path.read_bytes() == source_bytes


def test_figure3_panel_d_uses_merged_readout_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_order = ("fixed_mp", "center_mlp", "patch_cnn")
    rows = []
    for route_index, route in enumerate(route_order):
        for threshold_index, threshold in enumerate((1, 5, 10, 20)):
            rows.append({
                "route": route, "threshold_mm": threshold,
                "csi_mean": 0.1 + 0.2 * route_index + 0.01 * threshold_index,
                "csi_sd": 0.001 * route_index,
                "frequency_bias_mean": 0.4 + 0.1 * route_index + 0.02 * threshold_index,
                "frequency_bias_sd": 0.01 * route_index,
                "n_members": 1 if route == "fixed_mp" else 3,
            })
    merged = pd.DataFrame(rows[::-1])  # Route/threshold sorting must be explicit.
    frames = {
        "Fig2a_density": pd.DataFrame({
            "gauge_log1p": [0.1, 0.2], "fixed_readout_log1p": [0.08, 0.15],
            "pair_count": [2, 3],
        }),
        "Fig2a_summary": pd.DataFrame({"pearson_r": [0.8]}),
        "Fig2b_amount_bins": pd.DataFrame({
            "gauge_bin": ["1–5", "5–10"],
            "fixed_readout_to_gauge_amount_ratio": [0.9, 0.6],
            "gauge_conditioned_amount_bias_mm": [-0.1, -0.3],
        }),
        "Fig2c_fixed_metrics": pd.DataFrame({
            "threshold_mm": [1, 5, 10, 20], "frequency_bias": [0.7, 0.5, 0.3, 0.1],
        }),
        "Fig2d_readout_summary": merged,
    }
    requested = []

    def read_frame(_root, name):
        requested.append(name)
        return frames[name].copy()  # Obsolete split-table requests fail immediately.

    def inspect_and_close(figure, _output):
        csi_axis = next(ax for ax in figure.axes if ax.get_ylabel() == "CSI (bars)")
        fb_axis = next(ax for ax in figure.axes if ax.get_ylabel() == "Frequency bias (lines)")
        ordered = [merged[merged.route.eq(route)].sort_values("threshold_mm") for route in route_order]
        np.testing.assert_allclose(
            [bar.get_height() for bar in csi_axis.patches],
            np.concatenate([block.csi_mean.to_numpy() for block in ordered]),
        )
        csi_errors = [container for container in csi_axis.containers if hasattr(container, "has_yerr")]
        for block, bars, lines in zip(ordered, csi_errors, fb_axis.containers, strict=True):
            np.testing.assert_allclose(
                np.asarray(lines.lines[0].get_ydata(), dtype=float), block.frequency_bias_mean,
            )
            for container, mean, sd in (
                (bars, block.csi_mean, block.csi_sd),
                (lines, block.frequency_bias_mean, block.frequency_bias_sd),
            ):
                segments = np.asarray(container.lines[2][0].get_segments(), dtype=float)
                np.testing.assert_allclose(segments[:, :, 1], np.column_stack((mean - sd, mean + sd)))
        render_publication.plt.close(figure)
        return tmp_path / "figure3.png", tmp_path / "figure3.pdf"

    monkeypatch.setattr(render_publication, "read", read_frame)
    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure3(tmp_path, tmp_path)
    assert requested == list(frames)


def test_all_public_aggregate_renderers(tmp_path: Path) -> None:
    data = ROOT / "data/paper_aggregates"
    examples = ROOT / "data/examples"
    renderers = (
        lambda: render_figure3(data, tmp_path),
        lambda: render_figure4(data, tmp_path),
        lambda: render_figure5(data, tmp_path),
        lambda: render_figure9(data, tmp_path),
        lambda: render_figure10(data, examples, tmp_path),
        lambda: render_figure_s1(data, tmp_path),
        lambda: render_table1(data, tmp_path),
    )
    for renderer in renderers:
        png, pdf = renderer()
        assert png.is_file() and png.stat().st_size > 5_000
        assert pdf.is_file() and pdf.stat().st_size > 1_000


def test_figure10_masking_lines_and_dropout_bars_preserve_values_and_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.container import BarContainer, ErrorbarContainer

    examples = ROOT / "data/examples"
    radar = pd.read_csv(examples / "radar_only_csi_by_lead_threshold.csv")
    dropout = pd.read_csv(examples / "station_dropout_csi_by_lead_threshold.csv")

    def inspect_and_close(figure, _output):
        left, right = figure.axes
        assert figure.get_label() == "figure10_r2p_ablations"
        assert list(figure.get_size_inches()) == [12.2, 5.2]
        assert left._left_title.get_fontsize() == 15 and left._left_title.get_fontweight() == "bold"
        assert left.xaxis.label.get_fontsize() == 14
        assert right._left_title.get_text() == "(b) Station-dropout − No-station-dropout"
        assert not left.patches
        assert not any(isinstance(c, BarContainer) for c in left.containers)
        lines = [c for c in left.containers if isinstance(c, ErrorbarContainer)]
        bars = [c for c in right.containers if isinstance(c, BarContainer)]
        bar_errors = [c for c in right.containers if isinstance(c, ErrorbarContainer)]
        assert len(lines) == len(bars) == len(bar_errors) == 4
        for threshold, line, bar, bar_error in zip(
            render_publication.THRESHOLDS, lines, bars, bar_errors, strict=True,
        ):
            series = radar[radar.threshold_mm.eq(threshold)].set_index("lead_min").loc[list(render_publication.LEADS)]
            control = dropout[dropout.threshold_mm.eq(threshold)].set_index("lead_min").loc[list(render_publication.LEADS)]
            artist = line.lines[0]
            assert artist.get_linestyle() == "-" and artist.get_marker() == "o"
            assert artist.get_color() == render_publication.THRESHOLD_COLORS[threshold]
            np.testing.assert_array_equal(artist.get_xdata(), render_publication.LEADS)
            np.testing.assert_array_equal(artist.get_ydata(), series.delta_csi_radar_only_minus_standard)
            np.testing.assert_array_equal([b.get_height() for b in bar], control.delta_csi_station_dropout_minus_no_station_dropout)
            for errors, frame in ((line, series), (bar_error, control)):
                endpoints = np.asarray(errors.lines[2][0].get_segments())[:, :, 1]
                np.testing.assert_allclose(endpoints, np.column_stack((frame.ci_low, frame.ci_high)), atol=1e-17, rtol=1e-14)
        assert left.get_ylim() == (-0.01, 0.01)
        assert any(t.get_text() == "max |ΔCSI| = 0.0029" for t in left.texts)
        render_publication.plt.close(figure)
        return tmp_path / "figure10.png", tmp_path / "figure10.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure10(ROOT / "data/paper_aggregates", examples, tmp_path)


def test_figure3_panel_d_axis_spacing(
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
        assert figure.axes[0].get_ylabel() == "4-km HSR fixed-readout\nRN60 (mm)"
        assert figure.get_label() == "figure3_valid_time_readout"
        for panel in figure.axes[:4]:
            title = panel.title if panel.get_title() else panel._left_title
            assert title.get_fontweight() == "bold"
        return tmp_path / "figure3.png", tmp_path / "figure3.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure3(ROOT / "data/paper_aggregates", tmp_path)


def test_figure_s1_compact_threshold_legend(
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
        assert figure.get_label() == "figureS1_native_field_verification"
        return tmp_path / "figureS1.png", tmp_path / "figureS1.pdf"

    monkeypatch.setattr(render_publication, "save", inspect_and_close)
    render_figure_s1(ROOT / "data/paper_aggregates", tmp_path)


def test_figure4_current_headings_preserve_all_bar_values_and_intervals(tmp_path, monkeypatch):
    from matplotlib.container import BarContainer, ErrorbarContainer

    data = ROOT / "data/paper_aggregates"
    absolute = render_publication.read(data, "Categorical_mean_sd")
    contrasts = render_publication.read(data, "Fig3ef_route_contrasts")

    def inspect(figure, _output):
        assert figure.get_label() == "figure4_route_skill_and_contrasts"
        assert list(figure.get_size_inches()) == [11.8, 10.9]
        assert figure.axes[0].get_subplotspec().get_gridspec().hspace == .84
        for panel, (axis, threshold) in enumerate(zip(figure.axes[:4], render_publication.THRESHOLDS)):
            heading = axis.texts[0]
            assert heading.get_text() == f"({chr(97+panel)}) RN60 ≥ {threshold:g} mm"
            assert heading.get_fontweight() == "bold" and heading.get_ha() == "center"
            assert heading.get_position() == (.5, 1.045) and heading.get_bbox_patch() is None
            bars = [c for c in axis.containers if isinstance(c, BarContainer)]
            for route, container in zip(render_publication.ROUTES, bars):
                expected = absolute.loc[absolute.route_key.eq(route) & absolute.threshold_mm.eq(threshold)].set_index("lead_min").loc[list(render_publication.LEADS)]
                np.testing.assert_array_equal([bar.get_height() for bar in container], expected.csi_mean)
        for axis, name, heading in zip(figure.axes[4:], ("exprecast_minus_pysteps", "direct_minus_exprecast"),
                                      ("(e) exPreCast + CNN\n− pySTEPS + CNN", "(f) Direct R2P\n− exPreCast + CNN")):
            assert axis.texts[0].get_text() == heading
            assert axis.texts[0].get_fontweight() == "bold"
            bars = [c for c in axis.containers if isinstance(c, BarContainer)]
            errors = [c for c in axis.containers if isinstance(c, ErrorbarContainer)]
            for threshold, container, error in zip(render_publication.THRESHOLDS, bars, errors):
                expected = contrasts.loc[contrasts.contrast.eq(name) & contrasts.threshold_mm.eq(threshold)].set_index("lead_min").loc[list(render_publication.LEADS)]
                np.testing.assert_array_equal([bar.get_height() for bar in container], expected.delta_csi)
                np.testing.assert_allclose(np.asarray(error.lines[2][0].get_segments())[:, :, 1],
                                           np.column_stack((expected.ci_low, expected.ci_high)), atol=1e-17)
        render_publication.plt.close(figure)
        return tmp_path / "figure4.png", tmp_path / "figure4.pdf"

    monkeypatch.setattr(render_publication, "save", inspect)
    render_figure4(data, tmp_path)


def test_figure5_current_large_type_and_both_row_leads_preserve_curves(tmp_path, monkeypatch):
    data = ROOT / "data/paper_aggregates"
    source = render_publication.read(data, "Fig5_plot_source")

    def inspect(figure, _output):
        assert list(figure.get_size_inches()) == [18., 10.5]
        assert figure.legends[0]._ncols == 2
        assert figure.axes[0].get_ylabel() == "Gauge-conditioned\namount bias (mm)"
        for row, metric in enumerate(("conditional_mean_error", "frequency_bias")):
            for col, lead in enumerate((60, 120, 180)):
                axis = figure.axes[row*3+col]
                assert axis.get_title() == f"{lead}-min lead"
                assert axis.title.get_fontweight() == "bold"
                assert axis.xaxis.label.get_fontsize() == 21.75
                assert axis.texts[0].get_text() == f"({chr(97+row*3+col)})"
                for line, route in zip(axis.lines[:4], ("observed_hsr", *render_publication.ROUTES)):
                    expected = source.loc[source.route.eq(route) & source.metric.eq(metric) & source.lead_min.eq(lead)].sort_values("x_index")
                    np.testing.assert_array_equal(line.get_ydata(), expected.estimate)
        render_publication.plt.close(figure)
        return tmp_path / "figure5.png", tmp_path / "figure5.pdf"

    monkeypatch.setattr(render_publication, "save", inspect)
    render_figure5(data, tmp_path)


def test_figure9_sequential_lead_colors_preserve_eighty_values_and_intervals(tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba
    from matplotlib.container import BarContainer, ErrorbarContainer

    data = ROOT / "data/paper_aggregates"
    source = render_publication.read(data, "FigS1_distance_cells")
    assert tuple(render_publication.LEAD_COLORS.values()) == ("#A6CEE3", "#6BAED6", "#2C7FB8", "#1F5A99", "#163B65")

    def inspect(figure, _output):
        assert figure.get_label() == "figure9_allstation_distance"
        for axis, threshold in zip(figure.axes, render_publication.THRESHOLDS):
            assert axis._left_title.get_fontweight() == "bold"
            bars = [c for c in axis.containers if isinstance(c, BarContainer)]
            errors = [c for c in axis.containers if isinstance(c, ErrorbarContainer)]
            for lead, container, error in zip(render_publication.LEADS, bars, errors):
                expected = source.loc[source["Lead (min)"].eq(lead) & source["RN60 threshold (mm)"].eq(threshold)].sort_values("Distance group")
                np.testing.assert_array_equal([bar.get_height() for bar in container], expected["ΔCSI (All-station − Direct)"])
                assert all(bar.get_facecolor() == to_rgba(render_publication.LEAD_COLORS[lead], .92) for bar in container)
                np.testing.assert_allclose(np.asarray(error.lines[2][0].get_segments())[:, :, 1],
                                           expected[["95% CI low", "95% CI high"]], atol=1e-17)
        render_publication.plt.close(figure)
        return tmp_path / "figure9.png", tmp_path / "figure9.pdf"

    monkeypatch.setattr(render_publication, "save", inspect)
    render_figure9(data, tmp_path)


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
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # The test exercises the complete footprint/station display transform but
    # replaces the Natural Earth layer, whose files are intentionally not
    # bundled, with a network-free extent-only base map.
    monkeypatch.setattr(render_restricted, "configure_cartopy", lambda _: (ccrs, cfeature))
    def synthetic_terrain(ax, _ccrs, _cfeature, *, extent, lon, lat, elevation):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        cmap, norm = render_restricted.elevation_colour_scale()
        return ax.pcolormesh(lon, lat, np.ma.masked_less(elevation, 0),
                            cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    monkeypatch.setattr(render_restricted, "add_terrain_map", synthetic_terrain)
    lon_axis = np.linspace(124.0, 131.0, 50)
    lat_axis = np.linspace(33.0, 39.0, 45)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    footprint = ((lon - 127.5) ** 2 / 10 + (lat - 36.0) ** 2 / 7) <= 1
    archive = tmp_path / "coverage.npz"
    np.savez_compressed(archive, lon=lon, lat=lat, footprint=footprint,
                        extent=[124, 131, 32.8, 41])
    terrain = tmp_path / "terrain.npz"
    elevation = np.linspace(-100, 1500, 20 * 30).reshape(20, 30)
    np.savez_compressed(terrain, lon=np.linspace(123, 132, 30),
                        lat=np.linspace(32, 41, 20), elevation_m=elevation)
    stations = tmp_path / "stations.csv"
    _station_split().to_csv(stations, index=False)
    original_save = render_restricted.save

    def inspect_and_save(figure, *args):
        axis = figure.axes[0]
        assert figure.get_size_inches().tolist() == [8.8, 8.0]
        np.testing.assert_allclose(axis.get_extent(), [124, 131, 32.8, 40.4])
        shown = axis.collections[0]
        np.testing.assert_array_equal(shown.get_array().compressed(), elevation[elevation >= 0])
        assert shown.norm.vmin == 100 and shown.norm.vmax == 1000
        assert figure.axes[1].get_ylabel() == "Elevation (m)"
        assert len(axis.get_legend().get_texts()) == 3
        return original_save(figure, *args)

    monkeypatch.setattr(render_restricted, "save", inspect_and_save)
    render_restricted.render_figure1(archive, stations, tmp_path, tmp_path, terrain_file=terrain)
    for extension in ("png", "pdf", "svg"):
        assert (tmp_path / f"figure1_study_domain.{extension}").stat().st_size > 1_000


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


def _write_s2_source_data(path: Path, series: pd.DataFrame, cases: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    series.to_csv(path / "FigS2_station_timeseries.csv", index=False)
    cases.to_csv(path / "Case_definitions.csv", index=False)


def test_restricted_s2_display_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series, cases = _synthetic_s2_frames()
    # A missing observation remains a gap, not an interpolated or zero-filled point.
    series.loc[2, "value_mean_mm"] = np.nan
    source_data = tmp_path / "source_data"
    _write_s2_source_data(source_data, series, cases)
    original_save = render_restricted.save

    def inspect_and_save(figure, *args):
        assert len(figure.axes) == 4
        assert figure.get_size_inches().tolist() == [13.2, 8.7]
        assert len(figure.legends) == 1 and len(figure.legends[0].get_texts()) == 4
        for axis, case in zip(figure.axes, cases.itertuples(index=False)):
            assert len(axis.lines) == 4
            assert not axis.collections and not axis.patches  # no shading or CSI bars
            assert axis.get_ylabel() == "RN60 (mm)"
            assert axis.get_title().startswith(f"({case.panel})")
            assert "+60 min" not in axis.get_title()
            assert axis.title.get_fontweight() == "bold" and axis.title.get_fontsize() == 16
            assert axis.get_legend() is None
            assert axis.lines[0].get_color() == "black"
            for route, line in zip(render_restricted.ROUTE_ORDER, axis.lines):
                expected = series.loc[series.case_rank.eq(case.case_rank) & series.route.eq(route), "value_mean_mm"]
                np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
                assert line.get_linestyle() == "-"
        return original_save(figure, *args)

    monkeypatch.setattr(render_restricted, "save", inspect_and_save)
    render_restricted.render_s2(source_data, tmp_path)
    assert (tmp_path / "figure7_case_timeseries.png").stat().st_size > 5_000
    assert (tmp_path / "figure7_case_timeseries.pdf").stat().st_size > 1_000


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
    source_data = tmp_path / "invalid_source_data"
    _write_s2_source_data(source_data, series, cases)
    with pytest.raises(ValueError):
        render_restricted.render_s2(source_data, tmp_path)
    assert not (tmp_path / "figure7_case_timeseries.png").exists()


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
    source_data = tmp_path / "source_data"
    source_data.mkdir()
    pd.DataFrame(rows).to_csv(source_data / "FigS3_station_CSI.csv", index=False)
    original_save = render_restricted.save

    def inspect_labels_and_save(figure, *args, **kwargs):
        assert figure.get_size_inches().tolist() == [12.8, 13.2]
        assert figure.subplotpars.hspace == .20
        assert len(figure.axes) == 12
        for letter, axis in zip("abcdefghijkl", figure.axes):
            assert axis.get_title().startswith(f"({letter}) ")
            assert axis.title.get_fontweight() == "bold"
            annotations = [text for text in axis.texts if text.get_text().startswith("mean station CSI = ")]
            assert len(annotations) == 1
            annotation = annotations[0]
            assert annotation.get_position() == (.5, -.055)
            assert annotation.get_fontweight() == "normal"
            assert annotation.get_fontsize() == 14.5
            route = render_restricted.ROUTE_ORDER[1:][(ord(letter) - ord("a")) % 3]
            threshold = render_restricted.THRESHOLDS[(ord(letter) - ord("a")) // 3]
            expected = pd.read_csv(source_data / "FigS3_station_CSI.csv").query(
                "route == @route and threshold_mm == @threshold")
            np.testing.assert_array_equal(axis.collections[-1].get_array(), expected.csi_mean)
            np.testing.assert_array_equal(axis.collections[-1].get_offsets(), expected[["lon", "lat"]])
            assert annotation.get_text() == f"mean station CSI = {expected.csi_mean.mean():.3f}"
        colorbar_axes = [child for axis in figure.axes for child in axis.child_axes]
        assert len(colorbar_axes) == 4
        assert all(axis.get_ylabel() == "CSI" for axis in colorbar_axes)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(render_restricted, "save", inspect_labels_and_save)
    render_restricted.render_s3(source_data, stations, tmp_path, tmp_path)
    assert (tmp_path / "figure8_stationwise_csi.png").stat().st_size > 5_000
