"""Portable latest-JoH display contracts, using synthetic caller inputs only."""

from pathlib import Path

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pytest

from paper_outputs import render_restricted as renderer


def test_elevation_colour_scale_preserves_latest_limits() -> None:
    cmap, norm = renderer.elevation_colour_scale()
    white = colors.to_rgba("#FFFFFF")
    brown = colors.to_rgba("#664229")
    for elevation in (-20, 0, 99, 100):
        np.testing.assert_allclose(cmap(norm(elevation)), white)
    for elevation in (1000, 1500, 3000):
        np.testing.assert_allclose(cmap(norm(elevation)), brown)
    assert renderer.ELEVATION_TICKS_M == tuple(range(100, 1001, 100))


@pytest.mark.parametrize("format", ("npz", "nc"))
def test_supplied_terrain_loaded_without_value_changes(tmp_path: Path, format: str) -> None:
    lon = np.linspace(120, 134, 6)
    lat = np.linspace(31, 42, 5)
    elevation = np.arange(30, dtype=float).reshape(5, 6) * 80 - 100
    path = tmp_path / f"synthetic_terrain.{format}"
    if format == "npz":
        np.savez_compressed(path, lon=lon, lat=lat, elevation_m=elevation)
    else:
        xr = pytest.importorskip("xarray")
        xr.Dataset({"z": (("latitude", "longitude"), elevation)},
                   coords={"longitude": lon, "latitude": lat}).to_netcdf(path)
    actual = renderer.load_terrain(path)
    for values, expected in zip(actual, (lon, lat, elevation)):
        np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize("defect", ("schema", "shape", "nonfinite", "nonmonotonic"))
def test_terrain_rejects_invalid_inputs(tmp_path: Path, defect: str) -> None:
    values = {"lon": np.arange(6, dtype=float), "lat": np.arange(5, dtype=float),
              "elevation_m": np.ones((5, 6))}
    if defect == "schema":
        values.pop("elevation_m")
    elif defect == "shape":
        values["elevation_m"] = np.ones((6, 5))
    elif defect == "nonfinite":
        values["elevation_m"][0, 0] = np.nan
    else:
        values["lon"][1] = values["lon"][0]
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        renderer.load_terrain(path)


def test_current_source_names_take_precedence_over_legacy(tmp_path: Path) -> None:
    pd.DataFrame({"value": [1]}).to_csv(tmp_path / "Fig7_station_timeseries.csv", index=False)
    pd.DataFrame({"value": [2]}).to_csv(tmp_path / "FigS2_station_timeseries.csv", index=False)
    frame = renderer.read_source_csv(tmp_path, "Fig7_station_timeseries", "FigS2_station_timeseries")
    assert frame.value.tolist() == [1]


def test_current_commands_and_legacy_aliases() -> None:
    parser = renderer.parser()
    for command in ("figure7", "s2"):
        args = parser.parse_args([command, "--source-data", "supplied", "--output-dir", "out"])
        assert args.item == command
    for command in ("figure8", "s3"):
        args = parser.parse_args([command, "--source-data", "supplied", "--stations-csv", "split.csv",
                                  "--cartopy-data-dir", "map-cache", "--output-dir", "out"])
        assert args.item == command
    args = parser.parse_args(["figure1", "--coverage-npz", "coverage.npz", "--stations-csv", "split.csv",
                              "--cartopy-data-dir", "map-cache", "--terrain-nc", "terrain.nc",
                              "--output-dir", "out"])
    assert args.terrain_file == Path("terrain.nc")
    assert renderer.render_s2 is renderer.render_figure7
    assert renderer.render_s3 is renderer.render_figure8
