"""Regression checks for the final JoH figure-number and data-release boundary."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_outputs import render_additional_verification as additional
from paper_outputs import render_architecture, render_publication

ROOT = Path(__file__).resolve().parents[1]


def test_native_grid_rn60_contrast_uses_exprecast_headers_and_unchanged_scores():
    data = ROOT / "data/paper_aggregates"
    contrast = pd.read_csv(data / "Field_RN60_CSI_delta_CI.csv", float_precision="round_trip")
    scores = pd.read_csv(data / "Field_RN60_CSI.csv", float_precision="round_trip")
    assert list(contrast.columns) == [
        "lead_min", "threshold_mm", "exprecast_csi", "pysteps_csi",
        "delta_exprecast_minus_pysteps", "bootstrap_mean_delta", "ci95_lower",
        "ci95_upper", "probability_delta_gt_zero", "ci_excludes_zero",
    ]
    keys = ["lead_min", "threshold_mm"]
    contrast = contrast.set_index(keys).sort_index()
    assert len(contrast) == 20 and not contrast.index.has_duplicates
    for model, column in (("exPreCast", "exprecast_csi"), ("pySTEPS", "pysteps_csi")):
        expected = scores.loc[scores.model.eq(model)].set_index(keys).sort_index()
        assert expected.index.equals(contrast.index)
        np.testing.assert_array_equal(contrast[column], expected.csi)
    np.testing.assert_allclose(
        contrast.delta_exprecast_minus_pysteps,
        contrast.exprecast_csi - contrast.pysteps_csi,
        rtol=0, atol=1e-12,
    )


def test_data_policy_distinguishes_repository_and_supplementary_observations():
    policy = " ".join((ROOT / "docs/DATA_POLICY.md").read_text().split())
    assert "selected gauge observations used in the Figure 7 case time series" in policy
    assert "they are not bundled with the source repository" in policy
    assert "The complete raw KMA radar and gauge archives are not redistributed" in policy


def test_public_csv_data_have_no_station_resolved_observation_columns():
    forbidden = {
        "station_id", "station_name", "held_out_station", "held-out_station",
        "display_station_id", "lat", "lon", "latitude", "longitude",
        "valid_time_kst", "issue_time_kst", "observed_peak_time",
    }
    for path in (ROOT / "data").rglob("*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream))
        normalized = {column.strip().lower().replace(" ", "_") for column in columns}
        assert normalized.isdisjoint(forbidden), path.relative_to(ROOT)


def test_author_edited_figure2_is_not_redrawn(tmp_path):
    paths = render_architecture.render(tmp_path)
    for path in paths:
        assert path.read_bytes() == (ROOT / "figures/reference" / path.name).read_bytes()
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == "586f5eb09ffec98950a7e6939616a47acf7c76ec3d6b0907d7a2366632bdff4f"


def test_current_public_figure_cli_and_privacy_boundary():
    for item in ("2", "3", "4", "5", "6", "9", "10", "s1", "s2", "s3", "table1"):
        assert render_publication.parser().parse_args(["--item", item]).item == item
    manifest = json.loads((ROOT / "data/paper_aggregates/manifest.json").read_text())
    for record in manifest["records"]:
        root = ROOT if record["kind"] == "reference_rendering" else ROOT / "data/paper_aggregates"
        assert hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    assert {p.stem for p in (ROOT / "figures/reference").glob("Figure_*.png")} == {
        "Figure_2", "Figure_3", "Figure_4", "Figure_5", "Figure_6", "Figure_9", "Figure_10", "Figure_S1", "Figure_S2", "Figure_S3"}


def test_patch_all_panels_show_x_axes_and_no_default_difference(tmp_path, monkeypatch):
    def inspect(fig, output, name):
        assert name == "Figure_S2"
        assert len(fig.axes) == 4
        for ax in fig.axes:
            assert ax.get_xlabel() == "Lead time (min)"
            assert ax.get_ylabel() == "CSI"
            assert ax.get_title(loc="left").startswith("(")
            assert ax._left_title.get_fontweight() == "bold"
            assert len(ax.get_xticklabels()) == 5
        plt.close(fig)
        return [output / f"{name}.png", output / f"{name}.pdf"]
    monkeypatch.setattr(additional, "save", inspect)
    additional.render_patch(ROOT / "data/additional_verification", tmp_path)


def test_duration_has_no_internal_title_and_integer_hour_boundaries(tmp_path, monkeypatch):
    def inspect(fig, output, name):
        assert name == "Figure_S3"
        ax = fig.axes[0]
        assert not ax.get_title()
        np.testing.assert_array_equal(ax.get_xticks(), np.arange(17))
        assert "1,437" in ax.texts[0].get_text()
        plt.close(fig)
        return [output / f"{name}.png", output / f"{name}.pdf"]
    monkeypatch.setattr(additional, "save", inspect)
    additional.render_duration(ROOT / "data/additional_verification", tmp_path)
