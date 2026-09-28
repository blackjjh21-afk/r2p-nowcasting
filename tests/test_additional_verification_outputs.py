from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_outputs import render_additional_verification as render

DATA = Path(__file__).resolve().parents[1] / "data/additional_verification"


def test_registered_artifact_hashes():
    root = DATA.parents[1]
    with (root / "configs/artifact_manifest.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len({record["path"] for record in records}) == len(records)
    for record in records:
        path = root / record["path"]
        assert path.is_file(), record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["path"]


def test_aggregate_checksums_and_anonymity():
    render.validate_data(DATA)
    forbidden = {"event_id", "station_id", "lat", "lon", "start_time", "end_time", "observed_peak_time"}
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["station_resolved_content_included"] is False
    for entry in manifest["records"]:
        assert not set(entry["columns"]) & forbidden
        text = (DATA / entry["path"]).read_text()
        assert "/home/" not in text and "/data01/" not in text


def test_episode_member_first_metrics_and_samples():
    routes = pd.read_csv(DATA / "episode_route_metrics.csv")
    members = pd.read_csv(DATA / "episode_member_metrics.csv")
    metrics = ["observed_peak_mae_mm", "peak_mae_mm", "timing_mae_min", "observed_peak_bias_mm", "peak_bias_mm", "timing_bias_min"]
    keys = ["period", "route", "lead_min"]
    means = members.groupby(keys)[metrics].mean().sort_index()
    np.testing.assert_allclose(routes.set_index(keys).sort_index()[metrics], means, rtol=2e-14, atol=1e-12)
    pooled = routes.query("period == 'pooled'")
    assert set(pooled.n_episodes) == {1437}
    assert set(pooled.n_timing_episodes) == {1391}
    timing60 = pooled.query("lead_min == 60").set_index("route").timing_mae_min
    assert timing60["direct_r2p"] == pytest.approx(23.99, abs=.005)
    assert timing60["exprecast_cnn"] == pytest.approx(28.15, abs=.005)
    assert timing60["pysteps_cnn"] == pytest.approx(30.89, abs=.005)


def test_histogram_counts_and_bin_boundaries():
    hist = pd.read_csv(DATA / "episode_duration_histogram.csv")
    assert hist.n_episodes.sum() == 1437
    assert (hist.right_min - hist.left_min).eq(10).all()
    np.testing.assert_array_equal(hist.right_min[:-1], hist.left_min[1:])
    nonzero = hist.query("n_episodes > 0")
    assert nonzero.left_min.min() == 60
    assert nonzero.left_min.max() == 940
    assert hist.loc[hist.left_min.eq(50), "right_min"].item() == 60


def test_patch_member_first_metrics_and_selected_epoch():
    routes = pd.read_csv(DATA / "patch_metrics_mean_of_members.csv")
    members = pd.read_csv(DATA / "patch_metrics_per_member.csv")
    keys = ["route", "lead_min", "threshold_mm"]
    metrics = ["CSI", "POD", "FAR", "frequency_bias", "MAE_mm", "RMSE_mm", "mean_error_mm"]
    np.testing.assert_allclose(routes.set_index(keys).sort_index()[metrics], members.groupby(keys)[metrics].mean().sort_index(), rtol=2e-14, atol=1e-12)
    epochs = pd.read_csv(DATA / "patch_epoch_selection.csv")
    assert epochs.loc[epochs.is_selected, "epoch"].tolist() == [4]
    assert epochs.loc[epochs.macro_csi_13lead_4threshold.idxmax(), "epoch"] == 4
    contrasts = pd.read_csv(DATA / "patch_paired_CSI_intervals.csv")
    changes = contrasts.query("contrast == 'CNN5_minus_CNN3'").CSI_difference
    assert (changes > 0).sum() == 13
    assert (changes < 0).sum() == 7
    assert changes.min() == pytest.approx(-.0108, abs=.00005)
    assert changes.max() == pytest.approx(.0065, abs=.00005)


@pytest.mark.parametrize("renderer,number", [(render.render_episode, 2), (render.render_duration, 2), (render.render_patch, 2)])
def test_render_aggregate_figures(tmp_path, renderer, number):
    paths = renderer(DATA, tmp_path)
    assert len(paths) == number
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)


def test_plot_newly_computed_patch_comparison(tmp_path):
    results = tmp_path / "comparison"
    results.mkdir()
    for name in ("metrics_mean_of_members.csv", "paired_CSI_intervals.csv"):
        (results / name).write_bytes((DATA / f"patch_{name}").read_bytes())
    output = tmp_path / "figures"
    render.main(["--item", "patch", "--patch-results", str(results), "--output-dir", str(output)])
    assert {p.name for p in output.glob("*.png")} == {"Figure_S2.png"}
    assert {p.name for p in output.glob("*.pdf")} == {"Figure_S2.pdf"}


def test_optional_patch_contrast_plot_retains_audit_capability(tmp_path):
    paths = render.render_patch(DATA, tmp_path, include_contrasts=True)
    assert len(paths) == 4
    assert (tmp_path / "patch_spatial_support_contrasts.png").is_file()
