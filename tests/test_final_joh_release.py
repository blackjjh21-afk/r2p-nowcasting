"""Regression checks for the final JoH figure-number and data-release boundary."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from paper_outputs import render_additional_verification as additional
from paper_outputs import render_architecture, render_publication

ROOT = Path(__file__).resolve().parents[1]


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
