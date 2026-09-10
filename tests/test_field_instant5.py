"""Fig. S4b must plot genuine, complete native-grid 5-mm/h scores."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_outputs import render_publication as renderer


DATA = Path(__file__).resolve().parents[1] / "data/paper_aggregates"


def test_instantaneous_five_mm_series_is_complete_and_exact():
    frame = pd.read_csv(DATA / "Field_instant_CSI.csv", float_precision="round_trip")
    differences = renderer.instantaneous_csi_differences(frame)
    assert tuple(differences) == (1.0, 5.0, 10.0, 20.0)
    for series in differences.values():
        np.testing.assert_array_equal(series.index, np.arange(10, 181, 10))
        assert np.isfinite(series).all()
    # Independently stored native-grid contingency counts, not interpolation.
    assert differences[5.0].loc[10] == pytest.approx(
        15430923 / (15430923 + 2778280 + 11548871)
        - 12899773 / (12899773 + 5309430 + 3516607), abs=1e-15,
    )
    assert differences[5.0].loc[180] == pytest.approx(
        4307274 / (4307274 + 13905078 + 6316056)
        - 3165374 / (3165374 + 15046978 + 12175988), abs=1e-15,
    )


@pytest.mark.parametrize("defect", ["missing_threshold", "missing_lead", "duplicate_lead", "nonfinite"])
def test_instantaneous_panel_rejects_incomplete_series(defect):
    frame = pd.read_csv(DATA / "Field_instant_CSI.csv")
    five = frame.model.eq("exPreCast") & frame.threshold_mm_h.eq(5)
    first = frame.index[five][0]
    if defect == "missing_threshold":
        frame = frame.loc[~five]
    elif defect == "missing_lead":
        frame = frame.drop(index=first)
    elif defect == "duplicate_lead":
        frame = pd.concat([frame, frame.loc[[first]]], ignore_index=True)
    else:
        frame.loc[first, "csi"] = np.nan
    with pytest.raises(ValueError, match="5 mm h-1"):
        renderer.instantaneous_csi_differences(frame)


def test_figure_s4_panel_b_has_four_eighteen_point_curves(tmp_path, monkeypatch):
    def inspect(figure, _output):
        lines = [line for line in figure.axes[1].lines if not line.get_label().startswith("_")]
        assert len(lines) == 4
        for line in lines:
            np.testing.assert_array_equal(line.get_xdata(), np.arange(10, 181, 10))
            assert len(line.get_ydata()) == 18
        assert lines[1].get_label().startswith("5 mm")
        assert lines[1].get_color() == "#EE8420"
        renderer.plt.close(figure)
        return tmp_path / "figureS4.png", tmp_path / "figureS4.pdf"

    monkeypatch.setattr(renderer, "save", inspect)
    renderer.render_figure_s4(DATA, tmp_path)
