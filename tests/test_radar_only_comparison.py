from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "src" / "evaluation" / "radar_only_comparison.py"
    spec = importlib.util.spec_from_file_location("radar_only_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["radar_only_comparison"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_csi_uses_hit_miss_false_alarm_only() -> None:
    counts = np.asarray([[4, 2, 2, 99], [0, 0, 0, 10]], dtype=np.int64)
    values = MODULE.csi(counts)
    assert values[0] == pytest.approx(0.5)
    assert np.isnan(values[1])


def test_date_counts_preserve_paired_issuance_date_blocks() -> None:
    truth = np.asarray([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    prediction = np.asarray([[2.0, 2.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    finite = np.ones_like(truth, dtype=bool)
    date_index = np.asarray([0, 0, 1], dtype=np.int16)
    counts = MODULE.date_counts(
        prediction, truth, finite, date_index, threshold=1.0, n_dates=2
    )
    # Date 0: hit=1, miss=1, false alarm=1, correct negative=1.
    # Date 1: hit=1, miss=1, false alarm=0, correct negative=0.
    np.testing.assert_array_equal(counts, [[1, 1, 1, 1], [1, 1, 0, 0]])


def test_axis_audit_rejects_standard_radar_only_mismatch() -> None:
    left = np.asarray([60, 90, 120], dtype=np.int16)
    MODULE.assert_array_equal(left, left.copy(), "same")
    with pytest.raises(MODULE.RadarOnlyAuditError, match="array mismatch"):
        MODULE.assert_array_equal(left, np.asarray([60, 90, 150]), "lead")


def test_bootstrap_resamples_whole_dates() -> None:
    weights = MODULE.bootstrap_weights(n_dates=5, replicates=64, seed=123)
    assert weights.shape == (64, 5)
    np.testing.assert_array_equal(weights.sum(axis=1), np.full(64, 5))
    assert np.issubdtype(weights.dtype, np.integer)


def test_cli_requires_portable_input_and_output_paths() -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--r2p-root",
            "results",
            "--output-dir",
            "derived",
            "--figure-dir",
            "figures",
        ]
    )
    assert args.r2p_root == Path("results")


def test_pdf_rendering_is_byte_reproducible(tmp_path: Path) -> None:
    rows = []
    for lead in MODULE.LEADS:
        for threshold in MODULE.THRESHOLDS:
            rows.append(
                {
                    "lead_min": int(lead),
                    "threshold_mm": float(threshold),
                    "delta_csi_radar_only_minus_standard": 0.001,
                    "ci_low": 0.0,
                    "ci_high": 0.002,
                }
            )
    frame = MODULE.pd.DataFrame(rows)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    MODULE.render_radar_only(frame, tmp_path / "first.png", first)
    MODULE.render_radar_only(frame, tmp_path / "second.png", second)
    assert first.read_bytes() == second.read_bytes()
