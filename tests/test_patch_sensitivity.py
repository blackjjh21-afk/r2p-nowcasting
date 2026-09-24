from __future__ import annotations

import hashlib
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn_readout.model import CNN
from cnn_readout import patch_sensitivity as ps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source(tmp_path: Path, *, bad_fold_dtype: bool = False) -> tuple[Path, dict[str, np.ndarray]]:
    issue_dates = []
    for month in range(6, 10):
        issue_dates.extend(
            [
                np.datetime64(f"2023-{month:02d}-10T12:00:00", "ns"),
                np.datetime64(f"2023-{month:02d}-20T12:00:00", "ns"),
            ]
        )
    issues, stations, leads = 8, 6, 13
    base = np.arange(18 * 5 * 5, dtype=np.float32).reshape(18, 5, 5) / 1000.0
    patches = np.broadcast_to(base, (issues, stations, 18, 5, 5)).copy()
    target_values = np.asarray((0.0, 2.0, 7.0, 15.0, 25.0), dtype=np.float32)
    target_index = np.arange(issues * stations * leads).reshape(issues, stations, leads) % 5
    arrays = {
        "patches18": patches,
        "issue_times_ns": np.asarray(issue_dates).astype(np.int64),
        "station_ids": np.arange(100, 100 + stations, dtype=np.int64),
        "xy_norm": np.column_stack(
            (np.linspace(0.1, 0.9, stations), np.linspace(0.9, 0.1, stations))
        ).astype(np.float32),
        "station_folds": np.asarray(
            (0, 1, 2, 0, 1, 2), dtype=np.float32 if bad_fold_dtype else np.int8
        ),
        "targets_mm": target_values[target_index],
        "context36": np.zeros((issues, stations, 36), dtype=np.float32),
        "oof_context36": np.zeros((3, issues, stations, 36), dtype=np.float32),
    }
    paths = {}
    for name, value in arrays.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        paths[name] = path.name
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {
                "schema": ps.SOURCE_SCHEMA,
                "patch_normalization": ps.PATCH_NORMALIZATION,
                "issue_time_convention": ps.ISSUE_TIME_CONVENTION,
                "target_units": "mm",
                "arrays": paths,
                "provenance": {"fixture": True},
            }
        ),
        encoding="utf-8",
    )
    return config, arrays


def test_prepare_window_contract_and_parameter_control(tmp_path):
    config, arrays = _write_source(tmp_path)
    prepared = tmp_path / "prepared.json"
    assert ps.main(["prepare", "--config", str(config), "--output", str(prepared)]) == 0
    data = ps.load_prepared(prepared)
    issues = np.asarray((0, 1), dtype=np.int64)
    stations = np.asarray((0, 1), dtype=np.int64)
    leads = np.asarray((0, 12), dtype=np.int64)
    patch5 = ps._gather_patches(data, issues, stations, leads, 5)
    patch3 = ps._gather_patches(data, issues, stations, leads, 3)
    np.testing.assert_array_equal(patch3, patch5[:, :, 1:4, 1:4])
    np.testing.assert_array_equal(patch5[0], arrays["patches18"][0, 0, :6])
    np.testing.assert_array_equal(patch5[1], arrays["patches18"][1, 1, 12:18])
    assert sum(value.numel() for value in CNN(patch_size=3, aux_dim=39).parameters()) == 51_282
    assert sum(value.numel() for value in CNN(patch_size=5, aux_dim=39).parameters()) == 51_282


def test_prepare_rejects_fractional_fold_labels(tmp_path):
    config, _ = _write_source(tmp_path, bad_fold_dtype=True)
    with pytest.raises(ps.ContractError, match="station_folds must use an integer dtype"):
        ps.main(["prepare", "--config", str(config), "--output", str(tmp_path / "bad.json")])


@pytest.fixture(scope="module")
def tiny_completed_workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("patch-sensitivity")
    config, _ = _write_source(root)
    prepared = root / "prepared.json"
    ps.main(["prepare", "--config", str(config), "--output", str(prepared)])
    selection = root / "selection.json"
    ps.main(
        [
            "select-epoch",
            "--input",
            str(prepared),
            "--output",
            str(selection),
            "--patch-size",
            "5",
            "--max-epochs",
            "1",
            "--samples-per-epoch",
            "100",
            "--batch-size",
            "50",
            "--validation-batch-size",
            "128",
        ]
    )
    checkpoints = root / "checkpoints"
    ps.main(
        [
            "fit-final",
            "--input",
            str(prepared),
            "--selection",
            str(selection),
            "--output-dir",
            str(checkpoints),
            "--seeds",
            "21040",
            "--samples-per-epoch",
            "100",
            "--batch-size",
            "50",
        ]
    )
    predictions = root / "predictions"
    ps.main(
        [
            "predict",
            "--input",
            str(prepared),
            "--checkpoint-dir",
            str(checkpoints),
            "--output-dir",
            str(predictions),
            "--issue-batch-size",
            "2",
        ]
    )
    return root, prepared, selection, checkpoints, predictions


def test_oof_cell_reuse_rejects_changed_budget(tiny_completed_workflow):
    _, prepared, selection, _, _ = tiny_completed_workflow
    with pytest.raises(ps.ContractError, match="stale OOF cell"):
        ps.main(
            [
                "select-epoch",
                "--input",
                str(prepared),
                "--output",
                str(selection),
                "--patch-size",
                "5",
                "--max-epochs",
                "1",
                "--samples-per-epoch",
                "101",
                "--batch-size",
                "50",
                "--validation-batch-size",
                "128",
            ]
        )


def test_predict_rejects_nonfinite_model_output(tiny_completed_workflow, monkeypatch):
    root, prepared, _, checkpoints, _ = tiny_completed_workflow

    def nonfinite(value):
        return torch.full_like(value, float("inf"))

    monkeypatch.setattr(ps, "millimetres_from_normalized_prediction", nonfinite)
    with pytest.raises(RuntimeError, match="non-finite"):
        ps.main(
            [
                "predict",
                "--input",
                str(prepared),
                "--checkpoint-dir",
                str(checkpoints),
                "--output-dir",
                str(root / "bad_predictions"),
                "--issue-batch-size",
                "2",
            ]
        )


def test_compare_member_first_outputs(tiny_completed_workflow):
    root, prepared, _, _, predictions = tiny_completed_workflow
    output = root / "comparison"
    ps.main(
        [
            "compare",
            "--truth-input",
            str(prepared),
            "--route",
            f"A={predictions}",
            "--route",
            f"B={predictions}",
            "--contrast",
            "B_minus_A:B:A",
            "--output-dir",
            str(output),
            "--expected-members",
            "1",
            "--bootstrap-resamples",
            "10",
        ]
    )
    rows = list(csv.DictReader((output / "paired_CSI_intervals.csv").open(encoding="utf-8")))
    assert len(rows) == 20
    assert {float(row["CSI_difference"]) for row in rows} == {0.0}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["common_finite_pairs_all_13_leads"] == 8 * 6 * 13
    assert manifest["bootstrap"]["repetitions"] == 10
