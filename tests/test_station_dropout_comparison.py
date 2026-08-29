from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "src" / "evaluation" / "station_dropout_comparison.py"
    spec = importlib.util.spec_from_file_location("station_dropout_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["station_dropout_comparison"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
STORE = "heldout_predictions"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(station_dropout: bool) -> dict[str, object]:
    return {
        "contract": MODULE.EXPECTED_CONTRACT,
        "optimization": {
            "batch_size": 16,
            "station_dropout": {
                "p_range": [0.0, 0.9] if station_dropout else [0.0, 0.0],
                "full_probability": 0.05 if station_dropout else 0.0,
            },
        },
        "split": {"fitting_stations": 514, "heldout_stations": 128},
    }


def _write_route(root: Path, station_dropout: bool, seeds: tuple[int, ...]) -> None:
    root.mkdir(parents=True)
    (root / "scientific_contract.json").write_text(
        json.dumps(_contract(station_dropout)), encoding="utf-8"
    )
    anchors = np.asarray(
        [
            "2024-06-01T00:00",
            "2024-06-01T01:00",
            "2024-06-02T00:00",
            "2024-06-02T01:00",
        ],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    stations = np.asarray([101, 202], dtype=np.int64)
    leads = MODULE.EXPECTED_LEAD_AXIS.copy()
    truth = np.full((len(anchors), len(stations), len(leads)), 25.0, dtype=np.float32)
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        store = seed_root / STORE
        store.mkdir(parents=True)
        checkpoint = seed_root / f"epoch_{seed + 1:03d}.pt"
        checkpoint.write_bytes(f"checkpoint-{station_dropout}-{seed}".encode())
        prediction = np.full_like(truth, 25.0 if station_dropout else 0.0)
        np.save(store / "preds_mm.npy", prediction)
        np.save(store / "trues_mm.npy", truth)
        np.save(store / "anchor_times_ns.npy", anchors)
        np.save(store / "station_ids.npy", stations)
        np.save(store / "lead_minutes.npy", leads)
        (store / "COMPLETED").write_text("complete\n", encoding="utf-8")
        metadata = {
            "checkpoint_sha256": _sha256(checkpoint),
            "completed": True,
            "contract": MODULE.EXPECTED_CONTRACT,
            "evaluation_tag": "synthetic",
            "mode": "target_masked",
            "all_context_gauges_masked": False,
            "seed": seed,
            "selected_epoch": seed + 1,
            "staged_checkpoint_name": checkpoint.name,
        }
        (store / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_missing_no_dropout_seed_fails_clearly(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    no_dropout = tmp_path / "no_dropout"
    _write_route(direct, station_dropout=True, seeds=(0, 1, 2))
    _write_route(no_dropout, station_dropout=False, seeds=(0, 2))

    with pytest.raises(
        MODULE.StationDropoutAuditError,
        match=r"no-station-dropout seed 1",
    ):
        MODULE.validate_contract(
            direct,
            no_dropout,
            STORE,
            STORE,
            expected_issue_count=4,
            expected_station_count=2,
            expected_date_count=2,
            hash_prediction_sources=False,
        )


def test_matched_comparison_computes_station_dropout_minus_control(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct"
    no_dropout = tmp_path / "no_dropout"
    _write_route(direct, station_dropout=True, seeds=(0, 1, 2))
    _write_route(no_dropout, station_dropout=False, seeds=(0, 1, 2))
    audit, axes = MODULE.validate_contract(
        direct,
        no_dropout,
        STORE,
        STORE,
        expected_issue_count=4,
        expected_station_count=2,
        expected_date_count=2,
        hash_prediction_sources=False,
    )
    comparison, by_seed, daily, _ = MODULE.compute_csi_tables(
        direct,
        no_dropout,
        STORE,
        STORE,
        axes,
        bootstrap_reps=128,
        bootstrap_seed=7,
        expected_date_count=2,
    )

    assert audit["scientific_contracts_matched_except_station_dropout"] is True
    assert len(comparison) == 20
    np.testing.assert_allclose(
        comparison.delta_csi_station_dropout_minus_no_station_dropout, 1.0
    )
    np.testing.assert_allclose(comparison.ci_low, 1.0)
    np.testing.assert_allclose(comparison.ci_high, 1.0)
    assert len(by_seed) == 60
    assert len(daily) == 120


def test_fig4b_renderer_has_no_plot_title(tmp_path: Path) -> None:
    rows = []
    for lead in MODULE.LEADS:
        for threshold in MODULE.THRESHOLDS:
            rows.append(
                {
                    "lead_min": int(lead),
                    "threshold_mm": float(threshold),
                    "delta_csi_station_dropout_minus_no_station_dropout": 0.01,
                    "ci_low": 0.005,
                    "ci_high": 0.015,
                }
            )
    frame = MODULE.pd.DataFrame(rows)
    figure, axis = MODULE.make_station_dropout_plot(frame)
    try:
        assert axis.get_title() == ""
        assert len(axis.patches) == 20
    finally:
        plt.close(figure)
