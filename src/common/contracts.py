"""Schema validation for compact route-comparison fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "route_prediction_fixture_v1"


@dataclass(frozen=True)
class RouteFixture:
    issue_times: np.ndarray
    station_ids: np.ndarray
    lead_minutes: np.ndarray
    thresholds_mm: np.ndarray
    truth: np.ndarray
    predictions: dict[str, np.ndarray]


def _numeric_array(payload: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(payload, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def load_route_fixture(path: str | Path) -> RouteFixture:
    """Load and strictly validate a truth/prediction fixture JSON."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    issue_times = np.asarray(payload["issue_times"]).astype("datetime64[ns]")
    station_ids = np.asarray(payload["station_ids"], dtype=np.int64)
    lead_minutes = np.asarray(payload["lead_minutes"], dtype=np.int64)
    thresholds = np.asarray(payload["thresholds_mm"], dtype=np.float64)
    if issue_times.ndim != station_ids.ndim or issue_times.ndim != lead_minutes.ndim or issue_times.ndim != 1:
        raise ValueError("issue_times, station_ids, and lead_minutes must be one-dimensional")
    if len(issue_times) < 2 or len(station_ids) == 0 or len(lead_minutes) == 0:
        raise ValueError("fixture dimensions must be non-empty")
    if np.isnat(issue_times).any() or np.any(issue_times[1:] <= issue_times[:-1]):
        raise ValueError("issue_times must be finite and strictly increasing")
    if len(np.unique(station_ids)) != len(station_ids):
        raise ValueError("station_ids must be unique")
    if np.any(lead_minutes <= 0) or np.any(lead_minutes[1:] <= lead_minutes[:-1]):
        raise ValueError("lead_minutes must be positive and strictly increasing")
    if thresholds.ndim != 1 or len(thresholds) == 0 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds_mm must be a non-empty finite sequence")

    shape = (len(issue_times), len(station_ids), len(lead_minutes))
    truth = _numeric_array(payload["truth"], "truth", shape)
    prediction_payload = payload.get("predictions")
    if not isinstance(prediction_payload, dict) or len(prediction_payload) < 2:
        raise ValueError("predictions must contain at least two named routes")
    predictions = {
        str(name): _numeric_array(value, f"predictions[{name!r}]", shape)
        for name, value in prediction_payload.items()
    }
    return RouteFixture(
        issue_times=issue_times,
        station_ids=station_ids,
        lead_minutes=lead_minutes,
        thresholds_mm=thresholds,
        truth=truth,
        predictions=predictions,
    )
