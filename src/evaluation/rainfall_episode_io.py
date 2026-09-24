"""Explicit, read-only input contracts for the RN60 episode diagnostic.

Forecast arrays are (issuance, station, lead) in millimeters. Axis files use
integer KST-naive nanoseconds, station IDs, and lead minutes respectively.
Paths in a JSON manifest are relative to that manifest, never to this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROUTES = ("direct_r2p", "pysteps_cnn", "exprecast_cnn")
LABELS = {"direct_r2p": "Direct R2P", "pysteps_cnn": "pySTEPS + CNN",
          "exprecast_cnn": "exPreCast + CNN"}
COLORS = {"direct_r2p": "#D62728", "pysteps_cnn": "#2C7FB8", "exprecast_cnn": "#009E73"}
TIME_COLUMNS = ("Year", "Month", "Day", "Hour", "Minute")
MINUTE_NS = 60_000_000_000


class DiagnosticError(ValueError):
    """A scientific or input contract was violated."""


@dataclass(frozen=True)
class DiagnosticConfig:
    gauge_csv: Path
    stations_csv: Path
    forecast_manifest: Path
    output_dir: Path
    years: tuple[int, ...] = (2024, 2025)
    months: tuple[int, ...] = (6, 7, 8, 9)
    report_leads: tuple[int, ...] = (60, 90, 120, 150, 180)
    threshold_mm: float = 20.0
    selection_step_minutes: int = 10
    expected_station_count: int | None = None
    csv_chunk_rows: int = 8192
    prediction_float16_roundtrip: bool = True

    def __post_init__(self) -> None:
        if self.selection_step_minutes != 10:
            raise DiagnosticError("The episode diagnostic requires the fixed ten-minute grid.")
        for name, values in (("years", self.years), ("months", self.months),
                             ("report_leads", self.report_leads)):
            if (not values or len(set(values)) != len(values)
                    or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values)):
                raise DiagnosticError(f"{name} must contain unique integers.")
        if any(m < 1 or m > 12 for m in self.months):
            raise DiagnosticError("Months must lie between 1 and 12.")
        if any(v < 60 or v % 10 for v in self.report_leads):
            raise DiagnosticError("Report leads must be ten-minute multiples >=60 minutes.")
        if self.csv_chunk_rows < 1 or (self.expected_station_count is not None and self.expected_station_count < 1):
            raise DiagnosticError("Chunk size and expected station count must be positive.")


def evaluation_dates(config: DiagnosticConfig) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex([], dtype="datetime64[ns]")
    for year in sorted(config.years):
        year_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        dates = dates.append(year_dates[year_dates.month.isin(config.months)])
    if dates.empty or dates.has_duplicates:
        raise DiagnosticError("Evaluation dates must be nonempty and unique.")
    return dates


def integer_axis(values: np.ndarray, label: str, *, ordered: bool = False) -> np.ndarray:
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or len(np.unique(values)) != len(values)):
        raise DiagnosticError(f"{label} must be a nonempty unique integer vector.")
    if ordered and np.any(values[1:] <= values[:-1]):
        raise DiagnosticError(f"{label} must be strictly increasing.")
    return values


@dataclass
class ForecastStore:
    route: str
    member: int
    seed: int
    anchors_ns: np.ndarray
    station_ids: np.ndarray
    leads: np.ndarray
    predictions: np.ndarray

    def validate(self) -> None:
        if self.route not in ROUTES or self.member not in range(3):
            raise DiagnosticError("Unknown route or member ID.")
        integer_axis(self.anchors_ns, "Forecast issuance timestamps", ordered=True)
        integer_axis(self.station_ids, "Forecast station IDs")
        integer_axis(self.leads, "Forecast lead minutes")
        if np.any(self.anchors_ns % (10 * MINUTE_NS)) or np.any(self.leads <= 0):
            raise DiagnosticError("Issuances must lie on the ten-minute grid and leads must be positive.")
        if (self.predictions.shape != (len(self.anchors_ns), len(self.station_ids), len(self.leads))
                or not np.issubdtype(self.predictions.dtype, np.number)
                or np.issubdtype(self.predictions.dtype, np.complexfloating)):
            raise DiagnosticError("Forecast values must be a real numeric cube matching the named axes.")


def _validate_members(stores: Sequence[ForecastStore]) -> None:
    keys = [(s.route, s.member) for s in stores]
    if len(keys) != 9 or set(keys) != {(r, m) for r in ROUTES for m in range(3)}:
        raise DiagnosticError("Exactly three complete members of each of the three routes are required.")
    for store in stores:
        store.validate()


def read_manifest(config: DiagnosticConfig) -> dict[str, Any]:
    path = Path(config.forecast_manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("timestamp_convention") != "KST-naive":
        raise DiagnosticError("Manifest must declare timestamp_convention: KST-naive.")
    if not isinstance(manifest.get("forecasts"), list) or not isinstance(manifest.get("truth"), dict):
        raise DiagnosticError("Manifest requires a forecasts list and a named-axis truth cube.")
    for record in [*manifest["forecasts"], manifest["truth"]]:
        if not isinstance(record, dict) or record.get("completed") is not True:
            raise DiagnosticError("Every forecast and truth record must explicitly declare completed: true.")
        for key in ("values", "anchors_ns", "station_ids", "lead_minutes"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise DiagnosticError(f"Manifest record requires a {key} file path.")
            candidate = Path(record[key]).expanduser()
            record[key] = candidate if candidate.is_absolute() else path.resolve().parent / candidate
    return manifest


def load_station_contract(config: DiagnosticConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.stations_csv, encoding="utf-8-sig")
    if not {"station_id", "target_col"}.issubset(frame.columns):
        raise DiagnosticError("Station CSV requires station_id,target_col for evaluated stations only.")
    integer_axis(frame.station_id.to_numpy(), "Evaluated station IDs")
    if frame.target_col.isna().any() or frame.target_col.duplicated().any():
        raise DiagnosticError("Gauge target columns must be nonmissing and unique.")
    if config.expected_station_count is not None and len(frame) != config.expected_station_count:
        raise DiagnosticError("Evaluated station count differs from the requested contract.")
    return frame[["station_id", "target_col"]].copy()


def load_cube(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.load(record[k], mmap_mode="r", allow_pickle=False)
                   for k in ("anchors_ns", "station_ids", "lead_minutes", "values"))
    ForecastStore("direct_r2p", 0, 0, *arrays).validate()
    return arrays


def load_forecast_stores(config: DiagnosticConfig) -> tuple[list[ForecastStore], np.ndarray, np.ndarray]:
    manifest = read_manifest(config)
    stores = []
    for record in manifest["forecasts"]:
        for key in ("member", "seed"):
            if type(record.get(key)) is not int:
                raise DiagnosticError(f"Forecast {key} must be an integer.")
        store = ForecastStore(record.get("route"), record["member"], record["seed"], *load_cube(record))
        stores.append(store)
    _validate_members(stores)
    anchors, stations = stores[0].anchors_ns, stores[0].station_ids
    evaluated = load_station_contract(config).station_id.to_numpy()
    if set(stations) != set(evaluated):
        raise DiagnosticError("Gauge and forecast evaluated station sets differ.")
    for store in stores:
        if not np.array_equal(store.anchors_ns, anchors):
            raise DiagnosticError("Forecast issuance axes differ.")
        if set(store.station_ids) != set(stations):
            raise DiagnosticError("Forecast station sets differ.")
        if not set(config.report_leads).issubset(store.leads):
            raise DiagnosticError("A requested report lead is unavailable.")
    ta, ts, tl, _ = load_cube(manifest["truth"])
    if not np.array_equal(ta, anchors) or set(ts) != set(stations) or not set(config.report_leads).issubset(tl):
        raise DiagnosticError("Truth issuance, station or lead axes differ from forecast support.")
    return stores, anchors, stations


def verify_episode_truth(config: DiagnosticConfig, anchors: np.ndarray, valid_times: np.ndarray,
                         station_ids: np.ndarray, observed: np.ndarray) -> None:
    ta, ts, tl, truth = load_cube(read_manifest(config)["truth"])
    if not np.array_equal(ta, anchors):
        raise DiagnosticError("Truth issuance axes differ.")
    cols = pd.Index(ts).get_indexer(station_ids)
    if np.any(cols < 0):
        raise DiagnosticError("An episode station is absent from truth.")
    for lead in config.report_leads:
        issues = valid_times - lead * MINUTE_NS
        positions = np.searchsorted(ta, issues)
        lead_pos = np.flatnonzero(tl == lead)
        if (len(lead_pos) != 1 or np.any(positions >= len(ta))
                or not np.array_equal(ta[positions], issues)):
            raise DiagnosticError("Episode truth issuance/lead lookup failed.")
        reference = truth[positions, cols, lead_pos[0]]
        if not np.allclose(reference, observed, rtol=0, atol=1e-4, equal_nan=False):
            raise DiagnosticError("Original gauge episode values differ from manifest RN60 targets.")


def preflight(config: DiagnosticConfig) -> dict[str, Any]:
    stations = load_station_contract(config)
    header = pd.read_csv(config.gauge_csv, nrows=0).columns
    absent = set(TIME_COLUMNS).union(stations.target_col.astype(str)) - set(header)
    if absent:
        raise DiagnosticError(f"Gauge CSV columns missing: {sorted(absent)}")
    manifest = read_manifest(config)
    sources = []
    missing = []
    for record in [*manifest["forecasts"], manifest["truth"]]:
        for key in ("values", "anchors_ns", "station_ids", "lead_minutes"):
            path = record[key]
            if not path.is_file():
                missing.append(str(path))
            else:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                sources.append({"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype),
                                "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    return {"ready_paths": not missing, "missing": sorted(set(missing)), "sources": sources,
            "station_count": len(stations), "timestamp_convention": "KST-naive",
            "manifest_sha256": hashlib.sha256(Path(config.forecast_manifest).read_bytes()).hexdigest(),
            "prediction_payload_read": False, "full_file_hashes_verified": False}


def common_valid_times(anchors_ns: np.ndarray, leads: Sequence[int]) -> np.ndarray:
    supported = None
    for lead in leads:
        shifted = np.asarray(anchors_ns, dtype=np.int64) + int(lead) * MINUTE_NS
        supported = shifted if supported is None else np.intersect1d(supported, shifted, assume_unique=True)
    return np.asarray(supported, dtype=np.int64)


def _validate_error_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        raise DiagnosticError("No error rows to summarize.")
    if rows.duplicated(["event_id", "lead_min", "route", "member"]).any():
        raise DiagnosticError("Duplicate event/lead/route/member error rows.")
    if set(rows.route) != set(ROUTES):
        raise DiagnosticError("All three routes are required.")
    members = rows.groupby(["event_id", "lead_min", "route"]).member.agg(lambda x: tuple(sorted(x)))
    if (not rows.groupby(["event_id", "lead_min"]).size().eq(9).all()
            or not members.map(lambda value: value == (0, 1, 2)).all()):
        raise DiagnosticError("Each event/lead requires all three members of every route.")
    leads = tuple(sorted(rows.lead_min.unique()))
    if not rows.groupby("event_id").lead_min.agg(lambda x: tuple(sorted(set(x)))).map(lambda v: v == leads).all():
        raise DiagnosticError("Episodes must share identical lead sets.")
    if not np.isfinite(rows[["signed_error_mm", "absolute_error_mm"]].to_numpy()).all():
        raise DiagnosticError("Nonfinite amount errors are not allowed.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
