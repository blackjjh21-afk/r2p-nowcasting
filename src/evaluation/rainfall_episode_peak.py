#!/usr/bin/env python3
"""Observation-defined RN60 episodes: peak-time amount, maximum amount and timing.

Episodes are maximal consecutive observed RN60 >=5 mm runs on the ten-minute
grid, retained if their observed maximum is >=20 mm. Midnight does not split a
run. Missing observations censor wet runs rather than establishing dry bounds.
An entire run containing observed RN60 >=300 mm is excluded as an analysis
QC screen, not clipped to a new peak. Source observations are never modified.

Each fixed-lead series comprises successive issuances, not a single forecast
trajectory. Score each of three members separately before averaging errors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import rainfall_episode_io as base

DiagnosticError = base.DiagnosticError
SCHEMA = "r2p-rn60-episodes-v3-common-timing"
METRICS = ("observed_peak_mae_mm", "peak_mae_mm", "timing_mae_min")
TIMING_POLICY = (
    "Absolute difference between the earliest forecast and observed maximum timestamps. "
    "For timing only, exclude an entire station-episode if any forecast series is constant "
    "throughout it, using a common subset across all routes, members and leads. All amount-evaluable "
    "episodes remain in both amount metrics. Flat forecasts have no peak timestamp or timing error. "
    "No time cap, wrapping or nearest-peak tie breaking.")


@dataclass(frozen=True)
class EpisodeConfig(base.DiagnosticConfig):
    episode_threshold_mm: float = 5.0
    qc_upper_mm: float = 300.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0 < self.episode_threshold_mm <= self.threshold_mm < self.qc_upper_mm < np.inf):
            raise DiagnosticError("Require 0 < episode threshold <= peak threshold < QC upper bound.")


def preflight(config: EpisodeConfig) -> dict[str, Any]:
    report = base.preflight(config)
    report.update(schema=SCHEMA, policy=(
        f"Observed >= {config.episode_threshold_mm:g} mm consecutive 10-min runs; "
        f"maximum >= {config.threshold_mm:g} mm; entire run excluded at >= {config.qc_upper_mm:g} mm; "
        "both observed below-threshold boundaries required; complete common forecast support."))
    return report


def load_observations(config: EpisodeConfig, stations: pd.DataFrame,
                      progress: Callable[[str], None] = print) -> tuple[np.ndarray, np.ndarray]:
    """Load observations independently of predictions on the full evaluation grid."""
    dates = base.evaluation_dates(config).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    times = (dates[:, None] + np.arange(144)[None, :] * 10 * base.MINUTE_NS).ravel()
    truth = np.full((len(times), len(stations)), np.nan, dtype=np.float64)
    seen = np.zeros(len(times), dtype=bool)
    columns = stations.target_col.astype(str).tolist()
    progress("Reading original gauge observations on the ten-minute grid...")
    for chunk in pd.read_csv(config.gauge_csv, usecols=list(base.TIME_COLUMNS) + columns,
                             chunksize=max(config.csv_chunk_rows, 8192)):
        chunk = chunk.loc[chunk.Year.isin(config.years) & chunk.Month.isin(config.months)
                          & chunk.Minute.mod(10).eq(0)]
        if chunk.empty:
            continue
        stamp = pd.to_datetime({name.lower(): chunk[name] for name in base.TIME_COLUMNS}, errors="raise")
        ns = stamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        positions = np.searchsorted(times, ns)
        if (np.any(positions >= len(times)) or not np.array_equal(times[positions], ns)
                or len(np.unique(positions)) != len(positions) or seen[positions].any()):
            raise DiagnosticError("Duplicate or unexpected gauge timestamps on the evaluation grid.")
        seen[positions] = True
        truth[positions] = chunk[columns].to_numpy(dtype=np.float64)
    progress(f"Read {int(seen.sum()):,}/{len(times):,} ten-minute observation timestamps.")
    return times, truth


def select_episodes(times_ns: np.ndarray, truth: np.ndarray, station_ids: Sequence[int],
                    config: EpisodeConfig) -> pd.DataFrame:
    """Keep all wet runs in an audit, including censored/below-threshold/QC runs.

    No interpolation or bridging missing values. A wet fragment touching a gap
    is censored and cannot be promoted into an independently bounded episode.
    """
    times = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(truth, dtype=np.float64)
    stations = np.asarray(station_ids, dtype=np.int64)
    step = config.selection_step_minutes * base.MINUTE_NS
    if (values.shape != (len(times), len(stations)) or not len(times)
            or np.any(np.diff(times) <= 0) or np.any(times % step)
            or len(np.unique(stations)) != len(stations)):
        raise DiagnosticError("Invalid observation/station/time axes.")
    contiguous = np.diff(times) == step
    records = []
    for col, station in enumerate(stations):
        observed = values[:, col]
        valid = np.isfinite(observed) & (observed >= 0)
        wet = valid & (observed >= config.episode_threshold_mm)
        starts = np.flatnonzero(wet & ~np.r_[False, wet[:-1] & contiguous])
        ends = np.flatnonzero(wet & ~np.r_[wet[1:] & contiguous, False])
        for start, end in zip(starts, ends):
            series = observed[start:end + 1]
            max_mm = float(series.max())
            peak = start + int(series.argmax())  # earliest exact tie
            tie_positions = np.flatnonzero(series == max_mm) + start
            left = "observed_dry"
            right = "observed_dry"
            if start == 0 or not contiguous[start - 1]:
                left = "evaluation_boundary"
            elif not valid[start - 1]:
                left = "missing_or_invalid_observation"
            if end == len(times) - 1 or not contiguous[end]:
                right = "evaluation_boundary"
            elif not valid[end + 1]:
                right = "missing_or_invalid_observation"
            qc_count = int((series >= config.qc_upper_mm).sum())
            complete = left == right == "observed_dry"
            status = ("qc_high_observation" if qc_count else "observation_censored" if not complete
                      else "below_peak_threshold" if max_mm < config.threshold_mm else "candidate")
            start_time, end_time, peak_time = pd.to_datetime(times[[start, end, peak]])
            records.append({
                "event_id": f"{station}:{start_time:%Y%m%dT%H%M}", "station_id": int(station),
                "start_index": int(start), "end_index": int(end), "station_index": col,
                "start_time": start_time, "end_time": end_time,
                "observed_peak_time": peak_time, "observed_max_mm": max_mm,
                "valid_date": peak_time.normalize(), "year": peak_time.year,
                "month": peak_time.month, "n_slots": len(series),
                "duration_min": len(series) * config.selection_step_minutes,
                "first_to_last_span_min": (times[end] - times[start]) / base.MINUTE_NS,
                "crosses_midnight": start_time.normalize() != end_time.normalize(),
                "observed_max_tie_count": len(tie_positions),
                "observed_max_tie_span_min": (times[tie_positions[-1]] - times[tie_positions[0]]) / base.MINUTE_NS,
                "left_boundary": left, "right_boundary": right, "boundaries_complete": complete,
                "qc_high_observation_count": qc_count, "observation_status": status, "status": status,
            })
    if not records:
        raise DiagnosticError("No observed wet runs were found.")
    return pd.DataFrame(records).sort_values(["start_time", "station_id"]).reset_index(drop=True)


def qc_observations(times: np.ndarray, truth: np.ndarray, stations: Sequence[int],
                    config: EpisodeConfig) -> pd.DataFrame:
    records = []
    for row, col in zip(*np.where(np.isfinite(truth) & (truth >= config.qc_upper_mm))):
        item = {"station_id": int(stations[col]), "time": pd.Timestamp(times[row]),
                "rn60_mm": truth[row, col], "qc_rule": f"observed_RN60_ge_{config.qc_upper_mm:g}_mm"}
        for offset in (-1, 1):
            neighbor = row + offset
            exists = (0 <= neighbor < len(times) and
                      times[neighbor] - times[row] == offset * 10 * base.MINUTE_NS)
            item["previous_10min_mm" if offset < 0 else "next_10min_mm"] = truth[neighbor, col] if exists else np.nan
        records.append(item)
    return pd.DataFrame(records, columns=["station_id", "time", "rn60_mm", "qc_rule",
                                         "previous_10min_mm", "next_10min_mm"])


def episode_features(predictions: np.ndarray, times_ns: np.ndarray,
                     observed: np.ndarray) -> dict[str, Any]:
    """Three absolute errors for ONE station-episode, lead and model member."""
    values, obs = np.asarray(predictions, dtype=float), np.asarray(observed, dtype=float)
    times = np.asarray(times_ns, dtype=np.int64)
    if (values.ndim != 1 or len(values) < 1 or values.shape != obs.shape or times.shape != values.shape
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.isfinite(obs).all() or np.any(obs < 0) or np.any(np.diff(times) <= 0)):
        raise DiagnosticError("Episode vectors must share finite nonnegative values and ordered times.")
    oi, fi = int(obs.argmax()), int(values.argmax())
    maximum = float(values[fi])
    identifiable = len(values) > 1 and maximum > float(values.min())
    timing = float((times[fi] - times[oi]) / base.MINUTE_NS) if identifiable else np.nan
    amount_error, max_error = float(values[oi] - obs[oi]), float(values[fi] - obs[oi])
    ties = np.flatnonzero(values == maximum)
    return {
        "prediction_at_observed_peak_mm": float(values[oi]), "forecast_max_mm": maximum,
        "forecast_peak_time": pd.Timestamp(times[fi]) if identifiable else pd.NaT,
        "earliest_forecast_max_time": pd.Timestamp(times[fi]),
        "forecast_max_tie_count": len(ties),
        "forecast_max_tie_span_min": (times[ties[-1]] - times[ties[0]]) / base.MINUTE_NS,
        "forecast_peak_at_boundary": fi in (0, len(values) - 1),
        "forecast_flat": bool(values.max() == values.min()), "forecast_all_zero": bool(maximum == 0),
        "timing_defined": bool(identifiable),
        "observed_peak_signed_error_mm": amount_error, "observed_peak_absolute_error_mm": abs(amount_error),
        "peak_signed_error_mm": max_error, "peak_absolute_error_mm": abs(max_error),
        "timing_error_min": timing,
        "absolute_timing_error_min": abs(timing),
    }


def apply_timing_policy(rows: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the common nonconstant subset without changing amount errors."""
    rows, audit = rows.copy(), audit.copy()
    for column in ("start_time", "end_time", "observed_peak_time", "forecast_peak_time"):
        rows[column] = pd.to_datetime(rows[column])
    if not rows.forecast_flat.eq(~rows.timing_defined).all():
        raise DiagnosticError("Timing identifiability must agree with the forecast flatness flag.")
    left = (rows.observed_peak_time - rows.start_time).dt.total_seconds() / 60
    right = (rows.end_time - rows.observed_peak_time).dt.total_seconds() / 60
    if (left < 0).any() or (right < 0).any() or not np.isfinite(left + right).all():
        raise DiagnosticError("Observed peak must lie within finite episode bounds.")
    flat = rows.forecast_flat
    if rows.loc[flat, "forecast_peak_time"].notna().any() or rows.loc[flat, "timing_error_min"].notna().any():
        raise DiagnosticError("Flat forecasts must not be assigned a peak timestamp or signed timing error.")
    if not np.isfinite(rows.loc[~flat, "timing_error_min"]).all():
        raise DiagnosticError("Nonflat forecasts must have finite signed timing errors.")
    rows["absolute_timing_error_min"] = rows.timing_error_min.abs()
    common = rows.groupby("event_id").timing_defined.all()
    rows["timing_sample"] = rows.event_id.map(common).astype(bool)
    audit["timing_sample"] = audit.status.eq("included") & audit.event_id.map(common).eq(True)
    audit["timing_status"] = np.where(audit.timing_sample, "included", np.where(
        audit.status.eq("included"), "constant_forecast_in_any_route_member_or_lead", "not_in_amount_sample"))
    return rows, audit


def match_episodes(audit: pd.DataFrame, times: np.ndarray, truth: np.ndarray,
                   station_ids: Sequence[int], stores: Sequence[base.ForecastStore],
                   anchors: np.ndarray, config: EpisodeConfig,
                   progress: Callable[[str], None] = print,
                   *, check_cached_truth: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use every ten-minute point in the same observed episode for all 45 cells."""
    base._validate_members(stores)
    out = audit.copy()
    out["missing_forecast_cells"] = 0
    supported = base.common_valid_times(anchors, config.report_leads)
    on_grid = np.isin(times, supported)
    for index in out.index[out.status.eq("candidate")]:
        row = out.loc[index]
        if not on_grid[int(row.start_index):int(row.end_index) + 1].all():
            out.at[index, "status"] = "forecast_grid_incomplete"
    candidates = out.loc[out.status.eq("candidate")]
    if candidates.empty:
        raise DiagnosticError("No complete observation-defined episodes have common forecast support.")
    indices = np.concatenate([np.arange(int(r.start_index), int(r.end_index) + 1) for r in candidates.itertuples()])
    cols = np.repeat(candidates.station_index.to_numpy(dtype=int), candidates.n_slots.to_numpy(dtype=int))
    valid_times = times[indices]
    target = truth[indices, cols]
    point_station_ids = np.asarray(station_ids)[cols]
    lengths = candidates.n_slots.to_numpy(dtype=int)
    offsets = np.r_[0, np.cumsum(lengths)]
    cube = np.empty((len(indices), len(stores), len(config.report_leads)), dtype=float)
    for k, store in enumerate(stores):
        store.validate()
        if not np.array_equal(store.anchors_ns, anchors):
            raise DiagnosticError("Forecast issuance axes differ.")
        stations = pd.Index(store.station_ids).get_indexer(point_station_ids)
        if np.any(stations < 0):
            raise DiagnosticError("An episode station is absent from forecasts.")
        for j, lead in enumerate(config.report_leads):
            issues = valid_times - lead * base.MINUTE_NS
            positions = np.searchsorted(anchors, issues)
            lead_pos = np.flatnonzero(store.leads == lead)
            if (len(lead_pos) != 1 or np.any(positions >= len(anchors))
                    or not np.array_equal(anchors[positions], issues)):
                raise DiagnosticError("Common episode issuance/lead lookup failed.")
            values = np.asarray(store.predictions[positions, stations, lead_pos[0]], dtype=float)
            if config.prediction_float16_roundtrip:
                with np.errstate(over="ignore", invalid="ignore"):
                    values = values.astype(np.float16).astype(float)
            cube[:, k, j] = values
        progress(f"Episode forecasts: {base.LABELS[store.route]}, member {store.member + 1}/3")
    if check_cached_truth:
        base.verify_episode_truth(config, anchors, valid_times, point_station_ids, target)
    records = []
    for n, (index, episode) in enumerate(candidates.iterrows()):
        sl = slice(offsets[n], offsets[n + 1])
        forecast = cube[sl]
        bad = int((~np.isfinite(forecast) | (forecast < 0)).sum())
        out.at[index, "missing_forecast_cells"] = bad
        if bad:
            out.at[index, "status"] = "forecast_values_invalid"
            continue
        out.at[index, "status"] = "included"
        for k, store in enumerate(stores):
            for j, lead in enumerate(config.report_leads):
                fields = {key: episode[key] for key in (
                    "event_id", "station_id", "start_time", "end_time", "observed_peak_time", "observed_max_mm",
                    "valid_date", "year", "month", "n_slots", "duration_min", "observed_max_tie_count")}
                fields.update(episode_features(forecast[:, k, j], valid_times[sl], target[sl]))
                fields.update(route=store.route, member=store.member, seed=store.seed, lead_min=lead)
                fields["observed_peak_issuance_time"] = fields["observed_peak_time"] - pd.Timedelta(minutes=lead)
                fields["forecast_peak_issuance_time"] = fields["forecast_peak_time"] - pd.Timedelta(minutes=lead)
                records.append(fields)
    if not records:
        raise DiagnosticError("No episodes have finite complete forecast support.")
    return apply_timing_policy(pd.DataFrame(records), out)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base._validate_error_rows(rows.assign(signed_error_mm=rows.peak_signed_error_mm,
                                         absolute_error_mm=rows.peak_absolute_error_mm))
    if rows.empty or rows.groupby("event_id").size().nunique() != 1:
        raise DiagnosticError("All episodes must have identical lead/route/member support.")
    common = rows.groupby("event_id").timing_defined.all()
    if (not rows.timing_sample.eq(rows.event_id.map(common)).all()
            or not np.isfinite(rows.loc[rows.timing_sample, "absolute_timing_error_min"]).all()):
        raise DiagnosticError("Timing must use the common nonconstant subset across all routes, members and leads.")
    member_records = []
    for period, block in [("pooled", rows)] + [(str(y), g) for y, g in rows.groupby("year")]:
        for (route, member, seed, lead), group in block.groupby(["route", "member", "seed", "lead_min"]):
            timing = group.loc[group.timing_sample, "absolute_timing_error_min"]
            member_records.append({
                "period": period, "route": route, "member": member, "seed": seed, "lead_min": lead,
                "n_episodes": len(group), "n_timing_episodes": len(timing),
                "n_undefined_timing": int((~group.timing_defined).sum()),
                "observed_peak_mae_mm": group.observed_peak_absolute_error_mm.mean(),
                "peak_mae_mm": group.peak_absolute_error_mm.mean(), "timing_mae_min": timing.mean(),
                "observed_peak_bias_mm": group.observed_peak_signed_error_mm.mean(),
                "peak_bias_mm": group.peak_signed_error_mm.mean(),
                "timing_bias_min": group.loc[group.timing_sample, "timing_error_min"].mean(),
            })
    members = pd.DataFrame(member_records)
    routes = []
    for (period, route, lead), group in members.groupby(["period", "route", "lead_min"]):
        if len(group) != 3 or group.n_episodes.nunique() != 1 or group.n_timing_episodes.nunique() != 1:
            raise DiagnosticError("Route means require three equally supported member scores.")
        item = {"period": period, "route": route, "lead_min": lead, "n_members": 3,
                "n_episodes": int(group.n_episodes.iloc[0]), "n_timing_episodes": int(group.n_timing_episodes.iloc[0])}
        for metric in METRICS + ("observed_peak_bias_mm", "peak_bias_mm", "timing_bias_min"):
            item[metric] = group[metric].mean()
            item[metric + "_member_sd"] = group[metric].std(ddof=1)
        routes.append(item)
    # One row per station-episode/lead/route, averaging member ERRORS, not forecasts.
    ep = rows.assign(common_timing_absolute_error_min=rows.absolute_timing_error_min.where(rows.timing_sample))
    per_episode = ep.groupby(["event_id", "station_id", "start_time", "end_time", "observed_peak_time",
                              "observed_max_mm", "duration_min", "route", "lead_min"], dropna=False).agg(
        n_members=("member", "size"), observed_peak_mae_mm=("observed_peak_absolute_error_mm", "mean"),
        peak_mae_mm=("peak_absolute_error_mm", "mean"), timing_mae_min=("common_timing_absolute_error_min", "mean"),
        timing_sample=("timing_sample", "all")).reset_index()
    return members, pd.DataFrame(routes), per_episode


def duration_statistics(audit: pd.DataFrame) -> pd.DataFrame:
    records = []
    selections = {"observation_selected": audit.observation_status.eq("candidate"),
                  "forecast_evaluable": audit.status.eq("included"),
                  "common_timing": audit.timing_sample}
    for population, mask in selections.items():
        sample = audit.loc[mask]
        for period, block in [("pooled", sample)] + [(str(y), g) for y, g in sample.groupby("year")]:
            duration = block.duration_min / 60
            record = {"population": population, "period": period, "n_episodes": len(block),
                      "n_stations": block.station_id.nunique(), "n_cross_midnight": int(block.crosses_midnight.sum()),
                      "n_observed_peak_ties": int(block.observed_max_tie_count.gt(1).sum())}
            for label, value in {"min": duration.min(), "mean": duration.mean(), "median": duration.median(),
                                 "p25": duration.quantile(.25), "p75": duration.quantile(.75),
                                 "p90": duration.quantile(.90), "p95": duration.quantile(.95), "max": duration.max()}.items():
                record[label + "_duration_hours"] = value
            records.append(record)
    return pd.DataFrame(records)


def build_result_tables(rows: pd.DataFrame, audit: pd.DataFrame, qc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize fresh predictions or a verified score cache identically."""
    members, routes, per_episode = summarize(rows)
    counts = audit.groupby(["year", "status", "timing_status"], dropna=False).agg(
        n_episodes=("event_id", "size"), n_stations=("station_id", "nunique"),
        n_cross_midnight=("crosses_midnight", "sum")).reset_index()
    selected = audit.loc[audit.observation_status.eq("candidate")]
    by_station = selected.groupby(["year", "station_id"]).agg(
        n_observed_episodes=("event_id", "size"), total_episode_duration_min=("duration_min", "sum"),
        median_duration_min=("duration_min", "median"), max_duration_min=("duration_min", "max"),
        max_observed_rn60_mm=("observed_max_mm", "max"),
        n_forecast_evaluable=("status", lambda x: int(x.eq("included").sum()))).reset_index()
    quality = rows.groupby(["route", "member", "lead_min"]).agg(
        n_episodes=("event_id", "size"), n_tied_forecast_maxima=("forecast_max_tie_count", lambda x: int(x.gt(1).sum())),
        n_flat_forecasts=("forecast_flat", "sum"), n_all_zero_forecasts=("forecast_all_zero", "sum"),
        n_boundary_forecast_maxima=("forecast_peak_at_boundary", "sum"),
        n_common_timing_episodes=("timing_sample", "sum")).reset_index()
    return {"episode_errors": rows, "episode_member_mean_errors": per_episode,
            "episode_audit": audit, "exclusions": audit.loc[audit.status.ne("included")].copy(),
            "timing_exclusions": audit.loc[audit.status.eq("included") & ~audit.timing_sample].copy(),
            "flat_timing_forecasts": rows.loc[rows.forecast_flat].copy(),
            "qc_observations": qc, "counts": counts, "duration_statistics": duration_statistics(audit),
            "station_statistics": by_station, "forecast_quality": quality,
            "member_metrics": members, "route_metrics": routes}


def run_analysis(config: EpisodeConfig, progress: Callable[[str], None] = print) -> dict[str, Any]:
    report = preflight(config)
    if not report["ready_paths"]:
        raise DiagnosticError("Missing required sources:\n" + "\n".join(report["missing"]))
    stations = base.load_station_contract(config)
    stores, anchors, common_stations = base.load_forecast_stores(config)
    if set(map(int, stations.station_id)) != set(map(int, common_stations)):
        raise DiagnosticError("Gauge and forecast held-out station sets differ.")
    times, truth = load_observations(config, stations, progress)
    audit = select_episodes(times, truth, stations.station_id, config)
    qc = qc_observations(times, truth, stations.station_id.to_numpy(), config)
    progress(f"Observation-selected heavy episodes: {int(audit.status.eq('candidate').sum()):,}; "
             f"QC-excluded wet runs: {int(audit.status.eq('qc_high_observation').sum()):,}.")
    rows, audit = match_episodes(audit, times, truth, stations.station_id.to_numpy(), stores, anchors, config, progress)
    tables = build_result_tables(rows, audit, qc)
    selected = audit.loc[audit.observation_status.eq("candidate")]
    provenance = {
        "schema": SCHEMA, "config": base._jsonable(asdict(config)), "preflight": report,
        "episode_definition": "Maximal consecutive observed RN60 >=5 mm on the 10-min grid; peak >=20 mm; no midnight split.",
        "missing": "Missing/negative/nonfinite observations and evaluation boundaries censor wet fragments. Both bounds must be observed <5 mm.",
        "qc": "Entire wet run containing observed RN60 >=300 mm excluded before scoring; no clipping, replacement peak or raw-data edit. This is an analysis QC rule, not proof that all such rainfall is physically impossible.",
        "duration": "n_slots*10 minutes is discrete-grid occupancy; first_to_last_span_min=(n_slots-1)*10 also exported. RN60 values are NEVER summed as event rainfall.",
        "forecast_series": "At fixed lead L, F_L(t) was issued at t-L. Use only the observed episode's timestamps; no padding or temporal tolerance.",
        "ties": "Earliest exact observed/forecast maximum; tie counts and spans reported. No nearest-to-observed tie breaking.",
        "timing": TIMING_POLICY,
        "aggregation": "Compute errors per station-episode/lead/member; average episodes within each member, then three member scores. No forecast averaging before maxima/errors.",
        "interpretation": "Timing is displacement of maxima within the same observation-defined episode, not an explicit match of individual local peaks. Multiple peaks and observed-window boundary maxima can affect it.",
        "n_observation_selected_episodes": len(selected), "n_amount_episodes": rows.event_id.nunique(),
        "n_timing_episodes": rows.loc[rows.timing_sample, "event_id"].nunique(),
        "n_qc_excluded_episodes": int(audit.status.eq("qc_high_observation").sum()), "n_qc_observation_slots": len(qc),
        "truth_crosscheck": "All selected episode RN60 values agree with manifest targets at every lead, absolute tolerance 1e-4 mm.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_code_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "gauge_source_stat": {"bytes": config.gauge_csv.stat().st_size, "mtime_ns": config.gauge_csv.stat().st_mtime_ns},
    }
    return {**tables, "provenance": provenance}


def plot_results(results: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    summary = results["route_metrics"].query("period == 'pooled'")
    amount_top = float(np.ceil(summary[list(METRICS[:2])].max().max() / 5) * 5 + 5)
    timing_max = summary[METRICS[2]].max()
    time_top = float(np.ceil(timing_max / 5) * 5 + 5) if np.isfinite(timing_max) else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for i, (ax, metric, heading) in enumerate(zip(axes, METRICS,
            ("Amount at observed peak", "Episode maximum amount", "Episode maximum timing"))):
        for route in ("pysteps_cnn", "exprecast_cnn", "direct_r2p"):
            part = summary.loc[summary.route.eq(route)].sort_values("lead_min")
            ax.plot(part.lead_min, part[metric], "o-", label=base.LABELS[route], color=base.COLORS[route])
        ax.set_title(f"({chr(97+i)}) {heading}", loc="left", fontsize=11, pad=12)
        ax.set(xlabel="Lead time (min)", ylabel="MAE (min)" if i == 2 else "MAE (mm)")
        ax.set_xticks(sorted(summary.lead_min.unique()))
        ax.set_ylim(0, time_top if i == 2 else amount_top)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"episode_peak_mae.{suffix}", dpi=200)
    plt.close(fig)
    plot_duration_histogram(results["episode_audit"], output)


def duration_histogram_grid(duration_minutes: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Ten-minute [left, right) bins starting at zero, and one-hour axis ticks."""
    minutes = np.asarray(duration_minutes, dtype=float)
    if (minutes.ndim != 1 or not len(minutes) or not np.isfinite(minutes).all()
            or np.any(minutes <= 0) or np.any(minutes % 10)):
        raise DiagnosticError("Histogram durations must be positive ten-minute multiples.")
    # Include one full bin beyond the largest discrete duration.
    edges_hours = np.arange(0, int(minutes.max()) + 20, 10, dtype=float) / 60
    ticks_hours = np.arange(0, int(np.ceil(edges_hours[-1])) + 1, dtype=float)
    return edges_hours, ticks_hours


def plot_duration_histogram(audit: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    selected = audit.loc[audit.observation_status.eq("candidate"), "duration_min"].to_numpy()
    bins, ticks = duration_histogram_grid(selected)
    # Count in integer minutes so 60 belongs to [60,70), never [50,60).
    # Conversion to hours is only for drawing, not assigning boundary values.
    edges_minutes = np.rint(bins * 60).astype(np.int64)
    selected_counts, _ = np.histogram(selected, bins=edges_minutes)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.stairs(selected_counts, bins, fill=True, facecolor="#BBCBD7",
              edgecolor="#204F78", linewidth=1.3)
    ax.text(.98, .96, f"n = {len(selected):,} station-episodes",
            transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="Episode duration (h; 10-min bins)", ylabel="Number of station-episodes",
           title="Observed RN60 ≥5 mm episodes with maximum ≥20 mm")
    ax.set_xticks(ticks)
    ax.set_xlim(0, max(ticks[-1], bins[-1]))
    ax.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"episode_duration_histogram.{suffix}", dpi=200)
    plt.close(fig)


def export_results(results: dict[str, Any], output_dir: Path, *, make_plots: bool = True) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        output = output.with_name(output.name + datetime.now().strftime("_rerun_%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output / f"{name}.csv", index=False)
    (output / "protocol_and_provenance.json").write_text(
        json.dumps(base._jsonable(results["provenance"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        plot_results(results, output)
    p = results["provenance"]
    (output / "README.md").write_text(
        "# Observation-defined RN60 episode verification\n\n"
        f"Observation-selected episodes: {p['n_observation_selected_episodes']:,}. "
        f"Amount sample: {p['n_amount_episodes']:,}. Common timing sample: {p['n_timing_episodes']:,}. "
        f"QC-excluded wet runs: {p['n_qc_excluded_episodes']:,}.\n\n"
        "An episode is a consecutive run of observed RN60 >=5 mm on the ten-minute grid with "
        "maximum >=20 mm. It continues across midnight. Both bounds must be observed below 5 mm; "
        "missing/invalid observations do not count as dry. Entire runs containing observed RN60 "
        ">=300 mm are excluded before analysis. No clipping or replacement of extreme peaks is performed.\n\n"
        "Three scores: absolute error of the forecast at the observed maximum; absolute difference "
        "between independent observed and forecast episode maxima; absolute difference of their "
        "timestamps in minutes. Exact ties select the earliest time. Both amount errors use all "
        "amount-evaluable episodes. For timing, an entire episode is excluded if any route/member/lead "
        "forecast is constant throughout it; all routes, members and leads use the same remaining "
        "episodes. Flat forecasts have no defined peak timestamp or timing error. "
        "No time-tolerance MAE is computed.\n\n"
        "Duration is the number of ten-minute slots times ten minutes; first-to-last timestamp span "
        "is also exported. RN60 is trailing-hour accumulation, not ten-minute rainfall, and is not "
        "summed again. Each fixed-lead time series uses successive forecast issuances. Observed "
        "episode bounds are used for every route. This does not match individual local peaks.\n\n"
        "- episode_errors.csv: station-episode × route × member × lead; three absolute errors and peak timestamps.\n"
        "- episode_member_mean_errors.csv: member-averaged errors for each station-episode, route and lead.\n"
        "- member_metrics.csv / route_metrics.csv: pooled and yearly MAEs, signed biases and counts.\n"
        "- episode_audit.csv / exclusions.csv / timing_exclusions.csv: observation selection and forecast support.\n"
        "- flat_timing_forecasts.csv: individual constant forecasts that exclude their episode from timing evaluation.\n"
        "- qc_observations.csv: high-value observations and immediate ten-minute neighbours.\n"
        "- duration_statistics.csv / station_statistics.csv: episode lengths, counts and station summaries.\n"
        "- forecast_quality.csv: flat, tied and boundary forecast maxima.\n"
        "- episode_peak_mae.png/pdf: the three MAEs by lead, with zero-based axes and a shared amount scale.\n"
        "- episode_duration_histogram.png/pdf: observation-selected episode lengths in ten-minute bins.\n",
        encoding="utf-8")
    (output / "COMPLETED.json").write_text(json.dumps({
        "schema": SCHEMA, "completed": True, "n_amount_episodes": int(p["n_amount_episodes"]),
        "n_timing_episodes": int(p["n_timing_episodes"]),
        "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in sorted(output.iterdir()) if f.is_file()}}, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge-csv", type=Path, required=True)
    parser.add_argument("--stations-csv", type=Path, required=True)
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--leads", type=int, nargs="+", default=[60, 90, 120, 150, 180])
    parser.add_argument("--expected-stations", type=int)
    parser.add_argument("--preflight-only", action="store_true", help="Inspect headers/contracts without scoring or writing outputs.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-float16-roundtrip", action="store_true", help="Opt out of the reported storage-precision harmonization.")
    args = parser.parse_args(argv)
    try:
        config = EpisodeConfig(gauge_csv=args.gauge_csv, stations_csv=args.stations_csv,
            forecast_manifest=args.forecast_manifest, output_dir=args.output_dir,
            years=tuple(args.years), months=tuple(args.months), report_leads=tuple(args.leads),
            expected_station_count=args.expected_stations,
            prediction_float16_roundtrip=not args.no_float16_roundtrip)
        if args.preflight_only:
            report = preflight(config)
            if report["ready_paths"]:
                base.load_forecast_stores(config)
            print(json.dumps(base._jsonable(report), indent=2))
            if not report["ready_paths"]:
                parser.exit(1)
            return
        results = run_analysis(config)
        output = export_results(results, config.output_dir, make_plots=not args.no_plots)
    except (DiagnosticError, OSError) as error:
        parser.error(str(error))
    print(results["route_metrics"].query("period == 'pooled'")[["route", "lead_min",
          "n_episodes", "n_timing_episodes", *METRICS]].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
