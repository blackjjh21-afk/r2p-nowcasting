"""Synthetic episode diagnostics; no project observations or predictions are read."""

from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


from evaluation import rainfall_episode_peak as mod


base = mod.base


def config(tmp_path, **overrides):
    options = dict(years=(2024,), months=(6,), expected_station_count=2,
                   prediction_float16_roundtrip=False)
    options.update(overrides)
    return mod.EpisodeConfig(tmp_path / "gauge.csv", tmp_path / "stations.csv",
                             tmp_path / "forecasts.json", tmp_path / "output", **options)


def times(n, start="2024-06-02 12:00"):
    return pd.date_range(start, periods=n, freq="10min").asi8


def select(values, cfg, *, start="2024-06-02 12:00", stamps=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return mod.select_episodes(times(len(values), start) if stamps is None else stamps,
                               values, np.arange(values.shape[1]) + 7, cfg)


def synthetic_case(tmp_path):
    """Two wet runs and deliberately permuted forecast station/lead axes."""
    cfg = config(tmp_path)
    stamp = times(15)
    truth = np.zeros((len(stamp), 2))
    truth[1:5, 0] = [5, 50, 15, 5]
    truth[8:12, 0] = [5, 50, 15, 5]
    audit = mod.select_episodes(stamp, truth, [7, 11], cfg)
    anchors = np.arange(stamp[0] - 180 * base.MINUTE_NS,
                        stamp[-1] + 10 * base.MINUTE_NS,
                        10 * base.MINUTE_NS, dtype=np.int64)
    stores = []
    for ri, route in enumerate(base.ROUTES):
        for member in range(3):
            leads = np.roll(np.array([180, 90, 60, 150, 120]), member)
            stations = np.array([11, 7] if member % 2 == 0 else [7, 11])
            prediction = np.ones((len(anchors), 2, 5))
            for sj, station in enumerate(stations):
                for lj, lead in enumerate(leads):
                    valid = anchors + lead * base.MINUTE_NS
                    offset = ri + member / 10 + lead / 1000
                    prediction[:, sj, lj] += offset + 100 * (station == 11)
                    if station == 7:
                        prediction[np.isin(valid, stamp[[2, 9]]), sj, lj] = 25 + offset
                        prediction[np.isin(valid, stamp[[3, 10]]), sj, lj] = 49 + offset
            stores.append(base.ForecastStore(route, member,
                member if route == "direct_r2p" else 21040 + member,
                anchors.copy(), stations, leads, prediction))
    return cfg, stamp, truth, stores, anchors, audit


def run_case(case):
    cfg, stamp, truth, stores, anchors, audit = case
    return mod.match_episodes(audit, stamp, truth, [7, 11], stores, anchors, cfg,
                              progress=lambda _: None, check_cached_truth=False)


def forecast_cell(store, valid, *, station=7, lead=180):
    return (int(np.flatnonzero(store.anchors_ns == valid - lead * base.MINUTE_NS)[0]),
            int(np.flatnonzero(store.station_ids == station)[0]),
            int(np.flatnonzero(store.leads == lead)[0]))


def test_thresholds_inclusive_and_midnight_does_not_split(tmp_path):
    cfg = config(tmp_path)
    frame = select([0, 5, 20, 5, 0], cfg, start="2024-06-01 23:40")
    row = frame.iloc[0]
    assert len(frame) == 1
    assert row.status == "candidate"
    assert row.start_time == pd.Timestamp("2024-06-01 23:50")
    assert row.end_time == pd.Timestamp("2024-06-02 00:10")
    assert row.observed_peak_time == pd.Timestamp("2024-06-02")
    assert row.crosses_midnight
    assert row.n_slots == 3
    assert row.duration_min == 30
    assert row.first_to_last_span_min == 20
    assert row.left_boundary == row.right_boundary == "observed_dry"


def test_peak_threshold_below20_remains_audited_not_selected(tmp_path):
    frame = select([0, 5, 19.999, 5, 0, 20, 0], config(tmp_path))
    assert frame.status.tolist() == ["below_peak_threshold", "candidate"]
    assert frame.n_slots.tolist() == [3, 1]
    assert frame.duration_min.tolist() == [30, 10]
    assert frame.first_to_last_span_min.tolist() == [20, 0]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1])
def test_missing_or_invalid_observation_censors_both_wet_fragments(tmp_path, bad):
    frame = select([0, 5, 25, bad, 30, 5, 0], config(tmp_path))
    assert len(frame) == 2
    assert frame.status.eq("observation_censored").all()
    assert frame.iloc[0].right_boundary == "missing_or_invalid_observation"
    assert frame.iloc[1].left_boundary == "missing_or_invalid_observation"
    assert not frame.boundaries_complete.any()


def test_temporal_gap_censors_instead_of_joining_or_calling_it_dry(tmp_path):
    stamp = times(8)[[0, 1, 2, 5, 6, 7]]
    frame = select([0, 5, 25, 30, 5, 0], config(tmp_path), stamps=stamp)
    assert frame.status.tolist() == ["observation_censored", "observation_censored"]
    assert frame.iloc[0].right_boundary == "evaluation_boundary"
    assert frame.iloc[1].left_boundary == "evaluation_boundary"


def test_evaluation_edges_censor_wet_runs(tmp_path):
    frame = select([25, 5, 0, 5, 30], config(tmp_path))
    assert frame.status.eq("observation_censored").all()
    assert frame.iloc[0].left_boundary == "evaluation_boundary"
    assert frame.iloc[1].right_boundary == "evaluation_boundary"


@pytest.mark.parametrize("spike", [300.0, 300.1, 900.0])
def test_qc_excludes_entire_original_wet_run_not_only_spike(tmp_path, spike):
    source = np.array([0, 5, 50, spike, 25, 5, 0, 20, 0], dtype=float)
    copy = source.copy()
    frame = select(source, config(tmp_path))
    np.testing.assert_array_equal(source, copy)
    assert len(frame) == 2
    row = frame.iloc[0]
    assert row.status == "qc_high_observation"
    assert row.start_index == 1 and row.end_index == 5
    assert row.n_slots == 5
    assert row.observed_max_mm == spike
    assert row.qc_high_observation_count == 1
    assert frame.iloc[1].status == "candidate"


def test_value_just_below_qc_threshold_is_not_silently_removed(tmp_path):
    frame = select([0, 5, 299.99, 5, 0], config(tmp_path))
    assert frame.iloc[0].status == "candidate"


def test_earliest_observed_tie_and_span_are_explicit(tmp_path):
    frame = select([0, 5, 20, 5, 20, 0], config(tmp_path))
    row = frame.iloc[0]
    assert row.observed_peak_time == pd.Timestamp("2024-06-02 12:20")
    assert row.observed_max_tie_count == 2
    assert row.observed_max_tie_span_min == 20


def test_station_runs_are_selected_independently(tmp_path):
    frame = select([[0, 0], [5, 0], [20, 0], [5, 25], [0, 0]], config(tmp_path))
    assert frame.station_id.tolist() == [7, 8]
    assert frame.n_slots.tolist() == [3, 1]


def test_qc_audit_records_threshold_and_true_tenminute_neighbours(tmp_path):
    cfg = config(tmp_path)
    stamp = times(5)
    values = np.array([0, 25, 300, 10, 0], dtype=float)[:, None]
    frame = mod.qc_observations(stamp, values, [7], cfg)
    row = frame.iloc[0]
    assert row.time == pd.Timestamp("2024-06-02 12:20")
    assert row.previous_10min_mm == 25 and row.next_10min_mm == 10
    assert row.rn60_mm == 300 and row.station_id == 7


def test_three_errors_distinguish_amount_at_observed_peak_and_independent_maxima():
    row = mod.episode_features([5, 25, 49, 5], times(4), [5, 50, 15, 5])
    assert row["prediction_at_observed_peak_mm"] == 25
    assert row["observed_peak_absolute_error_mm"] == 25
    assert row["forecast_max_mm"] == 49
    assert row["peak_absolute_error_mm"] == 1
    assert row["timing_error_min"] == row["absolute_timing_error_min"] == 10


def test_timing_is_not_capped_or_nearest_local_peak_matched():
    obs = np.full(13, 5.)
    pred = np.full(13, 6.)
    obs[1], obs[10] = 50, 49
    pred[1], pred[10] = 48, 51
    row = mod.episode_features(pred, times(13), obs)
    assert row["observed_peak_absolute_error_mm"] == 2
    assert row["peak_absolute_error_mm"] == 1
    assert row["timing_error_min"] == 90


def test_nonflat_forecast_tie_uses_earliest_not_nearest_observed_peak():
    row = mod.episode_features([49, 5, 49, 5], times(4), [5, 5, 50, 5])
    assert row["forecast_max_tie_count"] == 2
    assert row["forecast_max_tie_span_min"] == 20
    assert row["timing_error_min"] == -20
    assert row["absolute_timing_error_min"] == 20
    assert row["forecast_peak_at_boundary"]


@pytest.mark.parametrize("constant", [0., 4.5])
def test_flat_forecast_retains_amount_errors_without_inventing_peak_time(constant):
    row = mod.episode_features([constant] * 4, times(4), [5, 50, 15, 5])
    assert row["observed_peak_absolute_error_mm"] == 50 - constant
    assert row["peak_absolute_error_mm"] == 50 - constant
    assert not row["timing_defined"]
    assert pd.isna(row["forecast_peak_time"])
    assert np.isnan(row["absolute_timing_error_min"])
    assert "timing_penalty_applied" not in row
    assert np.isnan(row["timing_error_min"])
    assert row["forecast_flat"]
    assert row["forecast_all_zero"] == (constant == 0)


def test_single_slot_episode_has_no_identifiable_forecast_peak():
    row = mod.episode_features([22], times(1), [20])
    assert row["observed_peak_absolute_error_mm"] == row["peak_absolute_error_mm"] == 2
    assert not row["timing_defined"]
    assert np.isnan(row["timing_error_min"])
    assert pd.isna(row["forecast_peak_time"])
    assert np.isnan(row["absolute_timing_error_min"])


@pytest.mark.parametrize("observed", [[50, 5, 5, 5], [5, 5, 50, 5], [5, 5, 5, 50]])
def test_flat_forecast_timing_is_undefined_regardless_of_observed_peak_position(observed):
    row = mod.episode_features([.2] * 4, times(4), observed)
    assert np.isnan(row["absolute_timing_error_min"])
    assert pd.isna(row["forecast_peak_time"])
    assert np.isnan(row["timing_error_min"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_forecast_values_raise(bad):
    with pytest.raises(base.DiagnosticError):
        mod.episode_features([5, bad, 20], times(3), [5, 25, 5])


def test_matching_aligns_stations_and_leads_and_requires45cells_per_episode(tmp_path):
    rows, audit = run_case(synthetic_case(tmp_path))
    assert rows.groupby("event_id").size().tolist() == [45, 45]
    assert audit.status.eq("included").all()
    assert rows.timing_sample.all()
    row = rows.query("route == 'pysteps_cnn' and member == 2 and lead_min == 90").iloc[0]
    assert row.prediction_at_observed_peak_mm == pytest.approx(26.29)
    assert row.forecast_max_mm == pytest.approx(50.29)
    assert row.observed_peak_absolute_error_mm == pytest.approx(23.71)
    assert row.peak_absolute_error_mm == pytest.approx(.29)
    assert row.timing_error_min == 10
    assert row.observed_peak_issuance_time == row.observed_peak_time - pd.Timedelta(minutes=90)


@pytest.mark.parametrize("failure", ["missing_anchor", "nan", "negative", "infinite"])
def test_one_bad_nonpeak_forecast_slot_excludes_entire_episode_allroutes(tmp_path, failure):
    case = synthetic_case(tmp_path)
    cfg, stamp, truth, stores, anchors, audit = case
    at = forecast_cell(stores[-1], stamp[1])
    if failure == "missing_anchor":
        anchors = np.delete(anchors, at[0])
        for store in stores:
            store.anchors_ns = np.delete(store.anchors_ns, at[0])
            store.predictions = np.delete(store.predictions, at[0], axis=0)
    else:
        stores[-1].predictions[at] = {"nan": np.nan, "negative": -1, "infinite": np.inf}[failure]
    rows, updated = run_case((cfg, stamp, truth, stores, anchors, audit))
    assert rows.event_id.nunique() == 1
    assert len(rows) == 45
    assert updated.iloc[0].status != "included"
    assert updated.iloc[1].status == "included"


def test_one_flat_forecast_excludes_episode_from_timing_for_every_route_member_and_lead(tmp_path):
    case = synthetic_case(tmp_path)
    _, stamp, _, stores, _, _ = case
    for time in stamp[1:5]:
        stores[-1].predictions[forecast_cell(stores[-1], time)] = 0
    rows, audit = run_case(case)
    assert len(rows) == 90
    assert audit.status.eq("included").all()
    assert audit.timing_sample.tolist() == [False, True]
    assert audit.timing_status.iloc[0] == "constant_forecast_in_any_route_member_or_lead"
    first = rows.loc[rows.event_id.eq(audit.iloc[0].event_id)]
    assert first.timing_defined.sum() == 44
    assert not first.timing_sample.any()
    flat = first.loc[first.forecast_flat].iloc[0]
    assert np.isnan(flat.absolute_timing_error_min)
    assert pd.isna(flat.forecast_peak_time)
    assert pd.isna(flat.forecast_peak_issuance_time)
    assert np.isnan(flat.timing_error_min)
    assert first.loc[~first.forecast_flat, "absolute_timing_error_min"].eq(10).all()
    members, routes, per_episode = mod.summarize(rows)
    assert members.n_episodes.eq(2).all()
    assert members.n_timing_episodes.eq(1).all()
    assert routes.n_episodes.eq(2).all()
    assert routes.n_timing_episodes.eq(1).all()
    assert routes.timing_mae_min.eq(10).all()
    assert len(per_episode) == 30
    assert per_episode.loc[per_episode.timing_sample, "timing_mae_min"].eq(10).all()
    assert per_episode.loc[~per_episode.timing_sample, "timing_mae_min"].isna().all()
    assert per_episode.observed_peak_mae_mm.notna().all()
    route = routes.query("period == 'pooled' and route == 'exprecast_cnn' and lead_min == 180").iloc[0]
    assert route.timing_mae_min == 10


def test_cached_timing_policy_preserves_amounts_and_nonflat_errors(tmp_path):
    rows, audit = run_case(synthetic_case(tmp_path))
    rows.loc[0, "forecast_flat"] = True
    rows.loc[0, "timing_defined"] = False
    rows.loc[0, "forecast_peak_time"] = pd.NaT
    rows.loc[0, "timing_error_min"] = np.nan
    rows.loc[0, "absolute_timing_error_min"] = np.nan
    prior_amount = rows[["observed_peak_absolute_error_mm", "peak_absolute_error_mm"]].copy()
    updated, updated_audit = mod.apply_timing_policy(rows, audit)
    pd.testing.assert_frame_equal(prior_amount, updated[prior_amount.columns])
    assert updated.absolute_timing_error_min.isna().sum() == 1
    assert np.isnan(updated.loc[0, "absolute_timing_error_min"])
    assert updated.loc[1:, "absolute_timing_error_min"].eq(10).all()
    assert updated_audit.timing_sample.tolist() == [False, True]
    assert updated.groupby("event_id").timing_sample.nunique().eq(1).all()
    assert pd.isna(updated.loc[0, "forecast_peak_time"])


def test_cached_policy_rejects_spurious_peak_for_flat_forecast(tmp_path):
    rows, audit = run_case(synthetic_case(tmp_path))
    rows.loc[0, "forecast_flat"] = True
    rows.loc[0, "timing_defined"] = False
    with pytest.raises(base.DiagnosticError, match="must not be assigned"):
        mod.apply_timing_policy(rows, audit)



def test_summary_rejects_method_specific_timing_samples(tmp_path):
    rows, _ = run_case(synthetic_case(tmp_path))
    rows.loc[0, "timing_sample"] = False
    with pytest.raises(base.DiagnosticError, match="common nonconstant subset"):
        mod.summarize(rows)


def test_summary_averages_absolute_errors_not_member_predictions(tmp_path):
    case = synthetic_case(tmp_path)
    _, stamp, _, stores, _, _ = case
    for store in stores:
        for peak in stamp[[2, 9]]:
            for lead in store.leads:
                store.predictions[forecast_cell(store, peak, lead=int(lead))] = [40, 60, 50][store.member]
        for secondary in stamp[[3, 10]]:
            for lead in store.leads:
                store.predictions[forecast_cell(store, secondary, lead=int(lead))] = 5
    rows, audit = run_case(case)
    members, routes, per_episode = mod.summarize(rows)
    assert len(members) == 90  # pooled + 2024; 9 members × 5 leads.
    assert len(routes) == 30
    np.testing.assert_allclose(routes.observed_peak_mae_mm, 20 / 3)
    np.testing.assert_allclose(routes.peak_mae_mm, 20 / 3)
    np.testing.assert_allclose(routes.timing_mae_min, 0)
    np.testing.assert_allclose(per_episode.observed_peak_mae_mm, 20 / 3)
    stats = mod.duration_statistics(audit)
    pooled = stats.loc[stats.period.eq("pooled")]
    assert pooled.n_episodes.eq(2).all()
    np.testing.assert_allclose(pooled.median_duration_hours, 2 / 3)


def test_missing_route_member_raises(tmp_path):
    cfg, stamp, truth, stores, anchors, audit = synthetic_case(tmp_path)
    with pytest.raises(base.DiagnosticError, match="Exactly three"):
        run_case((cfg, stamp, truth, stores[:-1], anchors, audit))


def test_different_store_issuance_axis_raises(tmp_path):
    case = synthetic_case(tmp_path)
    store = case[3][-1]
    store.anchors_ns = store.anchors_ns[1:]
    store.predictions = store.predictions[1:]
    with pytest.raises(base.DiagnosticError, match="issuance axes differ"):
        run_case(case)


@pytest.mark.parametrize("override", [{"episode_threshold_mm": 0}, {"episode_threshold_mm": 21},
                                      {"qc_upper_mm": 20}, {"qc_upper_mm": np.inf}])
def test_invalid_threshold_order_raises(tmp_path, override):
    with pytest.raises(base.DiagnosticError):
        config(tmp_path, **override)


def test_histogram_ten_minute_bins_and_hourly_ticks_keep_entire_tail():
    durations = np.array([60, 70, 80, 90, 940])
    edges, ticks = mod.duration_histogram_grid(durations)
    np.testing.assert_allclose(np.diff(edges) * 60, 10)
    np.testing.assert_array_equal(np.diff(ticks), 1)
    counts, _ = np.histogram(durations / 60, bins=edges)
    assert counts.sum() == len(durations)
    assert (counts == 1).sum() == len(durations)
    assert ticks[-1] == 16
    left_edges_minutes = edges[:-1] * 60
    np.testing.assert_allclose(left_edges_minutes[counts.astype(bool)], durations)
    assert edges[0] == 0
    assert edges[-1] * 60 == 950


def test_histogram_hour_tick_is_boundary_and_exact_hour_enters_right_bin():
    durations = np.array([50, 60, 60, 70, 120])
    edges, ticks = mod.duration_histogram_grid(durations)
    edges_minutes = np.rint(edges * 60).astype(np.int64)
    counts, _ = np.histogram(durations, bins=edges_minutes)
    assert 1 in edges and 1 in ticks
    assert counts[5] == 1  # [50,60): only 50 min
    assert counts[6] == 2  # [60,70): both 60 min durations
    assert counts[7] == 1  # [70,80): only 70 min
    assert counts[12] == 1  # [120,130): retain the maximum in a full bin
    assert counts.sum() == len(durations)


@pytest.mark.parametrize("durations", [[], [np.nan], [0], [-10], [65]])
def test_histogram_rejects_invalid_durations(durations):
    with pytest.raises(base.DiagnosticError):
        mod.duration_histogram_grid(durations)


def test_histogram_draws_one_population_with_count_and_no_legend(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    audit = pd.DataFrame({"observation_status": ["candidate", "candidate", "below_peak_threshold"],
                          "duration_min": [60, 70, 90]})
    mod.plot_duration_histogram(audit, tmp_path)
    ax = figures[0].axes[0]
    assert len(ax.patches) == 1
    assert ax.get_legend() is None
    assert [text.get_text() for text in ax.texts] == ["n = 2 station-episodes"]
    assert ax.patches[0].get_data().values.sum() == 2


def test_mae_plot_uses_zero_origin_shared_amount_scale_and_marked_lines(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    monkeypatch.setattr(mod, "plot_duration_histogram", lambda *args: None)
    summary = pd.DataFrame([
        {"period": "pooled", "route": route, "lead_min": lead,
         "n_episodes": 12, "n_timing_episodes": 10,
         "observed_peak_mae_mm": 28, "peak_mae_mm": 23, "timing_mae_min": 54}
        for route in base.ROUTES for lead in (60, 90, 120, 150, 180)
    ])
    mod.plot_results({"route_metrics": summary, "episode_audit": pd.DataFrame()}, tmp_path)
    axes = figures[0].axes
    assert [ax.get_ylim() for ax in axes] == [(0, 35), (0, 35), (0, 60)]
    for ax in axes:
        assert "n =" not in ax.get_title(loc="left")
        assert len(ax.lines) == 3
        assert all(line.get_marker() == "o" for line in ax.lines)
        np.testing.assert_array_equal(ax.get_xticks(), [60, 90, 120, 150, 180])


def portable_inputs(tmp_path):
    """Write only small synthetic CSV/NPY inputs using the public manifest contract."""
    cfg, stamp, truth, stores, anchors, _ = synthetic_case(tmp_path)
    dates = pd.DatetimeIndex(stamp)
    gauges = pd.DataFrame({"Year": dates.year, "Month": dates.month, "Day": dates.day,
                           "Hour": dates.hour, "Minute": dates.minute,
                           "g7": truth[:, 0], "g11": truth[:, 1]})
    gauges.to_csv(cfg.gauge_csv, index=False)
    pd.DataFrame({"station_id": [7, 11], "target_col": ["g7", "g11"]}).to_csv(
        cfg.stations_csv, index=False)

    def write_record(name, axes, values):
        record = {"completed": True}
        for key, value in zip(("anchors_ns", "station_ids", "lead_minutes", "values"), (*axes, values)):
            relative = f"{name}_{key}.npy"
            np.save(tmp_path / relative, value, allow_pickle=False)
            record[key] = relative
        return record

    manifest = {"timestamp_convention": "KST-naive", "forecasts": []}
    for store in stores:
        record = write_record(f"{store.route}_{store.member}",
                              (store.anchors_ns, store.station_ids, store.leads), store.predictions)
        record.update(route=store.route, member=store.member, seed=store.seed)
        manifest["forecasts"].append(record)
    station_ids = np.array([11, 7])
    leads = np.array([180, 120, 60, 90, 150])
    cube = np.full((len(anchors), 2, len(leads)), np.nan)
    for j, lead in enumerate(leads):
        positions = pd.Index(stamp).get_indexer(anchors + lead * base.MINUTE_NS)
        valid = positions >= 0
        cube[valid, :, j] = truth[positions[valid]][:, [1, 0]]
    manifest["truth"] = write_record("truth", (anchors, station_ids, leads), cube)
    cfg.forecast_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return cfg, manifest


def cli_arguments(cfg):
    return ["--gauge-csv", str(cfg.gauge_csv), "--stations-csv", str(cfg.stations_csv),
            "--forecast-manifest", str(cfg.forecast_manifest), "--output-dir", str(cfg.output_dir),
            "--years", "2024", "--months", "6", "--expected-stations", "2", "--no-plots"]


def test_portable_manifest_full_analysis_crosschecks_original_gauges(tmp_path):
    cfg, _ = portable_inputs(tmp_path)
    result = mod.run_analysis(cfg, progress=lambda _: None)
    assert result["provenance"]["n_amount_episodes"] == 2
    assert result["provenance"]["n_timing_episodes"] == 2
    assert result["route_metrics"].timing_mae_min.eq(10).all()
    assert "manifest targets" in result["provenance"]["truth_crosscheck"]
    assert not cfg.output_dir.exists()


def test_manifest_truth_mismatch_rejected_at_nonpeak_time(tmp_path):
    cfg, manifest = portable_inputs(tmp_path)
    path = tmp_path / manifest["truth"]["values"]
    values = np.load(path)
    values[np.isfinite(values)] += 0.2
    np.save(path, values)
    with pytest.raises(mod.DiagnosticError, match="differ from manifest RN60"):
        mod.run_analysis(cfg, progress=lambda _: None)


@pytest.mark.parametrize("mutation", ["missing_member", "incomplete", "timezone", "lead", "station", "issuance"])
def test_portable_manifest_rejects_inconsistent_contract(tmp_path, mutation):
    cfg, manifest = portable_inputs(tmp_path)
    if mutation == "missing_member":
        manifest["forecasts"].pop()
    elif mutation == "incomplete":
        manifest["forecasts"][0]["completed"] = False
    elif mutation == "timezone":
        manifest["timestamp_convention"] = "UTC"
    else:
        key = {"lead": "lead_minutes", "station": "station_ids", "issuance": "anchors_ns"}[mutation]
        path = tmp_path / manifest["truth"][key]
        values = np.load(path)
        values[0] += 10 if mutation == "lead" else 1
        np.save(path, values)
    cfg.forecast_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mod.DiagnosticError):
        base.load_forecast_stores(cfg)


def test_gauge_duplicate_timestamps_rejected(tmp_path):
    cfg, _ = portable_inputs(tmp_path)
    gauges = pd.read_csv(cfg.gauge_csv)
    pd.concat([gauges, gauges.iloc[[1]]]).to_csv(cfg.gauge_csv, index=False)
    with pytest.raises(mod.DiagnosticError, match="Duplicate or unexpected gauge timestamps"):
        mod.run_analysis(cfg, progress=lambda _: None)


def test_preflight_cli_does_not_create_outputs(tmp_path, capsys):
    cfg, _ = portable_inputs(tmp_path)
    mod.main([*cli_arguments(cfg), "--preflight-only"])
    report = json.loads(capsys.readouterr().out)
    assert report["ready_paths"] and not report["prediction_payload_read"]
    assert not cfg.output_dir.exists()


def test_public_module_cli_exports_complete_results_without_project_layout(tmp_path):
    cfg, _ = portable_inputs(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(Path(mod.__file__).resolve().parents[1]))
    completed = subprocess.run([sys.executable, "-m", "evaluation.rainfall_episode_peak",
                                *cli_arguments(cfg)], cwd=tmp_path, env=env,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((cfg.output_dir / "COMPLETED.json").read_text())
    assert manifest["completed"]
    assert manifest["n_amount_episodes"] == manifest["n_timing_episodes"] == 2
    assert "episode_errors.csv" in manifest["files_sha256"]
    assert (cfg.output_dir / "route_metrics.csv").is_file()


def test_no_timing_episodes_remains_valid_amount_analysis(tmp_path):
    case = synthetic_case(tmp_path)
    for store in case[3]:
        store.predictions[:] = 0
    rows, audit = run_case(case)
    members, routes, episodes = mod.summarize(rows)
    assert routes.n_episodes.eq(2).all()
    assert routes.n_timing_episodes.eq(0).all()
    assert routes.timing_mae_min.isna().all()
    assert routes.observed_peak_mae_mm.eq(50).all()
    mod.plot_results({"route_metrics": routes, "episode_audit": audit}, tmp_path)
    assert (tmp_path / "episode_peak_mae.png").is_file()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_descending_integer_issuance_axis_is_rejected_without_unsigned_underflow(dtype):
    store = base.ForecastStore("direct_r2p", 0, 0,
        np.array([20 * base.MINUTE_NS, 10 * base.MINUTE_NS], dtype=dtype),
        np.array([7]), np.array([60]), np.zeros((2, 1, 1)))
    with pytest.raises(mod.DiagnosticError, match="strictly increasing"):
        store.validate()


def test_preflight_cli_rejects_station_csv_mismatch(tmp_path):
    cfg, _ = portable_inputs(tmp_path)
    stations = pd.read_csv(cfg.stations_csv)
    stations.loc[0, "station_id"] = 999
    stations.to_csv(cfg.stations_csv, index=False)
    with pytest.raises(SystemExit) as error:
        mod.main([*cli_arguments(cfg), "--preflight-only"])
    assert error.value.code == 2
    assert not cfg.output_dir.exists()
