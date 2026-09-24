# Observation-defined RN60 episode verification

Run the installed module with Python 3.12 and the package's NumPy, pandas and
Matplotlib dependencies.

```bash
python -m evaluation.rainfall_episode_peak \
  --gauge-csv /path/to/gauge_rn60.csv \
  --stations-csv /path/to/evaluated_stations.csv \
  --forecast-manifest /path/to/inputs.json \
  --output-dir /path/to/episode_results \
  --years 2024 2025 --months 6 7 8 9 \
  --leads 60 90 120 150 180 --expected-stations 128
```

Add `--preflight-only` to check input headers, axes and declared completion
without scoring or writing results. Add `--no-plots` to export tables only.
The output directory must be dedicated to this analysis; if nonempty, a new
timestamp-suffixed directory is created rather than replacing existing files.
No source input is modified. The optional `--no-float16-roundtrip` changes the
reported storage-precision policy and is recorded in provenance; omit it for
the reported analysis. The default years, months, leads and thresholds match
the reported protocol, but sample counts are calculated, never hard-coded.

## Input contract

The gauge CSV has `Year,Month,Day,Hour,Minute` plus original, unnormalized RN60
columns in mm. Timestamps are KST wall-clock components, with no UTC conversion
or additional nine-hour offset. The loader builds the complete ten-minute grid
for the selected calendar months and years; missing timestamps remain missing.
Five-minute source rows between ten-minute grid points are not used. RN60 is
already trailing-hour accumulation: do not shift or reaccumulate it.

The station CSV explicitly lists only the stations to evaluate, for example:

```csv
station_id,target_col
101,rn60_101
102,rn60_102
```

Station IDs and target columns must be unique. The caller is responsible for
providing the prespecified held-out set; the diagnostic does not infer training
membership from station numbers. For the reported analysis this is the frozen
128-station set, not all 642 stations.

The JSON manifest declares `timestamp_convention: "KST-naive"`, a `forecasts`
list, and a `truth` object. Each array record contains these entries:

```json
{
  "completed": true,
  "values": "arrays/predictions.npy",
  "anchors_ns": "arrays/anchor_times_ns.npy",
  "station_ids": "arrays/station_ids.npy",
  "lead_minutes": "arrays/lead_minutes.npy"
}
```

Each forecast record additionally declares `route`, `member` and `seed`.
Exactly nine records are required: `direct_r2p`, `pysteps_cnn` and
`exprecast_cnn`, each with members `0,1,2`. Seed is an integer run identifier;
members remain separate throughout scoring. Records may share axis files.
`completed: true` is an explicit caller attestation that export is finished;
it is not an independent verification of model fitting or checkpoint identity.

All paths are user-controlled; relative paths resolve against the manifest's
directory. Arrays are plain, non-object NPY files, loaded read-only:

- `values`: real numeric `(issuance, station, lead)` RN60 values **in mm**.
- `anchors_ns`: strictly increasing integer nanoseconds encoding KST-naive
  issuance times, on the ten-minute grid. Every store must share this axis.
- `station_ids`: unique integer IDs. Store order may differ; sets must match
  the evaluated station set exactly.
- `lead_minutes`: unique positive integer minutes. All requested leads must
  exist, but their order and additional exported leads may differ.

The `truth` record follows the same four-file layout, without route/member/seed.
It contains verifying RN60 at `issuance + lead`, not observations at issuance.
Its axes must align with the forecast support. Original gauge values in every
candidate episode with common issuance support are cross-checked against this
truth cube at every requested lead to absolute tolerance `1e-4 mm`. A mismatch,
including nonfinite truth, is an error. This check does not replace the original
gauge time series: episode selection always uses the gauge CSV independently.

A minimal manifest-building example, assuming files already exported in
`arrays/<route>_<member>/` and `arrays/truth/`, is:

```python
import json
from pathlib import Path

def record(directory):
    return {
        "completed": True,
        "values": f"{directory}/rn60_mm.npy",
        "anchors_ns": f"{directory}/anchor_times_ns.npy",
        "station_ids": f"{directory}/station_ids.npy",
        "lead_minutes": f"{directory}/lead_minutes.npy",
    }

forecasts = []
for route in ("direct_r2p", "pysteps_cnn", "exprecast_cnn"):
    for member in range(3):
        forecasts.append({
            **record(f"arrays/{route}_{member}"),
            "route": route, "member": member,
            "seed": member if route == "direct_r2p" else 21040 + member,
        })
manifest = {"timestamp_convention": "KST-naive", "forecasts": forecasts,
            "truth": record("arrays/truth")}
Path("inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

## Scientific protocol

The analysis uses the following protocol:

1. Find maximal consecutive observed RN60 >=5 mm runs on the ten-minute grid;
   retain runs whose maximum is >=20 mm. Do not split at midnight. Both adjacent
   observations must be valid and below 5 mm. Missing/negative/nonfinite
   observations and evaluation gaps censor wet fragments; they are not dry
   boundaries. Any run containing RN60 >=300 mm is excluded in full as analysis
   QC, without clipping or selecting a replacement peak.
2. Require common issuance support and finite, nonnegative predictions for every
   episode time, route, member and requested lead. A bad nonpeak forecast value
   excludes the entire episode from all routes, not just the affected score.
   Prediction float16 round-trip harmonization precedes these checks and scoring.
3. For each member independently, compute absolute errors in forecast RN60 at
   the observed maximum, independently identified episode-maximum amount, and
   the timestamps of those two independent maxima. Exact ties choose the
   earliest timestamp, not the nearest observed peak.
4. For timing only, remove a whole station-episode if **any** forecast series is
   constant over it, across **all** routes, members and requested leads. Both
   amount errors retain these episodes. Flat forecasts have no defined peak
   timestamp or timing error; no maximum-distance penalty, time cap, wrapping
   or nearest-local-peak matching is used. If every episode is excluded from
   timing, its sample count is zero and timing scores are NaN, not zero.
5. Average episode errors within each member, then average the three member
   scores. Never find peaks on a three-member mean forecast series.

Each fixed-lead series consists of successive issuances, not one issuance's
trajectory. RN60 values are never summed as event rainfall. Duration is occupied
ten-minute slots times ten; first-to-last timestamp span is also exported.
Histogram bins are ten-minute `[left,right)` intervals, including the full tail.
These are station-episodes, not independently tracked storms. Multiple local
peaks can affect the independent-maximum timing error.

The completed reference analysis contains **1,437 amount episodes** and
**1,391 common nonconstant timing episodes**. Those are results of its inputs,
not counts imposed on new data. Neither raw gauge observations nor large raw
forecast arrays are supplied by this module.

## Outputs and checks

Outputs include individual errors, member-mean episode errors, the full wet-run
audit and exclusions, QC observations, per-member and route metrics, duration
and station statistics, and forecast quality flags. With plotting enabled, the
module also produces zero-origin three-panel MAE and duration-histogram PNG/PDF
figures. The two amount panels share a y scale; plots show point estimates, not
confidence intervals. Provenance records input paths/statistics, manifest/code
hashes and the configuration. Raw-array file content hashes are not verified;
the output `COMPLETED.json` does hash the generated output files.

```bash
python -m pytest tests/test_rainfall_episode_peak.py
```

Tests use synthetic CSV/NPY inputs only, including permuted station/lead axes,
truth mismatches, missing and invalid values, midnight and boundary censoring,
QC, earliest ties, common timing masks, member-first errors and the public CLI.
