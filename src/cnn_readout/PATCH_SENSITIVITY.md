# CNN patch-size sensitivity

Compare centered 3 × 3 and 5 × 5 readouts of the same exPreCast forecast
fields. Attention pooling keeps both models at **51,282 trainable parameters**.
The 3 × 3 route remains the primary comparator; this experiment evaluates
within-route patch sensitivity, not the cause of Direct R2P's advantage.

Run `python -m cnn_readout.patch_sensitivity --help` after installing the
package, or set `PYTHONPATH=src` from the repository root.

## Prepared input arrays

Supply a JSON source configuration and plain, non-object NPY arrays. Paths
resolve relative to the configuration file. For example:

```json
{
  "schema": "cnn_patch_sensitivity_source_v1",
  "patch_normalization": "nonnegative_dBZ_divided_by_100",
  "issue_time_convention": "KST-naive_int64_nanoseconds",
  "target_units": "mm",
  "arrays": {
    "patches18": "arrays/patches18.npy",
    "issue_times_ns": "arrays/issue_times_ns.npy",
    "station_ids": "arrays/station_ids.npy",
    "xy_norm": "arrays/xy_norm.npy",
    "targets_mm": "arrays/targets_mm.npy",
    "context36": "arrays/context36.npy",
    "station_folds": "arrays/station_folds.npy",
    "oof_context36": "arrays/oof_context36.npy"
  }
}
```

Let `I` be issuance count and `S` station count.

| Array | Shape | Meaning |
|---|---|---|
| `patches18` | `[I,S,18,5,5]` | Station-centered normalized reflectivity (`nonnegative dBZ / 100`) at +10,+20,…,+180 min. |
| `issue_times_ns` | `[I]` | Strictly increasing integer KST-naive issuance timestamps in nanoseconds. |
| `station_ids` | `[S]` | Unique integer station IDs. |
| `xy_norm` | `[S,2]` | Query coordinates using the same normalization as the readout. |
| `targets_mm` | `[I,S,13]` | Gauge RN60 in mm at +60,+70,…,+180 min; NaN denotes unavailable truth. Required for selection, fitting and comparison. |
| `context36` | `[I,S,36]` | Issuance-time gauge summary for final fitting and prediction. |
| `station_folds` | `[S]` | Fixed fold labels 0,1,2; required for out-of-fold selection. |
| `oof_context36` | `[3,I,S,36]` | Selection context; slice `f` must exclude fold-`f` stations from its gauge summary. |

For RN60 ending at lead `L`, the readout uses six consecutive fields at
`L−50,…,L` min. The 3 × 3 input is the centered crop of the same 5 × 5
source. Inputs are not rain-rate or preaccumulated-RN60 fields. Patch and
auxiliary features must be finite.

Extract 5 × 5 patches from exported exPreCast fields with the station mapping:

```bash
python -m exprecast_adapter.adapter extract-patches \
  --field-h5 data/private/forecast_fields.h5 \
  --mapping-csv data/private/station_mapping.csv \
  --patch-size 5 --output outputs/patch_sensitivity/extracted_patches
```

This produces `patches18_f16.npy`, `issue_times_ns.npy` and `station_ids.npy`.
Use these paths in the source configuration and supply the corresponding
targets, normalized coordinates, gauge context and folds separately.

The 36 context entries summarize 12 five-minute RN60 values, 12 RN15 values
and 12 availability values, using fitting-station information available by
issuance. For the reported configuration, select up to ten fitting stations
within 30 km by coordinates, apply a Gaussian kernel with 6-km scale, and
average available `log1p` RN60/RN15 with normalized spatial weights. Retain the
available spatial-weight fraction. Held-out evaluation stations must not
supply context. Missing values must follow the numeric-zero/availability
convention before arrays are supplied. The readout appends normalized `x,y`
and `lead/180`, giving 39 auxiliary features.

Prepare fitting and evaluation inputs separately. The reported fitting set
contains 2,887 hourly issuances during June–September 2023 at 514 fitting
stations; evaluation uses 35,088 common issuances in June–September 2024–2025
at 128 held-out stations. Omit selection-only fold arrays from evaluation
inputs if unused.

```bash
python -m cnn_readout.patch_sensitivity prepare \
  --config data/private/patch_sensitivity/source_2023.json \
  --output outputs/patch_sensitivity/prepared_2023.json
python -m cnn_readout.patch_sensitivity prepare \
  --config data/private/patch_sensitivity/source_2024_2025.json \
  --output outputs/patch_sensitivity/prepared_2024_2025.json
```

Preparation records array hashes, shapes and dtypes. Later stages verify these
records by default; regenerate the manifest if the source arrays change.

## Epoch selection, final fitting and prediction

```bash
python -m cnn_readout.patch_sensitivity select-epoch \
  --input outputs/patch_sensitivity/prepared_2023.json \
  --patch-size 5 --max-epochs 50 \
  --work-dir outputs/patch_sensitivity/oof_patch5 \
  --output outputs/patch_sensitivity/selection_patch5.json --device cuda
python -m cnn_readout.patch_sensitivity fit-final \
  --input outputs/patch_sensitivity/prepared_2023.json \
  --selection outputs/patch_sensitivity/selection_patch5.json \
  --output-dir outputs/patch_sensitivity/checkpoints_patch5 \
  --seeds 21040,21041,21042 --device cuda
python -m cnn_readout.patch_sensitivity predict \
  --input outputs/patch_sensitivity/prepared_2024_2025.json \
  --checkpoint-dir outputs/patch_sensitivity/checkpoints_patch5 \
  --output-dir outputs/patch_sensitivity/predictions_patch5 --device cuda
```

To independently refit the 3 × 3 readout, use `--patch-size 3` during selection
and separate selection, checkpoint and prediction paths. Do not replace the
fixed primary comparator's outputs.

The selection and training settings are:

- Four held months (June–September 2023) × three held station folds. Each cell
  trains outside both its held month and held fold, and validates inside both.
  Issuances within six hours of an internal month boundary are purged.
- Pool contingency counts across all 12 cells, then maximize equally weighted
  macro CSI over 13 leads (+60,…,+180 min) and thresholds 1,5,10,20 mm.
  The earliest epoch wins an exact tie. Select independently for each patch
  size; both sizes selected epoch 4 in the reported experiment.
- OOF cell seed: `7131000 + 100 * held_month_index + held_station_fold`, where
  month index is 0–3. Final refit seeds: `21040,21041,21042`.
- AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size 4,096 and up to
  2,097,152 samples per epoch. Target-stratified sampling uses exact `p_s/q_s`
  importance correction with all target-range weights equal to one.
- Targets use the fixed fitting-station RN60 normalization `RN60 / 735.0`.
  The objective is normalized RN60 mean-square error; changing patch size
  does not change the field source, target scale, context or loss.

`--execution-months` and `--execution-folds` can distribute OOF cells across
runs using the same work directory. Month indices are 0–3 and folds 0–2;
each option accepts `all` or a comma-separated subset. Selection is finalized
only after all 12 cells are complete. Reduced epoch/sample settings are useful
for small tests but do not reproduce the reported training configuration.

## Matched comparison

Compare three-member 5 × 5, fixed primary 3 × 3, and Direct R2P forecasts on
the same truth, issuance, station and lead support. Prediction values are in
mm. The comparison applies a common float16 round trip and common finite
support across every supplied route/member, computes each member's metrics,
then averages the three member scores.

```bash
python -m cnn_readout.patch_sensitivity compare \
  --truth-input outputs/patch_sensitivity/prepared_2024_2025.json \
  --route exPreCast_CNN3=data/private/patch_sensitivity/primary_patch3.json \
  --route exPreCast_CNN5=outputs/patch_sensitivity/predictions_patch5 \
  --route Direct_R2P=data/private/patch_sensitivity/direct_predictions.json \
  --contrast CNN5_minus_CNN3:exPreCast_CNN5:exPreCast_CNN3 \
  --contrast Direct_minus_CNN5:Direct_R2P:exPreCast_CNN5 \
  --contrast Direct_minus_CNN3:Direct_R2P:exPreCast_CNN3 \
  --output-dir outputs/patch_sensitivity/comparison \
  --bootstrap-resamples 2000 --bootstrap-seed 20260916
```

Each contrast uses `NAME:LEFT:RIGHT` and reports `LEFT − RIGHT` CSI.

`--route` accepts either a completed `predict` output directory or an external
prediction JSON. For existing 3 × 3 or Direct R2P predictions, provide:

```json
{
  "schema": "cnn_patch_sensitivity_external_predictions_v1",
  "predictions": "arrays/predictions_mm.npy",
  "issue_times_ns": "arrays/issue_times_ns.npy",
  "station_ids": "arrays/station_ids.npy",
  "lead_minutes": "arrays/lead_minutes.npy",
  "sha256": {
    "predictions": "<prediction-file SHA256>",
    "issue_times_ns": "<issuance-axis SHA256>",
    "station_ids": "<station-axis SHA256>",
    "lead_minutes": "<lead-axis SHA256>"
  }
}
```

Replace each hash placeholder with the corresponding file's SHA256, for
example using `sha256sum`; all four hashes are required under default
verification. Store user-supplied source and external-prediction configurations
under `data/private/patch_sensitivity/` as in the commands above.

The external prediction array is floating-point `[3,I,S,13]`, with members
kept separate. Axes must match the evaluation input exactly, including station
order and the 13 target leads +60,+70,…,+180 min. Select these leads from any
more densely spaced model output before supplying the array. Relative paths
resolve against this JSON file. A generated prediction directory instead
contains `member_0.npy` through `member_2.npy`, the three named axis files,
`contract.json` and `COMPLETED.json`.

Reported comparisons use +60,+90,+120,+150,+180 min and 1,5,10,20 mm
thresholds. Paired pointwise 95% percentile intervals resample KST issuance
dates 2,000 times with seed `20260916`, using the same date multiplicities
for all compared routes. Member statistics are recomputed before contrasts;
stations remain fixed.

Comparison outputs are `metrics_per_member.csv`, `metrics_mean_of_members.csv`,
`paired_CSI_intervals.csv`, `daily_statistics.npz` and `manifest.json`. The CSVs
contain the five reported leads; the daily statistics retain all 13 target
leads for audit and aggregation.

Render the newly computed comparison using the route and contrast names above:

```bash
python -m paper_outputs.render_additional_verification \
  --item patch --patch-results outputs/patch_sensitivity/comparison \
  --output-dir outputs/patch_sensitivity/figures
```
