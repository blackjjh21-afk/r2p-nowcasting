# CNN

This package contains the field-to-point readout used after the frozen
pySTEPS and exPreCast field forecasters. For RN60 ending at lead `L`, the
model receives the six lead-aligned 3×3 forecast patches at
`L-50, L-40, ..., L` min and a 39-entry auxiliary vector. The auxiliary vector
contains normalized query coordinates and lead (three entries) followed by a
36-entry summary constructed only from fitting-station gauge histories
available by forecast issuance.

The model predicts normalized RN60 and is fitted with mean squared error.
Its target scale is the frozen Direct R2P min-max transform fitted on
2019–2022 fitting-station observations; the same scale is used for every
2023 OOF cell and final refit, without using evaluation-period targets. The
optional deterministic target-stratified sampler uses exact `p_s/q_s`
importance correction, so it changes tuple coverage but not the unweighted
natural-prevalence objective.

## Prepared NPZ contract

The command-line workflow starts from a prepared NPZ and never opens private
radar, gauge, or forecast archives. The canonical schema is dense over issue
time, station, and lead:

| key | dtype and shape | meaning |
|---|---|---|
| `schema` | scalar string | exactly `cnn_readout_prepared_npz_v1` |
| `patches` | float32 `[N,S,L,6,3,3]` | six normalized-reflectivity forecast patches per tuple |
| `issue_times_ns` | int64 `[N]` | strictly increasing issue times in Unix nanoseconds; use the study's local issue-time convention consistently |
| `station_ids` | int64 `[S]` | unique nonnegative station identifiers |
| `station_folds` | int8 `[S]` | fixed station folds `0`, `1`, or `2` |
| `lead_minutes` | int16 `[13]` | exactly `+60,+70,...,+180` min, the RN60 target-lead axis |
| `targets_mm` | float32 `[N,S,L]` | gauge RN60 in millimetres; required for selection/fitting, optional for prediction; `NaN` marks unavailable truth |
| `auxiliary` | float32 `[N,S,L,39]`, optional | deployment/final-refit auxiliary features; omission explicitly requests an all-zero (no-context) vector |
| `oof_auxiliary` | float32 `[3,N,S,L,39]`, optional | leakage-safe OOF auxiliary features; entry `f` must exclude fold-`f` stations from its issuance-time gauge summary |

`oof_auxiliary` is required by `select-epoch`. Keeping it separate from
`auxiliary` makes the month×station-fold selection contract explicit: a held
station fold cannot leak into its own context summary. Missing gauge values
inside the 36-entry summary must already be represented by the adapter's
numeric value/missingness convention; all patch and auxiliary entries must be
finite.

A flat compatibility schema is also accepted for adapters that emit eligible
tuples directly. It uses `patches [T,6,3,3]`, `issue_time_ns [T]`,
`station_id [T]`, `station_fold [T]`, `lead_min [T]`, optional
`target_mm [T]`, optional `auxiliary [T,39]`, and optional
`oof_auxiliary [3,T,39]`. The singular and plural key sets must not be mixed.
Flat inputs may contain any subset of the same `+60,...,+180` min target-lead
axis. Leads below 60 min are invalid because a tuple already represents the
six-field window ending at its RN60 target lead.

## Validate, select, fit, and predict

Run commands from the release root after installation, or set `PYTHONPATH=src`.

```bash
python -m cnn_readout.workflow validate \
  --input prepared_2023.npz \
  --require-target \
  --require-oof-auxiliary
```

Epoch selection reproduces the paper's compact 4×3 out-of-fold contract:

- held months are June, July, August, and September 2023;
- held station folds are `0`, `1`, and `2`;
- a cell trains on `month != held month` **and**
  `station_fold != held fold`;
- it validates on the held month **and** held station fold;
- tuples within six hours of an internal month boundary are purged;
- contingency counts are pooled across all 12 cells at each epoch;
- the score is the unweighted mean CSI over 13 leads from 60 to 180 min and
  RN60 thresholds of 1, 5, 10, and 20 mm;
- the earliest epoch wins an exact tie.

```bash
python -m cnn_readout.workflow select-epoch \
  --input prepared_2023.npz \
  --output selection.json \
  --max-epochs 50 \
  --seed 7131000 \
  --device cuda
```

Pass `selected_epoch` from `selection.json` explicitly to each final refit.
For example, the paper used three independently refitted readouts; repeat the
command with the registered refit seeds.

```bash
python -m cnn_readout.workflow fit \
  --input prepared_2023.npz \
  --output cnn_readout_seed21040.pt \
  --epochs 4 \
  --seed 21040 \
  --device cuda

python -m cnn_readout.workflow predict \
  --input prepared_2024_2025.npz \
  --checkpoint cnn_readout_seed21040.pt \
  --output predictions_seed21040.npz \
  --device cuda
```

The final prediction command applies the common float16 round trip before
storing millimetre predictions as float32. For canonical dense input, output contains
`predictions_mm [N,S,L]`, `issue_times_ns`, `station_ids`, and
`lead_minutes`, plus `targets_mm` when it was present in the input. Flat input
produces the corresponding singular keys and a one-dimensional
`prediction_mm` array. Both outputs record the checkpoint hash, epoch, and
seed.

All stochastic sources used by this workflow are seeded, training order is
explicit, and deterministic PyTorch algorithms are requested. Exact
bit-for-bit equality across different GPU models, CUDA libraries, or PyTorch
versions is not guaranteed; the release environment should be recorded when
reproducing reported checkpoints.
