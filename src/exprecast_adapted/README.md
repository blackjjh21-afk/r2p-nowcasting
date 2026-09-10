# Adapted exPreCast field forecasts

Training, checkpoint selection and field generation for the 4-km/10-min
exPreCast source. The long model maps seven observed HSR fields to eighteen
future fields (+10 to +180 min).

## Inputs

Obtain the original source separately from
[exPreCast](https://github.com/tony890048/exPreCast) at revision
`092922c126bdf6098fbda2e08b2f1f3b2873bd9a`. The model loader checks the SHA-256
of `model.py` before importing it. Attribution and sharing terms are in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

Supply prepared 256 × 256 KMA TIFFs with 10-min phase-0 timestamps and
`grid_coordinates.npz` containing `source_index_valid_mask`. Preparation is
described in [HSR preprocessing](../../docs/HSR_PREPROCESSING_4KM10MIN.md).
Only June–September windows are used. Training uses 2019–2022; selection uses
2023. Raw data and trained weights are not bundled.

Commands below assume an installed package and user-supplied paths. Run
`python -m exprecast_adapted.workflow inspect --radar-root /path/to/radar
--official-repo /path/to/exPreCast` to inspect the prepared input contract.

## Train the short and long stages

Both stages use 100,000 optimizer steps, micro-batch 4 and gradient
accumulation 4, BF16, AdamW at initial learning rate 0.001 and FACL. First
train the six-output short model:

```bash
python -m exprecast_adapted.workflow train \
  --radar-root /path/to/radar --official-repo /path/to/exPreCast \
  --run-name short --output-root outputs/exprecast \
  --output-frames 6 --profile memory_safe_add \
  --batch-size 4 --gradient-accumulation 4 --seed 6455
```

Initialize the eighteen-output model from that seed's short `best.pt`, loading
all shape-compatible tensors. Freeze the patch embedding and early encoder
stages; the final encoder stage remains trainable.

```bash
python -m exprecast_adapted.workflow train \
  --radar-root /path/to/radar --official-repo /path/to/exPreCast \
  --run-name long --output-root outputs/exprecast \
  --output-frames 18 --profile paper_long_concat \
  --initialization-checkpoint outputs/exprecast/short/seed_6455/best.pt \
  --freeze-encoder --trust-checkpoint \
  --batch-size 4 --gradient-accumulation 4 --seed 6455
```

`--trust-checkpoint` permits Python-state deserialization; use it only for
checkpoints you trust. Resuming training from an existing `last.pt` also
requires this flag. The upstream source is executable Python and must likewise
come from the trusted, pinned source.

## Select the final field checkpoint

The trainer's `best.pt` uses last-lead instantaneous rain-rate CSI for short
initialization. It is **not** the final long-stage RN60 selection. Evaluate all
23 completed long-stage epochs using:

```bash
python -m exprecast_adapted.selection run \
  --radar-root /path/to/radar --official-repo /path/to/exPreCast \
  --run-dir outputs/exprecast/long/seed_6455 \
  --device cuda:0 --trust-checkpoint
```

The selection score is the equally weighted mean native-grid RN60 CSI across
13 leads (+60 to +180 min every 10 min) and thresholds of 1, 5, 10 and 20 mm.
RN60 sums six consecutive rain-rate fields multiplied by 10/60 h. The
source-backed mask contains 59,136 cells; the 2023 evaluation has 1,441 windows
at a 12-frame issuance stride. Exact score ties select the earliest epoch.
The selector checks the expected window, mask and checkpoint counts and writes
per-epoch metrics plus the selected checkpoint path and hash. `preflight`
checks the inputs without loading weights; `self-test` checks the metric on
synthetic arrays.

## Export fields on the requested issuance axis

Use the checkpoint identified by the RN60 selector, not the trainer's long
`best.pt`. For the 2,887 fitting issuance times:

```bash
python -m exprecast_adapted.workflow export \
  --radar-root /path/to/radar --official-repo /path/to/exPreCast \
  --checkpoint /path/to/selected_epoch.pt --trust-checkpoint \
  --export-years 2023 --export-anchor-stride 1 \
  --export-anchor-times /path/to/fit_2023_issue_times.npy \
  --expected-export-anchors 2887 \
  --export-output outputs/exprecast/fit_2023.h5
```

For verification, use `--export-years 2024,2025` with the exact common 35,088
issuance times and `--expected-export-anchors 35088`. The timestamp file is a
one-dimensional NumPy datetime or Unix-nanosecond axis (NPY, or NPZ with
`anchor_times`). Keep stride 1 when supplying exact times, and check the
reported matches. The exports need substantial disk space: the default 10-GiB
limit requires explicit `--allow-large-export` for the full verification set.

The HDF5 stores normalized reflectivity, issue times, lead minutes, completion
flags and checkpoint provenance. Existing exports resume by default.
[The field-export adapter](../exprecast_adapter/README.md) validates this file
and constructs the station-patch cache for the CNN readout.
