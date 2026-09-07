# Direct R2P: 4-km/10-min input contract

This module contains the Direct R2P train/evaluate runner used for the
4-km/10-min comparison. It does not use exPreCast internally.

## Frozen scientific contract

- radar input: seven KMA TIFF frames at `t-60, t-50, ..., t`;
- gauge context: twelve fitting-station RN60/RN15 histories at
  `t-55, t-50, ..., t`;
- output: station RN60 at `+5, +10, ..., +180` min (36 heads);
- displayed leads: `+60, +90, +120, +150, +180` min;
- fitting/checkpoint-selection/test years: 2019–2022 / 2023 / 2024–2025;
- fitting/test stations: 514 / 128;
- training batch: 16; validation and evaluation batch: 32;
- seeds: 0, 1, 2.

Checkpoint selection uses 2023 fitting-station validation. The validation
support retains every issue with at least one fitting-station RN60 target of
0.1 mm or more over +5 to +180 min, plus otherwise dry issues within 180 min
of one of those issues. Epochs 1–3 are excluded; the earliest epoch maximizing
the unweighted mean CSI over 13 leads (+60 to +180 min every 10 min) and four
thresholds (1, 5, 10 and 20 mm) is selected independently for each seed.

The held-out 128 target histories are absent from training, model selection,
and model inputs. They are opened only as evaluation truth. Standard inference
retains issuance-time gauge context from the 514 fitting stations. The
`--mask-all-context-gauges` switch evaluates the same checkpoint with those
remaining gauge histories additionally masked; this is the paper's
same-checkpoint radar-only sensitivity, not a newly trained model.

## Required data

The runner expects prepared 4-km TIFFs, the gauge CSV, the 514/128 split, the
station metadata table, and the station-to-TIFF-grid mapping. Real station
records are not included in this source release. See `contracts/README.md`.

The commands below use the repository-relative private layout documented in
`docs/DATA_POLICY.md` and pass every scientific input explicitly. Run them from
the release root after the installation described in the top-level README.

### Inspect and preflight the exact input contract

```bash
python -m r2p_4km10min.run_vanilla_r2p preflight \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p \
  --device cpu
```

### Train and evaluate Direct R2P

```bash
python -m r2p_4km10min.run_vanilla_r2p train \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p \
  --device cuda:0 \
  --seeds 0,1,2

python -m r2p_4km10min.run_vanilla_r2p evaluate \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p \
  --device cuda:0 \
  --seeds 0,1,2
```

`evaluate` performs inference and writes both held-out prediction stores and
metrics for the selected checkpoints.

R2P checkpoints include optimizer, scheduler and Python/NumPy random-number
generator states so interrupted training can resume exactly. This full-state
format is not compatible with PyTorch's restricted `weights_only` loader.
Accordingly, the runner accepts checkpoints only from beneath the configured
local output root and emits a warning before loading them. Treat `--resume`
and `evaluate` as trusted-local operations: never place downloaded or otherwise
untrusted `.pt` files in the output tree. Patch MLP checkpoints contain only
tensors and primitive metadata and are loaded with `weights_only=True`.

### Same-checkpoint radar-only inference

Run a second evaluation against the same Direct R2P output root and checkpoint
set, changing only the inference-time context mask:

```bash
python -m r2p_4km10min.run_vanilla_r2p evaluate \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p \
  --device cuda:0 \
  --seeds 0,1,2 \
  --mask-all-context-gauges \
  --evaluation-tag radar_only

python src/evaluation/radar_only_comparison.py \
  --r2p-root outputs/direct_r2p \
  --output-dir outputs/radar_only \
  --figure-dir figures/radar_only
```

The standalone radar-only rendering corresponds to **Fig. 4a** in the current
manuscript. The bundled aggregate and reference rendering use the current
`fig4a_radar_only_same_checkpoint_4km10min` filename.

### Matched no-station-dropout training command

The matched training control uses a separate output root and sets all three
station-masking probabilities to zero:

```bash
python -m r2p_4km10min.run_vanilla_r2p train \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p_no_station_dropout \
  --device cuda:0 \
  --seeds 0,1,2 \
  --station-dropout-min 0 \
  --station-dropout-max 0 \
  --station-dropout-full-probability 0

python -m r2p_4km10min.run_vanilla_r2p evaluate \
  --project-root . \
  --radar-root data/private/radar_4km10min_tiff \
  --gauge-csv data/private/gauge_5min/gauge_rn60_rn15.csv \
  --mapping-csv data/private/station_contract/mapping.csv \
  --stations-csv data/private/station_contract/stations.csv \
  --split-csv data/private/station_contract/split.csv \
  --output-root outputs/direct_r2p_no_station_dropout \
  --device cuda:0 \
  --seeds 0,1,2
```

This reproduces the computational control. The public comparison command is
documented in `src/evaluation/README.md`; its final aggregate, sanitized audit
manifest and reference rendering are bundled under `data/examples/` and
`figures/`.

The default frozen-hash checks are intentionally strict and require the exact
authorized station files. Use `--allow-unfrozen-contract` only for a new
station/data contract and report that departure separately. This runner covers
Direct R2P and its ablations. The field-first components are documented under
`src/pysteps_adapter/`, `src/exprecast_adapter/` and `src/patch_mlp/`.
