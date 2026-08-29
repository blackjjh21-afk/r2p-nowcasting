# Evaluation

`radar_only_comparison.py` verifies that standard and radar-only Direct R2P
predictions share the same checkpoints, truth arrays, issue times, stations,
and leads. It computes three-member CSI at the five displayed leads and four
RN60 thresholds, then estimates paired pointwise 95% intervals by resampling
the 244 issuance dates.

First create both prediction stores from the same trained output root:

1. run the standard `evaluate` command;
2. run `evaluate` again with `--mask-all-context-gauges --evaluation-tag
   radar_only` and the same `--output-root`, seeds and checkpoint name.

The complete commands are in `src/r2p_4km10min/README.md`. The comparison then
runs as follows:

```bash
python src/evaluation/radar_only_comparison.py \
  --r2p-root outputs/direct_r2p \
  --output-dir outputs/radar_only \
  --figure-dir figures/radar_only
```

The standalone sensitivity rendering corresponds to **Fig. 4a** in the current
manuscript and is written as `fig4a_radar_only_same_checkpoint_4km10min`.
Full tensor hashes are computed by default; `--skip-source-hashes` is intended
only for development runs. The route contrasts reported as Fig. 3e,f are a
separate result and are not mixed into this ablation renderer.

## Training-time station-dropout ablation (Fig. 4b)

`station_dropout_comparison.py` compares the standard Direct R2P route with a
separately trained control in which station dropout was disabled. It requires
all three seeds for both routes and verifies identical issue-time, station,
lead and truth axes. It also verifies that the two frozen scientific contracts
differ only in their station-dropout configuration. A missing control seed,
including a missing seed 1, is a hard error rather than a partial comparison.

Pass both result roots and both evaluation-store names explicitly:

```bash
python src/evaluation/station_dropout_comparison.py \
  --direct-root outputs/direct_r2p \
  --no-dropout-root outputs/no_station_dropout \
  --direct-store target_masked_heldout128_predictions_raw_macro_csi_v1 \
  --no-dropout-store target_masked_heldout128_predictions_raw_macro_csi_v1 \
  --output-dir outputs/station_dropout_comparison \
  --figure-dir figures/station_dropout_comparison
```

The output is station-dropout Direct R2P minus the no-station-dropout control
at 60, 90, 120, 150 and 180 min and RN60 thresholds of 1, 5, 10 and 20 mm.
The same issuance-date bootstrap draws are applied to both routes. The
standalone Fig. 4b renderer uses grouped bars and intentionally adds no plot
title; the manuscript caption supplies the panel title.
