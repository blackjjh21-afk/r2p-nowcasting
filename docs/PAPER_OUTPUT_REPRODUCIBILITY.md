# Paper-output reproducibility scope

This guide uses the current Journal of Hydrology figure numbering. Historical
CSV names are deliberately stable; they do not determine the current figure
number. The correspondence is listed below.

The repository supports three reproducibility levels:

1. **Public aggregate reproduction** reads frozen CSVs in
   `data/paper_aggregates/`, `data/examples/` and `data/additional_verification/`.
   It needs no station identifiers, coordinates or station-time observations.
   Figure 2 is an artwork exception: the command copies the author's final
   PNG/PDF without redrawing it.
2. **Authorized-input reproduction** supplies display and analysis code, but
   users must lawfully obtain the required station or radar inputs themselves.
3. **Model training and prediction** require KMA observations. The
   [adapted exPreCast workflow](../src/exprecast_adapted/README.md) also requires
   separately obtained upstream source; its short and long stages train the
   field checkpoints locally.

## Figure and table reproduction

| Manuscript output | Command or source | Public reproduction scope |
|---|---|---|
| Fig. 1: study domain and station split | `python -m paper_outputs.render_restricted figure1` | Authorized footprint and station split, caller-supplied elevation and Natural Earth files required |
| Fig. 2: forecast-route architecture | `r2p-paper-figures --item 2` | Copies the final author-edited PNG/PDF from `figures/reference/`; not a code-redrawn diagram |
| Fig. 3: valid-time readout diagnostic | `r2p-paper-figures --item 3` | Public aggregate reconstruction, including center-cell MLP and local-patch CNN results |
| Fig. 4: forecast CSI and paired route contrasts | `r2p-paper-figures --item 4` | Public aggregate reconstruction |
| Fig. 5: heavy-rain amounts and exceedance frequencies | `r2p-paper-figures --item 5` | Public aggregate reconstruction |
| Fig. 6: rainfall-episode peak errors | `r2p-paper-figures --item 6` | Public aggregate reconstruction from `data/additional_verification/` |
| Fig. 7: four station time series | `python -m paper_outputs.render_restricted figure7` | Authorized case definitions and station-time CSVs required |
| Fig. 8: stationwise CSI maps | `python -m paper_outputs.render_restricted figure8` | Authorized stationwise CSI, complete station split and Natural Earth files required; public pairwise counts alone cannot recreate the maps |
| Fig. 9: All-station distance diagnostic | `r2p-paper-figures --item 9` | Public distance-group aggregates; no individual station assignments |
| Fig. 10: gauge masking and station dropout | `r2p-paper-figures --item 10` | Public aggregate reconstruction from audited example CSVs |
| Supplementary Fig. S1: native-grid field verification | `r2p-paper-figures --item s1` | Public aggregate reconstruction |
| Supplementary Fig. S2: readout patch-size skill | `r2p-paper-figures --item s2` | Public aggregate reconstruction; optional CSI-contrast audit plot is not part of this figure |
| Supplementary Fig. S3: episode-duration histogram | `r2p-paper-figures --item s3` | Public aggregate bin counts; no episode identifiers needed |
| Table 1 | `r2p-paper-figures --item table1` | Public full-precision source with display formatting |
| Supplementary Table S1 | `data/paper_aggregates/TableS1_training.csv` | Public training/selection summary |
| Supplementary Table S2 | `TableS2_cases` in authorized Supplementary Data 1 | Exact case windows and held-out station identifiers are not bundled |
| Supplementary Data 1 | Public CSV subsets in the aggregate directories | Complete reader workbook remains a separate data-archive item because it includes station-resolved sheets |

Figure 3's final center-cell MLP and local-patch CNN aggregate results are
included. Their valid-time model-training workflow is not distributed;
`src/cnn_readout/` implements the separate forecast-route CNN workflow.

## Stable CSV names and current figure numbers

Names below refer to `data/paper_aggregates/` unless another directory is
stated. Their historical numbering is retained to avoid changing scientific
inputs when manuscript figures are renumbered.

| Stable source name | Historical figure | Current use |
|---|---|---|
| `Fig2a_density.csv`, `Fig2a_summary.csv`, `Fig2b_amount_bins.csv`, `Fig2c_fixed_metrics.csv`, `Fig2d_readout_summary.csv` | Fig. 2 | Fig. 3 valid-time diagnostic; related `Fig2d_*` member/contrast tables also remain available |
| `Categorical_mean_sd.csv`, `Fig3ef_route_contrasts.csv`, `Fig3_macro_contrasts.csv` | Fig. 3 | Fig. 4 forecast skill and contrasts |
| `Fig4a_radar_only.csv`, `Fig4b_no_dropout.csv` | Fig. 4 | Fig. 10 ablation summaries; renderer uses corresponding audited `radar_only_csi_by_lead_threshold.csv` and `station_dropout_csi_by_lead_threshold.csv` in `data/examples/` |
| `Fig5_plot_source.csv`, `Fig5_amount.csv`, `Fig5_frequency.csv` | Fig. 5 | Fig. 5 heavy-rain diagnostics |
| `FigS1_distance_cells.csv`, `FigS1_distance_summary.csv` | Supplementary Fig. S1 | Fig. 9 distance diagnostic |
| `FigS3_pairwise_counts.csv` | Supplementary Fig. S3 | Aggregate station-count summary associated with Fig. 8; not the stationwise map input |
| `Field_instant_CSI.csv`, `Field_instant_CSIM.csv`, `Field_RN60_CSI.csv`, `Field_RN60_CSIM.csv`, `Field_RN60_CSI_delta_CI.csv` | Supplementary Fig. S4 | Supplementary Fig. S1 native-grid verification |
| `episode_route_metrics.csv` in `data/additional_verification/` | Additional episode analysis | Fig. 6 |
| `patch_metrics_mean_of_members.csv`, `patch_paired_CSI_intervals.csv` in `data/additional_verification/` | Earlier patch-size supplement | Supplementary Fig. S2 skill and optional contrast audit |
| `episode_duration_histogram.csv` in `data/additional_verification/` | Earlier duration supplement | Supplementary Fig. S3 |

For authorized station-resolved inputs, `Fig7_station_timeseries.csv` and
`Fig8_station_CSI.csv` are current names. Legacy aliases
`FigS2_station_timeseries.csv` and `FigS3_station_CSI.csv` are accepted when
the current-named file is absent. New names take precedence if both exist.

## Re-render public outputs

After installing the package, run from the repository root:

```bash
r2p-paper-figures --item all --output-dir outputs/paper_figures
```

`all` produces Figs. 2, 3, 4, 5, 6, 9 and 10, Supplementary Figs. S1–S3,
and Table 1. It does not attempt the authorized-input Figs. 1, 7 or 8.
Select one output with `--item` and an individual number, `s1`, `s2`, `s3`
or `table1`.

When invoking the installed command outside the checkout, provide explicit
source directories:

```bash
r2p-paper-figures --item all \
  --data-root /path/to/checkout/data/paper_aggregates \
  --examples-root /path/to/checkout/data/examples \
  --additional-data-root /path/to/checkout/data/additional_verification \
  --reference-root /path/to/checkout/figures/reference \
  --output-dir outputs/paper_figures
```

These directories are checkout/archive artifacts, not assumed to be installed
wheel package data. The command emits PNG/PDF files; Table 1 also has CSV and
Markdown display exports. Public reconstructions retain current figure labels,
lead colors and typography but are not assertions of byte-identical page
artwork. Matplotlib hashes can vary across FreeType, Matplotlib and operating
systems.

Figure 2 is copied byte for byte from the supplied `Figure_2.png` and
`Figure_2.pdf` files. The PDF contains the same raster image; the command
does not regenerate the diagram from plotting code.

`Table1_main.csv` retains full precision. Display formatting rounds CSI to
four decimal places and continuous metrics to three, with the last column
labelled **Pearson r**. `Table1_seed_numeric.csv` retains member values.
`Fig2d_readout_summary.csv` combines CSI and frequency-bias means, standard
deviations and member counts on common route–threshold rows. Member values
and paired contrasts remain in `Fig2d_readout_members.csv` and
`Fig2d_delta_CI.csv`.

## Authorized station-resolved figures

Supply required inputs explicitly:

```bash
python -m paper_outputs.render_restricted figure1 \
  --coverage-npz data/private/figure1_coverage.npz \
  --stations-csv data/private/station_split.csv \
  --terrain-file data/private/etopo_2022_subset.nc \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted figure7 \
  --source-data data/private/station_figure_sources \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted figure8 \
  --source-data data/private/station_figure_sources \
  --stations-csv data/private/station_split.csv \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures
```

For Fig. 1, the coverage NPZ contains matching two-dimensional `lon`, `lat`
and `footprint` arrays and may retain `extent`. The station split contains
`station_id`, `lat`, `lon` and `split`, with 514 `train` and 128 `test`
stations. Terrain is required: `--terrain-file`, `--terrain-npz` and
`--terrain-nc` are aliases for one option. Terrain NPZ uses one-dimensional
`lon`/`lat` and two-dimensional `elevation_m`; ETOPO-compatible NetCDF uses
`longitude`, `latitude` and `z`, with elevation in metres on a
latitude × longitude grid. Terrain must cover the displayed domain. The
current display uses brown elevation shading, 100-m colorbar ticks from
100 to 1000 m and a northern map boundary of 40.4°N.

The Cartopy directory must contain the Natural Earth 10m land, ocean,
coastline and national-boundary shapefiles and companion files. Expected
cache locations are checked; these inputs are never downloaded implicitly.

For Fig. 7, supply `Case_definitions.csv` and
`Fig7_station_timeseries.csv` (or its legacy alias) under `--source-data`.
The renderer validates panels a–d, display stations, 24-hour windows and
common +60-min forecasts on a 10-min valid-time grid. Gauge observations
and three-member forecast means share a single legend, without uncertainty
shading. The cases are qualitative illustrations; member-resolved values
remain in the full Supplementary Data 1.

For Fig. 8, supply `Fig8_station_CSI.csv` (or its legacy alias), the complete
station split and the Natural Earth cache. The CSI file contains
`station_id`, `lat`, `lon`, `route`, `threshold_mm` and `csi_mean` for every
held-out station, route and threshold. Station identities and coordinates
are validated. The current compact four-row layout uses bold panel and RN60
headings, regular mean CSI below each map and **CSI** colorbar labels.

The old restricted-command aliases `s2` and `s3` remain accepted for
Fig. 7 and Fig. 8 respectively; these aliases do not denote the current
Supplementary Figs. S2 and S3. Station-resolved KMA-derived inputs are excluded
under the [data policy](DATA_POLICY.md) and are not supplied by the public
plotting commands.

## Episode verification and patch-size sensitivity

The additional figures can also be rendered independently:

```bash
python -m paper_outputs.render_additional_verification \
  --data-root data/additional_verification \
  --output-dir outputs/additional_verification
```

This checks aggregate CSV hashes and schemas and produces three PNG/PDF
pairs: `Figure_6`, `Figure_S2` and `Figure_S3`. The standalone `--item`
choices are `all`, `episode`, `patch` and `duration`.

Fig. 6's amount panels use all 1,437 station-episodes; maximum timing uses
the common 1,391-episode subset with non-constant forecasts. Errors are
averaged over episodes within each member and then over members. The
renderer reads stored scores and does not recompute errors from displayed
curves. Supplementary Fig. S3 uses ten-minute bins and hourly ticks at bin
boundaries, with no internal whole-figure title.

Supplementary Fig. S2 shows three-route CSI curves. Paired patch-size CSI
differences and their 95% confidence intervals are provided in
`data/additional_verification/patch_paired_CSI_intervals.csv`.
Add `--include-contrasts` to render these comparisons as an additional plot:

```bash
python -m paper_outputs.render_additional_verification \
  --item patch --include-contrasts \
  --data-root data/additional_verification \
  --output-dir outputs/patch_audit
```

This additionally emits `patch_spatial_support_contrasts.png/.pdf`;
its bands show stored paired date-bootstrap 95% intervals. To plot newly
computed results, use `--item patch --patch-results /path/to/comparison_directory`,
optionally with `--include-contrasts`. That directory must provide
`metrics_mean_of_members.csv` and `paired_CSI_intervals.csv` in the
documented comparison schema.

Recomputation with authorized observations and predictions is documented in
[episode verification](../src/evaluation/RAINFALL_EPISODES.md) and
[CNN patch sensitivity](../src/cnn_readout/PATCH_SENSITIVITY.md).

## Workflow limitations

This is not an unrestricted raw-KMA-to-paper reproduction package. Raw
observations, station-resolved plotting inputs and prediction stores must
be supplied under the applicable permissions. Construction of the CNN tuple
cache remains an external authorized-input step. Valid-time readout training
is not distributed. Aggregate plotting does not rerun training, inference,
episode selection or bootstrap resampling. Figure 2 reproduces the final
author-supplied artwork, not its editable design history.
