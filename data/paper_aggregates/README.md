# Public paper aggregate bundle

These CSV files contain station-anonymous results for the final 4-km/10-min
paper. `manifest.json` binds each sheet and reference rendering to its frozen
source Supplementary Data workbook and reader-asset manifest. Frozen CSV
filenames retain their original numbering so existing analysis scripts
continue to work; `current_figure_mapping` in the manifest and the figure
guide map those keys to the final manuscript.

The bundle contains aggregate and seed-resolved counts, skill summaries,
paired date-block intervals, valid-time readout values, All-station distance-
group summaries, native-grid verification and Tables 1 and S1. Case definitions
(including Table S2), station identifiers and station-resolved verification
results are provided separately in Supplementary Data 1. This public subset does
not contain station identifiers, station coordinates, station assignments,
station-time observations, stationwise CSI or case-station time series. It is
an aggregate subset of the full Supplementary Data 1. Data access and reuse
are subject to the applicable provider terms.

`Table1_main` stores full-precision numerical values. Table renderings
display CSI to four decimal places and continuous
metrics to three; `Table1_seed_numeric` provides the individual members.
`Fig2d_readout_summary` contains one row per route and threshold, with CSI and
frequency-bias means, standard deviations and the member count. Its columns
are `route`, `threshold_mm`, `csi_mean`, `csi_sd`, `frequency_bias_mean`,
`frequency_bias_sd` and `n_members`. Individual readouts and paired contrasts
are provided in `Fig2d_readout_members` and `Fig2d_delta_CI`.

See `docs/PAPER_OUTPUT_REPRODUCIBILITY.md` for the output-by-output matrix and
render commands.
