# Public paper aggregate bundle

These CSV files contain station-anonymous results for the final 4-km/10-min
paper. `manifest.json` binds each sheet and reference rendering to its frozen
source Supplementary Data workbook and reader-asset manifest.

The bundle contains aggregate and seed-resolved counts, skill summaries,
paired date-block intervals, valid-time readout values, All-station distance-
group summaries, native-grid verification and Tables 1 and S1. Table S2 and
the Fig. S2 case definitions contain station identifiers and are supplied only
in the authorized full Supplementary Data 1. This public subset does
not contain station identifiers, station coordinates, station assignments,
station-time observations, stationwise CSI or case-station time series. It is
an aggregate subset of the full Supplementary Data 1 distributed with the
article or an authorized data archive.

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
