# Release information

Software version: **1.2.1** (2026-09-29).

## Verification-helper corrections in 1.2.1

- Paired date-block CSI comparisons use the intersection of finite truth and
  both predictions, in addition to paired date weights.
- Common CSI helpers return NaN when the event-union denominator is zero.
- Bootstrap intervals exclude undefined paired differences cell by cell and
  report the number of valid resamples; all-undefined cells retain NaN bounds.
- Documentation distinguishes KST issuance times, valid times and their
  respective resampling dates.

These corrections affect reusable helpers in `src/common/`. The manuscript
comparison workflows use separate implementations with common finite support
and undefined-ratio handling. Aggregate CSV values and figure assets are
unchanged from 1.2.0.

## Workflows included since 1.2.0

- Observation-defined rainfall-episode peak amount and timing verification.
- exPreCast CNN 3 × 3 versus 5 × 5 patch-size sensitivity workflows.
- Aggregate results and rendering for manuscript Figure 6 and Supplementary Figures S2–S3.
- Final JoH figure numbering (Figs. 1–10 and S1–S3), typography and palettes.
- The author-edited final Figure 2 PNG and its raster-wrapped PDF.
- Patch-size CSI contrast data remain available; the contrast plot is no longer a default figure.
- Aggregate-table provenance and checksums are recorded in the manifests; the complete Supplementary Data 1 workbook is distributed separately.
- Reader-facing exPreCast column names for native-grid RN60 contrasts, with the numerical rows unchanged.

Amount errors use 1,437 station-episodes. Timing errors use the common subset
of 1,391 episodes with non-constant forecasts across all routes, members and leads.
The original forecast-route workflows and aggregate results remain available.

Software and Supplementary Data 1 are separate archive items. Software 1.2.1
uses the same Supplementary Data 1 workbook as software 1.2.0. Repository
citation metadata are in `CITATION.cff`; the DOI below identifies only the
earlier archived software version explicitly named alongside it.

Previous version **1.1.0** is archived at
[10.5281/zenodo.22684438](https://doi.org/10.5281/zenodo.22684438).
That DOI identifies the earlier version, not the additional workflows in 1.2.0.

See [README.md](README.md) for the available workflows,
[figure/table reproduction](docs/PAPER_OUTPUT_REPRODUCIBILITY.md) for output
coverage, and [data policy](docs/DATA_POLICY.md) for access and redistribution.
