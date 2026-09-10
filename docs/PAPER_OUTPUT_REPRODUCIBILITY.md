# Paper-output reproducibility scope

The repository supports three reproducibility levels:

1. **Public aggregate reproduction** uses the CSV files under
   `data/paper_aggregates/` and requires no station identifiers, coordinates or
   station-time observations.
2. **Authorized-input reproduction** provides the display code but requires a
   user to supply the provider-derived station or radar inputs lawfully.
3. **Model training and prediction** require KMA observations. The
   [adapted exPreCast workflow](../src/exprecast_adapted/README.md) also requires
   separately obtained upstream source; its short and long stages train the
   field checkpoints locally.

## Reader-output matrix

| Reader output | Rendering code | Bundled numerical source | Scope |
|---|---|---|---|
| Figure 1 | `paper_outputs.render_restricted figure1` | No | Authorized radar footprint and frozen station split required; final coastline/border transform is implemented with caller-supplied Natural Earth data |
| Figure 2 | `r2p-paper-figures --item 2` | Yes | Public aggregate numerical/display reconstruction, including the valid-time center-cell MLP and local-patch CNN diagnostics |
| Figure 3 | `r2p-paper-figures --item 3` | Yes | Public aggregate numerical/display reconstruction |
| Figure 4 | `r2p-paper-figures --item 4`; component audits in `src/evaluation/` | Yes | Public aggregate numerical/display reconstruction; private stores are required only to recompute the aggregates |
| Figure 5 | `r2p-paper-figures --item 5` | Yes | Public aggregate numerical/display reconstruction |
| Figure 6 | `r2p-paper-figures --item 6` | Renderer source included | Code-defined Matplotlib diagram |
| Supplementary Figure S1 | `r2p-paper-figures --item s1` | Yes, distance-group aggregates only | Public aggregate numerical/display reconstruction; no station assignment table |
| Supplementary Figure S2 | `paper_outputs.render_restricted s2` | No | Four +60-min station time series, using three-member means without shading; authorized case-definition and station-time CSVs are required |
| Supplementary Figure S3 | `paper_outputs.render_restricted s3` | Pairwise station-count summary only | Exact stationwise CSI CSV plus the authorized full station-split CSV are required; renderer includes the fitting-station and geographic context layers |
| Supplementary Figure S4 | `r2p-paper-figures --item s4` | Yes | Public aggregate numerical/display reconstruction |
| Table 1 | `r2p-paper-figures --item table1` | Yes | Public aggregate numerical/display reconstruction |
| Supplementary Table S1 | CSV sheet in `data/paper_aggregates/` | Yes | Public aggregate reproduction |
| Supplementary Table S2 | `TableS2_cases` in authorized Supplementary Data 1 | No | Exact four-case windows and held-out station identifiers; not bundled with the public aggregate subset |
| Supplementary Data 1 | CSV subset in `data/paper_aggregates/` | Aggregate-only CSV tables included | The complete reader workbook remains a separate data-archive item because it contains station-resolved sheets |

Figure 2 includes the final center-cell MLP and local-patch CNN aggregate
results and reference rendering. Their valid-time model-training workflow is
not distributed; `src/cnn_readout/` is the separate forecast-route CNN workflow.

## Re-render public aggregate outputs

After installing the package, run from the repository root so the default
aggregate and example paths resolve:

```bash
r2p-paper-figures --item all --output-dir outputs/paper_figures
```

When invoking a wheel-installed command outside the checkout, pass
`--data-root` and `--examples-root` explicitly.

The command reads the frozen compact tables and emits PNG/PDF files.  The
manuscript renderings copied under `figures/reference/` are hash-bound reference
images. Figures 2--5, S1 and S4 are numerical/display reconstructions from
those aggregate values, not assertions of byte-identical page artwork.
Figure 6 is rendered directly from its code-defined diagram. Matplotlib output hashes can
vary across FreeType and operating-system versions even when numerical values
are identical.

## Bundled aggregate data

The frozen CSV tables in `data/paper_aggregates/` and bundled reference figures
exclude station identifiers, coordinates and station-time values.

`Table1_main` retains full-precision numerical values; the table renderer
displays CSI to four decimal places and continuous
metrics to three. `Table1_seed_numeric` retains the individual members.
`Fig2d_readout_summary` combines CSI and frequency-bias means and standard
deviations on the same route–threshold rows, with the member count; individual
readouts and paired contrasts remain in `Fig2d_readout_members` and
`Fig2d_delta_CI`.

## Authorized station-resolved outputs

Supply authorized inputs to render Figure 1 and Supplementary Figures S2–S3:

```bash
python -m paper_outputs.render_restricted figure1 \
  --coverage-npz data/private/figure1_coverage.npz \
  --stations-csv data/private/station_split.csv \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted s2 \
  --source-data data/private/station_figure_sources \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted s3 \
  --source-data data/private/station_figure_sources \
  --stations-csv data/private/station_split.csv \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures
```

The Cartopy directory must contain the Natural Earth 10m land, ocean,
coastline and national-boundary shapefiles.  The command validates these files
before rendering and never downloads them implicitly.

The `--source-data` directory must contain the externally supplied,
authorized CSVs named `Case_definitions.csv` and `FigS2_station_timeseries.csv`
for S2, and `FigS3_station_CSI.csv` for S3. The station-split CSV must contain
`station_id`, `lat`, `lon` and `split` (`train` or `test`). These inputs are not
bundled with the public repository.

Supplementary Figure S2 validates panels
a–d, their display stations, 24-hour windows and the common +60-minute
valid-time index before plotting. The four station examples are qualitative
illustrations; the plotted means are not uncertainty bands. Member-resolved
values remain available in the full Supplementary Data 1.

Station-resolved KMA-derived inputs are excluded under the
[data policy](DATA_POLICY.md). The release is not a raw-KMA-to-paper
reproduction package: constructing the CNN tuple cache remains an external
authorized-input step, and valid-time readout training is not included.
