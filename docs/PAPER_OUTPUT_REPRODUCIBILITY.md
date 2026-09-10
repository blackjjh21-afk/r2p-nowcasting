# Paper-output reproducibility scope

The repository separates three reproducibility levels so that restricted KMA
observations are not confused with publicly redistributable aggregate results.

1. **Public aggregate reproduction** uses the CSV files under
   `data/paper_aggregates/` and requires no station identifiers, coordinates or
   station-time observations.
2. **Authorized-input reproduction** provides the display code but requires a
   user to supply the provider-derived station or radar inputs lawfully.
3. **Upstream model reproduction** requires KMA observations and, for
   exPreCast, separately obtained upstream source and weights.

## Reader-output matrix

| Reader output | Rendering code | Bundled numerical source | Scope |
|---|---|---|---|
| Figure 1 | `paper_outputs.render_restricted figure1` | No | Authorized radar footprint and frozen station split required; final coastline/border transform is implemented with caller-supplied Natural Earth data |
| Figure 2 | `r2p-paper-figures --item 2` | Yes | Public aggregate numerical/display reconstruction, including the valid-time center-cell MLP and local-patch CNN diagnostics |
| Figure 3 | `r2p-paper-figures --item 3` | Yes | Public aggregate numerical/display reconstruction |
| Figure 4 | `r2p-paper-figures --item 4`; component audits in `src/evaluation/` | Yes | Public aggregate numerical/display reconstruction; private stores are required only to recompute the aggregates |
| Figure 5 | `r2p-paper-figures --item 5` | Yes | Public aggregate numerical/display reconstruction |
| Figure 6 | `r2p-paper-figures --item 6` | Editable source included | Exact source-bound finalizer and final PNG/PDF/PPTX |
| Supplementary Figure S1 | `r2p-paper-figures --item s1` | Yes, distance-group aggregates only | Public aggregate numerical/display reconstruction; no station assignment table |
| Supplementary Figure S2 | `paper_outputs.render_restricted s2` | No | Four +60-min station time series, using three-member means without shading; case definitions and station values require authorized Supplementary Data 1 |
| Supplementary Figure S3 | `paper_outputs.render_restricted s3` | Pairwise station-count summary only | Exact stationwise CSI plus the authorized full station split are required; renderer includes the fitting-station and geographic context layers |
| Supplementary Figure S4 | `r2p-paper-figures --item s4` | Yes | Public aggregate numerical/display reconstruction |
| Table 1 | `r2p-paper-figures --item table1` | Yes | Public aggregate numerical/display reconstruction |
| Supplementary Table S1 | CSV sheet in `data/paper_aggregates/` | Yes | Public aggregate reproduction |
| Supplementary Table S2 | `TableS2_cases` in authorized Supplementary Data 1 | No | Exact four-case windows and held-out station identifiers; not bundled with the public aggregate subset |
| Supplementary Data 1 | Frozen workbook and CSV subset in `data/paper_aggregates/` | Aggregate-only workbook included | The complete reader workbook remains a separate data-archive item because it contains station-resolved sheets |

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
`--data-root` and `--examples-root` explicitly. Figure 6 additionally requires
the included editable source and finalizer in a source checkout.

The command reads the frozen compact tables and emits PNG/PDF files.  The
manuscript renderings copied under `figures/reference/` are hash-bound reference
images.  Figures 2--5, S1 and S4 are numerical/display reconstructions from
those aggregate values, not assertions of byte-identical page artwork.  Figure
6 is the exception: its finalizer is bound to the included editable source and
its PNG hash matches the final manuscript asset.  Matplotlib output hashes can
vary across FreeType and operating-system versions even when numerical values
are identical.

## Bundled aggregate data

`data/paper_aggregates/` contains the frozen public-safe CSV sheets and
`Supplementary_Data_1_public_aggregate.xlsx`. These exclude station identifiers,
coordinates and station-time values. The bundled reference figures are also
non-station-resolved; Fig. 1, Supplementary Fig. S2 and Supplementary Fig. S3
require the authorized inputs described below.

## Authorized station-resolved outputs

The following commands intentionally fail without user-supplied inputs:

```bash
python -m paper_outputs.render_restricted figure1 \
  --coverage-npz data/private/figure1_coverage.npz \
  --stations-csv data/private/station_split.csv \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted s2 \
  --supplementary-data data/private/Supplementary_Data_1.xlsx \
  --output-dir outputs/paper_figures

python -m paper_outputs.render_restricted s3 \
  --supplementary-data data/private/Supplementary_Data_1.xlsx \
  --stations-csv data/private/station_split.csv \
  --cartopy-data-dir data/private/cartopy \
  --output-dir outputs/paper_figures
```

The Cartopy directory must contain the Natural Earth 10m land, ocean,
coastline and national-boundary shapefiles.  The command validates these files
before rendering and never downloads them implicitly.

Supplementary Figure S2 reads `Case_definitions` and
`FigS2_station_timeseries` from the authorized workbook. It validates panels
a–d, their display stations, 24-hour windows and the common +60-minute
valid-time index before plotting. The four station examples are qualitative
illustrations; the plotted means are not uncertainty bands. Member-resolved
values remain available in the full Supplementary Data 1.

This split is deliberate. It exposes the final transformations while keeping
station-resolved KMA-derived inputs outside the public repository under the
approved scope recorded in `DATA_REDISTRIBUTION_DECISION.md`. It does not turn
the release into a raw-KMA-to-paper reproduction package: upstream exPreCast
training/export and construction of the private CNN tuple cache remain
separate authorized-input workflows.
