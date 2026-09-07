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
| Figure 2 | `r2p-paper-figures --item 2` | Yes | Public aggregate numerical/display reconstruction |
| Figure 3 | `r2p-paper-figures --item 3` | Yes | Public aggregate numerical/display reconstruction |
| Figure 4 | `r2p-paper-figures --item 4`; component audits in `src/evaluation/` | Yes | Public aggregate numerical/display reconstruction; private stores are required only to recompute the aggregates |
| Figure 5 | `r2p-paper-figures --item 5` | Yes | Public aggregate numerical/display reconstruction |
| Figure 6 | `r2p-paper-figures --item 6` | Editable source included | Exact source-bound finalizer and final PNG/PDF/PPTX |
| Supplementary Figure S1 | `r2p-paper-figures --item s1` | Yes, distance-group aggregates only | Public aggregate numerical/display reconstruction; no station assignment table |
| Supplementary Figure S2 | `paper_outputs.render_restricted s2` | Case-level CSI aggregates only | Exact station time series require authorized Supplementary Data 1; renderer implements the final panel quantities and layout |
| Supplementary Figure S3 | `paper_outputs.render_restricted s3` | Pairwise station-count summary only | Exact stationwise CSI plus the authorized full station split are required; renderer includes the fitting-station and geographic context layers |
| Supplementary Figure S4 | `r2p-paper-figures --item s4` | Yes | Public aggregate numerical/display reconstruction |
| Table 1 | `r2p-paper-figures --item table1` | Yes | Public aggregate numerical/display reconstruction |
| Supplementary Tables S1/S2 | CSV sheets in `data/paper_aggregates/` | Yes | Public aggregate reproduction |
| Supplementary Data 1 | `scripts/sync_public_paper_outputs.py` | Aggregate-only workbook included | The complete reader workbook remains a manuscript/data-archive item because it contains station-resolved sheets |

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

## Rebuild the public aggregate layer (authors)

The author-side synchronization command takes explicit paths and has no
machine-specific defaults:

```bash
python scripts/sync_public_paper_outputs.py \
  --supplementary-data manuscript_assets/14_Supplementary_Data_1.xlsx \
  --reader-assets manuscript_assets
```

The command exports only an explicit sheet allow-list, rejects station-like
columns in those sheets, creates
`Supplementary_Data_1_public_aggregate.xlsx`, and copies only the non-station-
resolved reference figures.  It never copies Fig. 1, Supplementary Fig. S2 or
Supplementary Fig. S3.

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

This split is deliberate. It exposes the final transformations while keeping
station-resolved KMA-derived inputs outside the public repository under the
approved scope recorded in `DATA_REDISTRIBUTION_DECISION.md`. It does not turn
the release into a raw-KMA-to-paper reproduction package: upstream exPreCast
training/export and construction of the private Patch MLP tuple cache remain
separate authorized-input workflows.
