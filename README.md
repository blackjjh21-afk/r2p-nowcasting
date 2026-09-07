# Where historical gauge supervision enters radar-based precipitation nowcasting for gauge-referenced point accumulation

Published public code package for the 4-km/10-min experiments.

This directory is the source tree for the published 4-km/10-min public
release. It is maintained independently from the legacy analysis package and
the private research workspace. The scientific contract is
frozen in [`configs/scientific_contract_4km10min.json`](configs/scientific_contract_4km10min.json),
and the reader-facing route names are frozen in
[`configs/route_registry.json`](configs/route_registry.json).

> **Release status:** version 1.0.0 was published on GitHub and archived on
> Zenodo on 2026-08-29. The scientific artifacts, BSD-3-Clause repository
> license, responsible-author KMA redistribution decision and final citation
> metadata are complete. The software release DOI is
> [doi:10.5281/zenodo.22147192](https://doi.org/10.5281/zenodo.22147192).

## Scientific question

The paper first holds observed HSR fixed at the target valid time and asks how
much of the fixed field-to-point readout deficit can be recovered by
gauge-supervised Center and Patch MLP readouts. It then compares a direct
radar-to-point route with two complete field-first routes on exactly the same
held-out stations, issue times, gauge truth and reported leads:

- **pySTEPS + Patch MLP**
- **exPreCast + Patch MLP**
- **Direct R2P**

The forecast-route comparison asks whether Direct R2P can forecast
gauge-referenced 60-minute accumulated precipitation (RN60) competitively
without first producing a dense precipitation field. A matched **Direct R2P
(radar-only)** ablation evaluates the same trained checkpoints after masking
all issuance-time gauge context. It is an input ablation, not a separately
trained model.

## Frozen comparison contract

| Item | Contract |
|---|---|
| Radar input | Prepared KMA 4-km TIFF fields at 10-min spacing |
| Radar history | 7 frames: -60, -50, -40, -30, -20, -10 and 0 min |
| Field forecasts | 18 fields: +10 to +180 min at 10-min spacing |
| Direct R2P output | 36 gauge-RN60 targets: +5 to +180 min at 5-min spacing |
| Paper-facing leads | +60, +90, +120, +150 and +180 min |
| Thresholds | RN60 >= 1, 5, 10 and 20 mm |
| Station split | 514 fitting stations; 128 stations excluded from fitting |
| Evaluation support | 35,088 common issue times in June-September 2024-2025 |
| Primary score | CSI pooled over the frozen issue-time/station support |
| Uncertainty | Paired bootstrap over issuance-date blocks |

The 18 field-forecast frames and 36 Direct R2P targets are different output
contracts. Comparisons are therefore made only after conversion to the same
gauge-RN60 verification axes at the five paper-facing leads.

## Route definitions

The two field-first routes pair different upstream field sources with
separately fitted instances of the same Patch MLP architecture under a matched
fitting and evaluation protocol. Each readout uses a local 3 x 3 forecast patch
sequence and permitted issuance-time gauge context to predict station RN60.
`Direct R2P` receives the common radar history and its permitted context
directly. The 128 evaluation stations are excluded from model fitting and
checkpoint selection.

Internal experiment names such as `pySTEPS-LK`, `selected long exPreCast`,
`Context Patch MLP`, and `Direct R2P (stride 2)` are provenance details, not
reader-facing route names. Public figures and tables use the simplified names
listed above.

## Current reproducibility scope

This release is not a self-contained raw-data reproduction package. Its
current capabilities are:

| Task | Current status |
|---|---|
| Inspect the frozen scientific and route contracts | Available |
| Inspect and hash-audit the bundled Direct R2P radar-only and station-dropout aggregates and reference renderings | Available; recomputation and re-rendering require private prediction stores |
| Convert user-obtained provider-format KMA HSR to the frozen 4-km/10-min radar contract | Available |
| Train and evaluate Direct R2P | Available only with the authorized gauge and station-contract inputs described below |
| Validate prepared Patch MLP tuples, run out-of-fold epoch selection, refit and predict | Available from prepared inputs; construction from restricted inputs is an external authorized-input workflow and is not distributed |
| Generate pySTEPS field forecasts and station patch windows | Available through `src/pysteps_adapter/` |
| Audit project-adapted exPreCast field exports and extract station patches | Available through `src/exprecast_adapter/`; upstream exPreCast training/export remains external |
| Re-render Figs. 2--6, S1/S4 and Table 1 from public aggregate sources | Available; Figure 6 uses the exact editable-source finalizer, while the others are aggregate numerical/display reconstructions with hash-bound manuscript reference images |
| Re-render station-resolved Fig. 1 and S2/S3 | Display code available; authorized radar/station inputs and a caller-supplied Natural Earth cache are required and are not redistributed here |
| Inspect Supplementary Tables S1/S2 and the public portion of Supplementary Data 1 | Available as schema-bound CSV files and a deterministic aggregate-only XLSX |

The release provides a compact public audit and display layer. Users who
lawfully obtain the required KMA and third-party inputs can run the portable
components explicitly listed above; raw-to-paper reproduction also depends on
the external upstream exPreCast export and private-input tuple construction
identified in the status table. Raw KMA observations, prepared radar archives, gauge time series,
pretrained exPreCast artifacts, project checkpoints, and full prediction
arrays are not part of this release. See
[`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) and
[`RELEASE_STATUS.md`](RELEASE_STATUS.md).

## Release layout

```text
release_4km10min/
├── configs/                 # frozen scientific and route contracts
├── docs/                    # data and release policy
├── data/paper_aggregates/   # compact aggregate paper sources
├── figures/                 # reference assets and editable Figure 6
├── src/                     # models, evaluation and paper-output renderers
├── scripts/                 # synchronization, Figure 6 and release audits
├── tests/                   # synthetic and contract tests
├── README.md
├── RELEASE_STATUS.md
├── THIRD_PARTY_NOTICES.md
└── environment.yml
```

[`RELEASE_STATUS.md`](RELEASE_STATUS.md) records the audited status and scope
of the published release.

## Implemented public components

- `src/r2p_4km10min/`: portable Direct R2P model, data contract, training,
  standard evaluation, and same-checkpoint radar-only evaluation. Real station
  contract files remain user-supplied.
- `src/patch_mlp/`: standalone six-step 3×3 Patch MLP, deterministic
  target-stratified sampler, exact importance-corrected unweighted
  log1p(RN60) Smooth-L1 objective, and prepared-data validation, OOF selection,
  refit and prediction workflow. The private-input-to-prepared-cache builder
  remains outside the public release scope.
- `src/pysteps_adapter/`: explicit-file deterministic pySTEPS field generation
  and station patch-window extraction.
- `src/exprecast_adapter/`: validation of user-generated project-adapted
  exPreCast HDF5 exports and exact station-patch extraction. Upstream source,
  weights and field generation remain governed by the original project. The
  selected adapted epoch-10 field checkpoint is not redistributed; its
  SHA-256 provenance is frozen in
  [`configs/upstream_artifact_provenance.json`](configs/upstream_artifact_provenance.json).
- `src/evaluation/radar_only_comparison.py`: checkpoint/truth/axis audit,
  lead-by-threshold CSI, 50,000-replicate paired issuance-date bootstrap, and
  radar-only figure rendering.
- `src/evaluation/station_dropout_comparison.py`: complete three-seed audit of
  the separately trained no-station-dropout control, paired date-block
  intervals, and the grouped-bar Fig. 4b renderer. It fails rather than
  producing a partial comparison when any seed is missing.
- `src/common/`: compact categorical metrics, contracts, and paired block
  bootstrap helpers.
- `src/paper_outputs/`: public aggregate numerical/display renderers for Figs.
  2--6, S1/S4 and Table 1, plus authorized-input renderers for Fig. 1 and
  S2/S3. See [`docs/PAPER_OUTPUT_REPRODUCIBILITY.md`](docs/PAPER_OUTPUT_REPRODUCIBILITY.md).
- `scripts/sync_public_paper_outputs.py`: author-side, exact-schema export of
  the public-safe aggregate sheets and non-station-resolved reference assets.
  Unknown columns are rejected rather than silently copied.
- `scripts/finalize_fig6_architecture.py`: exact renderer for the included
  editable Figure 6 source; Office author metadata and volatile timestamps are
  scrubbed when release assets are generated.
- `src/preprocessing/prepare_kma_hsr_4km10min.py`: the exact provider-format
  500-m HSR to normalized 4-km exPreCast transform, including source decoding,
  coordinate audit, missing-frame accounting, and reproducibility manifests.
  See [`docs/HSR_PREPROCESSING_4KM10MIN.md`](docs/HSR_PREPROCESSING_4KM10MIN.md).

The aggregate paper tables, radar-only and station-dropout results, sanitized provenance
manifests and reference renderings under `data/` and `figures/`
contain no station identifiers, coordinates, or station-time observations.
They were regenerated from the final three-seed checkpoint selections and use
the current **Fig. 4a** and **Fig. 4b** filenames.

## Environment

```bash
conda env create -f environment.yml
conda activate r2p-4km10min-release
python -m pip install --no-deps -e .
python -m r2p_4km10min.run_vanilla_r2p --help
python scripts/audit_release.py
pytest -q -p no:cacheprovider tests
```

An installation is required because the package uses a `src/` layout; the
command above uses an editable install for convenient work from a source
checkout, while a regular wheel installation is also supported. `--no-deps`
leaves the versions installed from `environment.yml` unchanged. Direct package
and environment requirements are pinned consistently. The environment is
CLI-only and intentionally contains no Jupyter runtime.

GPU-specific PyTorch installation may need to be adjusted for the target CUDA
driver while retaining the recorded major software versions.

## Release audit and tests

Run the portable audit in staging mode during local development:

```bash
python scripts/audit_release.py
```

Staging must finish with no errors, and generated caches must be removed before
a clean audit. To verify a release, run the stricter gate:

```bash
python scripts/audit_release.py --release-ready
pytest -q -p no:cacheprovider tests
```

The release-ready command verifies that `CITATION.cff` records the final
version, release date, repository URL and archived software DOI. The approved
repository and Supplementary Data 1 scopes are recorded in
`DATA_REDISTRIBUTION_DECISION.md`. The audit also rejects machine-specific paths,
credential-like content, caches, symlinks, model/data archives, oversized
files, dirty notebook outputs, and drift in either frozen JSON contract. No
notebooks are distributed in this release; the supported workflows are
command-line interfaces.

## Evaluation conventions

- All timestamps are issue times; date blocks are defined from those issue
  times in the frozen project timezone.
- The standard and radar-only Direct R2P results must have identical truth,
  station, issue-time and lead axes and must reference the same checkpoint hash
  within each seed.
- CSI is `hit / (hit + miss + false alarm)` and is undefined when the
  denominator is zero.
- Confidence intervals are paired: competing routes are differenced within the
  same resampled issuance-date blocks.
- Test-period values must not be used for checkpoint or hyperparameter
  selection.

## Citation and license

Author-generated software is distributed under the BSD-3-Clause license in
[`LICENSE`](LICENSE). This license does not relicense KMA observations,
KMA-derived station data, exPreCast materials or other third-party assets.
The approved public-repository and Supplementary Data 1 scopes are recorded in
[`DATA_REDISTRIBUTION_DECISION.md`](DATA_REDISTRIBUTION_DECISION.md). Final
citation metadata are provided in [`CITATION.cff`](CITATION.cff), with the
software release DOI
[doi:10.5281/zenodo.22147192](https://doi.org/10.5281/zenodo.22147192).
