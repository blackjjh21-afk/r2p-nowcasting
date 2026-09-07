# 4-km/10-min release status

Status: **v1.0.0 was published on GitHub and archived on Zenodo on 2026-08-29**.

This directory is the audited source root of the public release. The old
2-km/5-min package is retained outside this release as legacy provenance.

## Release metadata

`CITATION.cff` records version 1.0.0, the 29 August 2026 release date, both
authors and ORCIDs, the public repository, and published software DOI
10.5281/zenodo.22147192. The audited tree and `v1.0.0` release were published
on GitHub, and the corresponding software archive was published on Zenodo, on
29 August 2026.

The approved BSD-3-Clause `LICENSE` and dated responsible-author
`DATA_REDISTRIBUTION_DECISION.md` are installed. Station-resolved material
remains outside GitHub, while only the named derived sheets in Supplementary
Data 1 are approved for journal/permanent-archive distribution.

The remaining upstream scope boundary is documented rather than treated as a
fourth author decision: building the complete Patch MLP prepared tuple cache
from restricted inputs and training/exporting adapted exPreCast fields remain
external authorized-input workflows. Public aggregate renderers now cover
Figs. 2--6, S1/S4 and Table 1; authorized-input display code covers Fig. 1 and
S2/S3. The package therefore claims a public audit/display layer, not a
self-contained raw-KMA-to-paper reproduction.

## Scientific contract status

- [x] 4-km/10-min radar history fixed to 7 frames (-60 to 0 min).
- [x] Dense-field output fixed to 18 frames (+10 to +180 min).
- [x] Direct R2P output fixed to 36 RN60 leads (+5 to +180 min).
- [x] Paper-facing leads fixed to +60/+90/+120/+150/+180 min.
- [x] RN60 thresholds fixed to 1/5/10/20 mm.
- [x] Evaluation support fixed to held-out 128 x 35,088 issue times.
- [x] Reader-facing primary routes fixed to pySTEPS + Patch MLP,
      exPreCast + Patch MLP, and Direct R2P.
- [x] Radar-only defined as same-checkpoint, all-context-gauges-masked
      evaluation.
- [x] Current radar-only aggregate source, renderer, contracts, and reference
      rendering regenerated after checkpoint reselection with the current
      Fig. 4a filename.
- [x] Final no-dropout checkpoint set and dependent manuscript artifacts frozen.
- [x] No-station-dropout result finalized and audited across all three seeds.

## Completed publication checks

The checks below were rerun after the final station-dropout aggregate, paper
output synchronization, and author-approved release metadata were installed.

- [x] Add an author-approved `LICENSE`.
- [x] Add an author-approved `CITATION.cff`.
- [x] Record both author-approved station-artifact scopes described in
      `docs/DATA_POLICY.md`: the GitHub repository decision and the derived
      station-resolved Supplementary Data 1 journal/archive decision, together
      with the official KMA source URL, access date and applicable terms.
- [x] Provide non-operative author-fillable templates for all three decisions
      and ensure that unfilled templates cannot satisfy the release-ready gate.
- [x] Update the release-ready data gate so it accepts either approved outcome
      rather than requiring positive redistribution approval.
- [x] Verify that official exPreCast source and weights are not redistributed.
- [x] Verify that no raw/prepared KMA arrays, secrets, absolute host paths,
      symlinks, caches or large model/prediction files are tracked.
- [x] Confirm that no notebooks are distributed; public workflows are CLI-only.
- [x] Rebuild and install the wheel after the final paper-output sync and run
      all nine CLI help smoke tests plus one wheel-installed figure render.
- [x] Run the 59 synthetic, renderer and contract tests successfully.
- [x] Recompute the bundled radar-only aggregate and Fig. 4a from the final
      reselected Direct R2P prediction stores.
- [x] Verify all display names against `configs/route_registry.json`.
- [x] Verify standard/radar-only checkpoint hashes and exact truth equality;
      retain the sanitized hashes in the public radar-only audit manifest.
- [x] Synchronize the public Direct R2P checkpoint-selection code with the
      final unweighted 52-cell macro-CSI criterion and regenerate its dependent
      radar-only outputs.
- [x] Finalize and audit the no-station-dropout control.
- [x] Refresh `configs/artifact_manifest.csv` for every current aggregate,
      reference figure and registered workflow artifact.
- [x] Review README claims against the manuscript. The manuscript's current
      promise of all figure-generation code must either be narrowed to the
      documented release scope or satisfied by extracting the remaining
      figure/table workflows before publication.
- [x] Add schema-bound public aggregate CSV/XLSX sources, reference renderings,
      public/authorized-input display code and the exact editable-source Fig. 6
      finalizer; document their distinct reproducibility levels.
- [x] Record the selected adapted exPreCast epoch-10 checkpoint SHA-256 without
      redistributing its weights.
- [x] Scrub Office author/title/timestamp metadata from public PPTX/XLSX files
      and extend the release audit to inspect OOXML internals.

## Latest verification

Verification date: **2026-08-29**.

- Synthetic, renderer and contract tests: **59 passed**.
- Wheel: `direct_r2p_4km10min-1.0.0-py3-none-any.whl` was built and installed
  without dependency resolution in an isolated `/tmp` environment.
- Console entry points: all **9** commands returned successful `--help`; the
  wheel-installed `r2p-paper-figures` command also rendered Figure 2 from the
  bundled aggregate CSV files.
- Staging audit: **0 errors, 0 warnings**.
- Release-ready audit: **0 errors, 0 warnings**.
- Source tree: no symlinks, notebooks, caches, build directories, egg-info,
  private arrays, model checkpoints or machine-specific paths detected.
- Public paper aggregation was repeated twice with identical XLSX and manifest
  hashes; all public Matplotlib/Finalizer renderings were repeated with
  identical PNG/PDF hashes.
- Final station-dropout audit: all three seeds present; all 20 point estimates
  positive and 18 pointwise intervals exclude zero.

## Final checkpoint-derived artifact freeze

- The verified 12-store return contains Direct R2P standard and radar-only
  inference for three seeds, the three-seed no-station-dropout control and the
  three-seed All-station comparator.
- Final selected epochs are 27/14/26 for Direct R2P and 28/11/20 for the
  no-station-dropout control.
- The bundled Fig. 4b aggregate, sanitized audit manifest, PNG and PDF were
  generated from the complete three-seed comparison.
- Supplementary Table S1a, Supplementary Data 1 and manuscript statements were
  regenerated from the same frozen checkpoint set.
- Tests, wheel/CLI smoke checks and both release audits were rerun after the
  final release metadata and data-policy decisions were installed.

## Explicitly out of scope

- All-station R2P is not a primary paper route.
- Patch CNN and GNN are not primary paper routes.
- The 2-km/5-min exPreCast experiment is not part of the main 4-km release.
- Public exPreCast (+60-min checkpoint) is not part of the +180-min matched
  three-route comparison.
- Private storage-mount, user-home and server-specific paths must never appear
  in the public release.
