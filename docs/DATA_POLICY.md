# Data and artifact policy

This policy separates code reproducibility from permission to redistribute KMA
observations and station-resolved derivatives.

## Never included in the public source repository

- KMA API credentials, cookies, account files or private download URLs.
- Raw KMA radar or gauge products.
- Prepared 4-km/10-min TIFF radar archives.
- Station-time gauge observations or target arrays.
- Dense exPreCast or pySTEPS forecast archives.
- Per-issue, per-station prediction arrays.
- Training checkpoints, optimizer states or pretrained upstream weights.
- Server-specific symlinks or paths.

These files can be reconstructed or mounted by an authorized user, but they
must remain ignored by version control.

## Permitted public audit layer, subject to final author review

- Scientific contracts and array schemas containing no observations.
- Aggregate categorical counts and metrics over the frozen evaluation support.
- Seed-resolved aggregate results and paired date-block contrast summaries.
- Figure/table rendering code and synthetic fixtures.
- Cryptographic hashes of private inputs and checkpoints, when the hash does
  not expose a credential or confidential filename.
- Truth-free lists of relative lead values and anonymous array dimensions.

## Required author decisions: repository and Supplementary Data 1

Do not copy the following into the candidate until the authors document both
the source-product terms and the intended public representation:

- KMA station identifiers and names;
- station latitude/longitude coordinates;
- the fitting-514/held-out-128 membership table;
- station-to-grid mappings;
- stationwise metrics, maps, case totals or time series;
- exact issue-time lists if joined to station-resolved outcomes.

For the public GitHub repository, the decision record must approve one of two
outcomes:

1. **Minimal redistribution approved.** Publish only the fields needed for
   audit, record provenance and attribution, and remove operational or
   unrelated metadata.
2. **Public redistribution excluded.** Publish schemas and synthetic examples
   only, and keep the exact contract in an authorized archive.

Both are valid repository decisions. Separately, because the complete
Supplementary Data 1 is intended to distribute selected-case time series,
stationwise verification metrics, held-out-station identifiers/names/
coordinates and distance groups through the journal and/or a permanent data
archive, the same record must explicitly approve the named sheets and fields
in that derived station-resolved scope. Repository exclusion does not imply
archive exclusion, and archive approval does not authorize station-resolved
files in GitHub. Until both scopes are signed, publication remains blocked.

Record the final choice in `DATA_REDISTRIBUTION_DECISION.md` at the repository
root using these machine-auditable fields. An author-fillable conservative
draft is available at
`docs/templates/DATA_REDISTRIBUTION_DECISION.md.in`; the template is not an
approval record and must remain outside the repository root until reviewed.

```text
repository_decision: minimal_redistribution_approved
supplementary_data_1_decision: derived_station_resolved_archive_approved
supplementary_data_1_contents: Distance_station_groups (station_id, nearest-fitting-station distance, distance group); FigS2_station_timeseries and FigS2_timeseries_summary (station_id, exact valid times, gauge truth, route predictions, member ranges); FigS3_station_members, FigS3_station_CSI and FigS3_pairwise_counts (station_id, name, lat, lon, member contingency counts and CSI, mean CSI, pairwise station tallies)
supplementary_data_1_destination: journal supplementary material and permanent data archive
approved_by: AUTHOR NAME
date: YYYY-MM-DD
source_product: OFFICIAL PRODUCT NAME
source_url: https://OFFICIAL-PRODUCT-OR-ACCESS-PAGE
accessed: YYYY-MM-DD
terms_reference: TERMS TITLE OR URL REVIEWED BY THE AUTHORS
```

The alternative repository decision value is
`public_redistribution_excluded`. The Supplementary Data 1 decision must be
`derived_station_resolved_archive_approved` unless the associated manuscript
and data-availability promises are first changed. The final
record must also identify the KMA source product, its official access URL, the
date on which it was accessed, and the applicable access or redistribution
terms (including a terms URL when available). Do not add this record until the
responsible author has reviewed and approved both the choice and that
provenance.

## User-supplied prepared inputs

A full rerun may reference repository-relative mount points such as:

```text
data/private/
├── radar_4km10min_tiff/       # prepared 4-km HSR archive
├── gauge_5min/
│   └── gauge_rn60_rn15.csv  # authorized gauge histories and truth
├── station_contract/
│   ├── mapping.csv
│   ├── stations.csv
│   └── split.csv
├── exprecast/                 # user-obtained source and weights
└── checkpoints/               # locally generated checkpoints
```

The Direct R2P documentation passes these locations explicitly through
`--radar-root`, `--gauge-csv`, `--mapping-csv`, `--stations-csv`,
`--split-csv`, and `--output-root`; users need not reproduce the runner's
historical project-relative defaults. The public code and documentation must
not contain usernames, machine names or author-specific absolute paths.

The exact paper contract is byte-hash checked and therefore requires the
authorized original mapping, station and split files. A user-constructed
replacement can be run with `--allow-unfrozen-contract`, but it defines a new
station/data contract and is not an exact reproduction of the paper split.

## Public result policy

Every public aggregate result must identify:

1. the scientific-contract version;
2. the route ID and reader-facing display name;
3. checkpoint/source hashes where applicable;
4. issue-time count, station count, lead and threshold axes;
5. missing-value and pooling rules;
6. seed aggregation rule;
7. bootstrap unit, replicate count and random seed.

Standard and radar-only Direct R2P results additionally require exact truth
and axis equality and a matching checkpoint hash within every seed.

## Retention outside GitHub

Private arrays and large archives may be retained in controlled project storage
with checksums and manifests. A GitHub release should contain only the compact
audit layer needed to substantiate the manuscript values. Storage location is
not a redistribution permission.
