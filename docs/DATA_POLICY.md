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

## Public audit layer

- Scientific contracts and array schemas containing no observations.
- Aggregate categorical counts and metrics over the frozen evaluation support.
- Seed-resolved aggregate results and paired date-block contrast summaries.
- Figure/table rendering code and synthetic fixtures.
- Cryptographic hashes of private inputs and checkpoints, when the hash does
  not expose a credential or confidential filename.
- Truth-free lists of relative lead values and anonymous array dimensions.

## Approved repository and Supplementary Data 1 scopes

Under the recorded repository decision, the following station-resolved
materials are excluded from this public source repository:

- KMA station identifiers and names;
- station latitude/longitude coordinates;
- the fitting-514/held-out-128 membership table;
- station-to-grid mappings;
- stationwise metrics, maps, case totals or time series;
- exact issue-time lists if joined to station-resolved outcomes.

The recorded decision excludes station-resolved material from the public
repository and separately approves the named derived sheets in Supplementary
Data 1 for journal and permanent-archive distribution. Repository exclusion
does not imply archive exclusion, and archive approval does not authorize
station-resolved files in GitHub.

The machine-auditable `DATA_REDISTRIBUTION_DECISION.md` at the repository root
uses `public_redistribution_excluded` for the repository and
`derived_station_resolved_archive_approved` for Supplementary Data 1. It also
identifies the KMA source products, official access URLs, access date and terms
reviewed by the responsible author.

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
