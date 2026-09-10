# Data and artifact policy

KMA observations and station-resolved derivatives have separate redistribution
requirements from the code.

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

## Public reproducibility materials

- Scientific contracts and array schemas containing no observations.
- Aggregate categorical counts and metrics over the frozen evaluation support.
- Seed-resolved aggregate results and paired date-block contrast summaries.
- Figure/table rendering code and synthetic fixtures.
- Cryptographic hashes of private inputs and checkpoints, when the hash does
  not expose a credential or confidential filename.
- Truth-free lists of relative lead values and anonymous array dimensions.

## Repository and Supplementary Data 1 scopes

The public source repository excludes:

- KMA station identifiers and names;
- station latitude/longitude coordinates;
- the fitting-514/held-out-128 membership table;
- station-to-grid mappings;
- stationwise metrics, maps, case totals or time series;
- exact issue-time lists if joined to station-resolved outcomes.

Station-resolved derived verification tables are provided separately in
Supplementary Data 1; they are not bundled with the source repository.

Source-product attribution, official access links and applicable third-party
terms are summarized in [Third-party notices](../THIRD_PARTY_NOTICES.md).

## User-supplied prepared inputs

A full rerun requires user-supplied inputs, for example:

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
`--split-csv`, and `--output-root`.

The exact paper contract is byte-hash checked and therefore requires the
authorized original mapping, station and split files. A user-constructed
replacement can be run with `--allow-unfrozen-contract`, but it defines a new
station/data contract and is not an exact reproduction of the paper split.

## Result metadata

The aggregate results and accompanying manifests record:

1. the scientific-contract version;
2. the route ID and reader-facing display name;
3. checkpoint/source hashes where applicable;
4. issue-time count, station count, lead and threshold axes;
5. missing-value and pooling rules;
6. seed aggregation rule;
7. bootstrap unit, replicate count and random seed.

Standard and radar-only Direct R2P results additionally require exact truth
and axis equality and a matching checkpoint hash within every seed.
