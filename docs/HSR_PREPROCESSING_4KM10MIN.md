# KMA HSR preprocessing contract

[`prepare_kma_hsr_4km10min.py`](../src/preprocessing/prepare_kma_hsr_4km10min.py)
converts provider-format 500-m KMA HSR composites to the normalized 4-km
TIFF fields used by the public KMA exPreCast data contract and by this study.
Raw KMA data and the provider coordinate file are not redistributed here; see
[`DATA_POLICY.md`](DATA_POLICY.md).

## Frozen transform

| Item | Contract |
|---|---|
| Raw field | `2881 x 2305`, little-endian signed `int16`, in `100 x dBZ` |
| Containers | 1,024-byte archive header or 4-byte API header; gzip detected by magic bytes |
| Spatial transform | Published crop and padding, aligned `[::8, ::8]` point decimation, central `256 x 256` crop, then y-axis reversal |
| Normalization | `max(raw_100xdbz, 0) / 10000`, equivalent to nonnegative dBZ divided by 100 |
| Output | `256 x 256` normalized `float32` TIFF at 4-km spacing |
| Time contract | KST timestamps at minute phase 0 on a 10-min grid |

The aligned point decimation is part of the published checkpoint contract. It
must not be replaced by block averaging, interpolation,
2-km effective-reflectivity aggregation, or `uint8 / 255` normalization.

The final grid covers a nominal 1,024-km square. Its first 231 columns map to
provider pixels (59,136 source-backed cells); the final 25 columns retain the
published zero padding. When the provider's `rdr_500m_latlon.nc` is supplied,
the program reconstructs the final curvilinear coordinates and checks every
source-backed coordinate before writing `grid_coordinates.npz`.

Malformed frames are either rejected or, with `--skip-invalid-sources`,
recorded explicitly as unavailable. They are never replaced with artificial
zero-rain fields. Each run also writes `frames.csv` and `manifest.json` so the
selected timestamps, failures, input contract, spatial transform, and code
hash remain auditable.

Source discovery is recursive. A selected filename must contain
`RDR_CMP_HSR`, end in `.bin` or `.bin.gz`, and contain one unambiguous
12-digit `YYYYMMDDHHMM` timestamp. Duplicate timestamps are rejected. The
coordinate NetCDF must contain two-dimensional `lon` and `lat` variables, each
with shape `2881 x 2305`.

## Usage

```bash
# One-frame smoke run
python src/preprocessing/prepare_kma_hsr_4km10min.py \
  --raw-root /authorized/path/provider_hsr \
  --coordinate-netcdf /authorized/path/rdr_500m_latlon.nc \
  --output-root data/private/radar_4km10min_tiff_smoke \
  --years 2019 \
  --limit 1 \
  --workers 1

# Full paper-period preparation
python src/preprocessing/prepare_kma_hsr_4km10min.py \
  --raw-root /authorized/path/provider_hsr \
  --coordinate-netcdf /authorized/path/rdr_500m_latlon.nc \
  --output-root data/private/radar_4km10min_tiff \
  --years 2019-2025 \
  --workers 4 \
  --resume \
  --skip-invalid-sources
```

Use `--dry-run` to audit inventory and coverage without writing TIFF files.
`--minute-phase 5` exists only for explicitly labelled off-contract
sensitivity work; the paper and public KMA checkpoint use minute phase 0.
`--skip-coordinate-audit` omits `grid_coordinates.npz`; such output is useful
only for transform development and cannot satisfy the frozen Direct R2P input
contract.

A completed output root has the following layout:

```text
data/private/radar_4km10min_tiff/
├── YYYY/MM/DD/YYYYMMDDHHMM.tiff
├── grid_coordinates.npz
├── frames.csv
└── manifest.json
```

`frames.csv` records every selected source and its outcome; `manifest.json`
records the transform, inventory and contract; and `grid_coordinates.npz`
contains the final curvilinear coordinates and source-support mask.

The synthetic test compares the memory-efficient production transform against
a literal implementation of the published pad-decimate-crop sequence and
checks orientation, normalization, source support, and TIFF round-tripping:

```bash
pytest -q -p no:cacheprovider tests/test_hsr_preprocessing.py
```

## Transform provenance

The pixel-selection contract follows the public KMA preprocessing released
with exPreCast:

- <https://github.com/tony890048/Processing-Radar-Datasets/blob/main/kma_preprocessing.py>
- <https://arxiv.org/html/2602.05204#A1.SS1>
