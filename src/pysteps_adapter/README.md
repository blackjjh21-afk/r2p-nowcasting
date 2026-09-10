# pySTEPS adapter

This package exposes the deterministic pySTEPS field source used by the
4-km/10-min field-first route.

The numerical contract is:

- input: seven normalized HSR fields (`max(dBZ, 0) / 100`) at -60, -50, ..., 0 min;
- conversion: `Z = 200 R^1.6`, retaining an exact stored zero as zero rain;
- pySTEPS: dB transform at 0.1 mm h⁻¹ with zero value -15 dB, Lucas–Kanade
  motion, then semi-Lagrangian extrapolation of the latest field;
- output: 18 fields at +10, +20, ..., +180 min;
- readout features: forecast rain rates converted back to normalized HSR,
  station-centred patches, and six-frame windows ending at +60 through +180 min.

Forecast failures raise an error. The adapter does not silently replace a
failed nowcast with persistence.

## Installation

Install the release package and the upstream pySTEPS dependency described in
`environment.yml`. Conversion, patch, and window functions can be imported
without pySTEPS because the dependency is loaded only when a forecast is run.

## Explicit-file CLI

No data root or institutional path is built into the CLI. Every input and
output path is supplied by the caller. Both `.npy` and `.npz` are accepted.

Generate a forecast:

```bash
python -m pysteps_adapter.cli forecast \
  --input /path/to/seven_normalized_hsr_fields.npz \
  --input-key normalized_hsr \
  --issue-time-ns 1719792000000000000 \
  --output /path/to/pysteps_forecast.npz
```

The input array has shape `[7, H, W]`. An NPZ output contains
`forecast_normalized_hsr`, `forecast_rain_rate_mm_h`, `lead_minutes`, and
`input_offsets_minutes`, together with the supplied `issue_times_ns`; an NPY
output contains only normalized forecast HSR.

Extract station patches and six-frame CNN windows. The mapping CSV uses
the station coordinate contract (`station_id`, `exprecast_y`,
`exprecast_x`) by default:

```bash
python -m pysteps_adapter.cli patches \
  --forecast /path/to/pysteps_forecast.npz \
  --mapping-csv /path/to/station_mapping.csv \
  --valid-mask /path/to/source_support.npy \
  --output /path/to/patch_windows.npz
```

The forecast NPZ must carry `issue_times_ns`; alternatively, pass an explicit
NPY/NPZ array with `--issue-times`. Coordinate column aliases can be supplied
with `--station-id-column`, `--y-column`, and `--x-column`. Instead of a CSV,
the three aligned arrays may be supplied explicitly with `--station-ids`,
`--station-y`, and `--station-x`.

Coordinates are zero-based integer row (`exprecast_y`) and column
(`exprecast_x`) indices. All cells in each patch must lie inside the grid and,
when a valid mask is supplied, inside valid source support. With a 3 x 3
patch, the canonical output is normalized reflectivity with keys:

- `patches`: `[N, S, L, 6, 3, 3]`, where N is issue time, S is station, and
  L is the 13 RN60 leads from +60 through +180 min;
- `issue_times_ns`: `[N]` Unix timestamps in nanoseconds;
- `station_ids`: `[S]`;
- `lead_minutes`: `[L]`.

The NPZ additionally records mapping coordinates and the six source-field
leads used for every RN60 lead. An NPY output contains only `patches`.

Pass `--overwrite` explicitly to replace an existing output.
