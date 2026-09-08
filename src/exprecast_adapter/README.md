# exPreCast field-export adapter

The paper uses a project-adapted exPreCast configuration. Upstream exPreCast
source, training code and weights are not vendored here. Users obtain those
items under their original terms and export the selected field checkpoint to
an HDF5 file containing:

- `forecast_normalized_dbz`: `[issue,18,256,256]`, finite values in `[0,1]`;
- `issue_time_ns`: increasing issue times as Unix nanoseconds;
- `lead_minutes`: `10,20,...,180`;
- optional `completed`: one true value per issue;
- optional `contract_json` attribute containing `checkpoint_sha256`.

Audit the export:

```bash
python -m exprecast_adapter.adapter audit-field \
  --field-h5 /path/to/exprecast_fields.h5 \
  --report-json /path/to/field_audit.json
```

Create the frozen 3 x 3 station-patch cache used by CNN:

```bash
python -m exprecast_adapter.adapter extract-patches \
  --field-h5 /path/to/exprecast_fields.h5 \
  --mapping-csv /path/to/station_mapping.csv \
  --output /path/to/exprecast_patch_cache
```

The mapping requires `station_id`, `exprecast_y` and `exprecast_x`. The cache
contains 18 forecast patches per issue and station. `WINDOW_INDICES` converts
these to the 13 six-field sequences at `L-50,...,L` used to predict RN60 at
`L=60,70,...,180` min. This adapter makes the downstream interface explicit;
it does not claim to redistribute or independently reproduce exPreCast.
