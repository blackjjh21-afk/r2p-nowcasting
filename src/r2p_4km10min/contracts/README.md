# Station-contract inputs

The release does not bundle the study's real station
identifiers, coordinates, fitting/held-out split, or station-to-grid mapping.
Provide them explicitly with `--mapping-csv`, `--stations-csv`, and
`--split-csv` after obtaining the underlying observations under their
applicable terms.

## Required CSV schemas

All three station files contain exactly 642 unique `station_id` values and must
describe the same stations.

### `mapping.csv`

| Column | Meaning |
|---|---|
| `station_id` | Integer station key shared by all inputs |
| `exprecast_y`, `exprecast_x` | Zero-based row and column on the prepared 256 x 256 4-km TIFF grid |
| `exprecast_nearest_distance_km` | Distance from the station coordinate to the mapped grid-cell coordinate |

The mapped cell must be source-backed rather than one of the published padding
cells.

### `stations.csv`

| Column | Meaning |
|---|---|
| `station_id`, `name` | Station key and label |
| `target_col` | Name of this station's RN60 column in the gauge CSV |
| `rn15m_col` | Name of this station's RN15 column in the gauge CSV |
| `iy`, `ix` | Zero-based indices in the original project grid used by the frozen station-attention geometry |
| `lat`, `lon` | Station latitude and longitude in degrees |
| `split` | `train` or `test`; must agree with `split.csv` |

### `split.csv`

The two columns are `station_id,split`. The frozen contract uses exactly 514
fitting-station (`train`) rows and 128 held-out (`test`) rows.

### Gauge CSV

The gauge table begins with the timestamp columns
`Year,Month,Day,Hour,Minute`. It then contains every column named by
`target_col` and `rn15m_col` in `stations.csv`. Rows must be in strictly
increasing timestamp order. Gauge values are numeric millimeters; missing
observations may be represented by `NaN`/empty numeric fields.

The following anonymous fragments illustrate column relationships only; they
do not satisfy the frozen hashes or row counts:

```csv
# mapping.csv
station_id,exprecast_y,exprecast_x,exprecast_nearest_distance_km
100001,121,98,1.42

# stations.csv
station_id,name,target_col,rn15m_col,iy,ix,lat,lon,split
100001,STN_A,RN60_100001,RN15_100001,247,198,36.50,127.25,train

# split.csv
station_id,split
100001,train

# gauge_rn60_rn15.csv
Year,Month,Day,Hour,Minute,RN60_100001,RN15_100001
2019,6,1,0,0,0.0,0.0
2019,6,1,0,5,0.0,0.0
```

## Frozen versus new contracts

Omitting `--allow-unfrozen-contract` checks byte-level hashes for the exact
paper mapping, metadata and split files; these files are not redistributed by
this release. `--allow-unfrozen-contract` disables those three hash checks
but does not relax the 642-row, 514/128 split, grid-support or schema checks.
Any such replacement is a new, explicitly documented station/data contract,
not an exact reproduction of the paper split.
