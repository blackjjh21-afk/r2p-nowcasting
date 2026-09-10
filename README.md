# Where historical gauge supervision enters radar-based precipitation nowcasting for gauge-referenced point accumulation

Code and aggregate results for the 4-km/10-min radar-to-point nowcasting
experiments. Experiment settings are recorded in
[`configs/scientific_contract_4km10min.json`](configs/scientific_contract_4km10min.json)
and route definitions in [`configs/route_registry.json`](configs/route_registry.json).

Software version 1.1.0:
[doi:10.5281/zenodo.22684438](https://doi.org/10.5281/zenodo.22684438).
See [`CITATION.cff`](CITATION.cff) for citation metadata.

## Scientific question

The valid-time diagnostic uses observed HSR to measure how much of the fixed
field-to-point readout deficit can be recovered by gauge-supervised center-cell
MLP and local-patch CNN readouts (Fig. 2). The forecast comparison evaluates
three complete routes at the same held-out stations, issue times, gauge truth
and reported leads:

- **pySTEPS + CNN**
- **exPreCast + CNN**
- **Direct R2P**

The two field-first routes pair different upstream field sources with
separately fitted instances of the same CNN architecture. Each readout uses a
3 × 3 forecast patch sequence and issuance-time gauge context to predict
60-minute station accumulation (RN60). Direct R2P predicts RN60 from the radar
history and gauge context without producing an intermediate forecast field.

The 128 held-out stations are excluded from fitting, checkpoint selection and
gauge-context inputs. The radar-only sensitivity test masks the remaining
fitting-station gauge context in the same Direct R2P checkpoints; it is not a
separately trained model.

## Experimental setup

| Item | Setting |
|---|---|
| Radar input | Prepared KMA 4-km TIFF fields at 10-min spacing |
| Radar history | 7 frames: -60, -50, -40, -30, -20, -10 and 0 min |
| Field forecasts | 18 fields: +10 to +180 min at 10-min spacing |
| Direct R2P output | 36 gauge-RN60 targets: +5 to +180 min at 5-min spacing |
| Reported leads | +60, +90, +120, +150 and +180 min |
| Thresholds | RN60 >= 1, 5, 10 and 20 mm |
| Station split | 514 fitting stations; 128 stations excluded from fitting |
| Evaluation support | 35,088 common issue times in June-September 2024-2025 |
| Primary score | CSI pooled over the common issue-time/station support |
| Uncertainty | Paired bootstrap over issuance-date blocks |

The 18 field-forecast frames and 36 Direct R2P targets have different time
axes. They are compared as gauge RN60 at the same five reported leads.

## Code and usage

| Component | Guide |
|---|---|
| Direct R2P training, evaluation and same-checkpoint radar-only inference | [Direct R2P](src/r2p_4km10min/README.md) |
| Forecast-route CNN validation, OOF epoch selection, refitting and prediction | [CNN readout](src/cnn_readout/README.md) |
| Deterministic pySTEPS field forecasts and station patches | [pySTEPS adapter](src/pysteps_adapter/README.md) |
| Adapted exPreCast training, RN60 checkpoint selection and field generation | [Adapted exPreCast](src/exprecast_adapted/README.md) |
| Adapted exPreCast export validation and station patches | [exPreCast adapter](src/exprecast_adapter/README.md) |
| Radar-only and station-dropout comparisons with paired date-block intervals | [Evaluation](src/evaluation/README.md) |
| Provider-format KMA HSR conversion to the 4-km/10-min grid | [HSR preprocessing](docs/HSR_PREPROCESSING_4KM10MIN.md) |
| Figure and table rendering | [Figure/table reproduction](docs/PAPER_OUTPUT_REPRODUCIBILITY.md) |
| Aggregate CSV values and member-resolved summaries | [Aggregate data](data/paper_aggregates/README.md) |

Shared metric, contract and bootstrap functions are in `src/common/`.

## Data and reproducibility

The bundled aggregates support rendering Figs. 2–6, S1/S4 and Table 1.
Fig. 1 and S2/S3 require separately obtained station or radar data; the map
renderers also require Natural Earth shapefiles. See the
[figure/table guide](docs/PAPER_OUTPUT_REPRODUCIBILITY.md) for inputs and commands.

Training and prediction require user-supplied KMA data. Raw observations,
prepared archives, checkpoints and full prediction arrays are not bundled.
The CNN workflow starts from prepared tuples; tuple construction remains an
external step. Adapted exPreCast training, checkpoint selection and field
generation are included and require separately obtained upstream source.
Upstream checkpoint
identities are recorded in
[`configs/upstream_artifact_provenance.json`](configs/upstream_artifact_provenance.json).
The valid-time MLP and CNN in Fig. 2 are provided as results and renderings,
not training workflows. This is therefore not a complete raw-data-to-results
package. Data access and redistribution details are in
[`docs/DATA_POLICY.md`](docs/DATA_POLICY.md).

## Environment

```bash
conda env create -f environment.yml
conda activate r2p-4km10min-release
python -m pip install --no-deps -e .
python -m r2p_4km10min.run_vanilla_r2p --help
pytest -q -p no:cacheprovider tests
```

The editable install makes the `src/` packages available from the checkout.
`--no-deps` preserves the versions installed from `environment.yml`; a regular
wheel installation is also supported.

GPU-specific PyTorch installation may need to be adjusted for the target CUDA
driver while retaining the recorded major software versions.

## Tests

The test suite checks model and data contracts, prepared-input validation,
evaluation routines and figure/table renderers using synthetic inputs.

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
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provider and upstream
notices, and [`CITATION.cff`](CITATION.cff) for citation metadata.
