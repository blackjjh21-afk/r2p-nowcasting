# CNN revision release status

Status: **version 1.1.0 prepared; new Zenodo version not yet published**.

The current manuscript compares pySTEPS + CNN, exPreCast + CNN and Direct R2P.
Both field-first readouts predict normalized RN60 with mean squared error.
Each uses three independently fitted readouts; exPreCast uses one fixed
upstream checkpoint. Epochs are selected using 2023-only month/station-fold
out-of-fold macro CSI and are 6 for pySTEPS and 4 for exPreCast.

The CNN revision changes scientific outputs and runtime code. The historical
GitHub tag `v1.0.0` and archived software DOI
10.5281/zenodo.22147192 are preserved. They do not contain these CNN results.
Software concept DOI: 10.5281/zenodo.22147191.
Historical data DOI: 10.5281/zenodo.22146749.
Data concept DOI: 10.5281/zenodo.22146748.

The current source includes the CNN model, normalized-MSE loss, prepared-data
validation, 2023 OOF epoch selection, refitting and prediction. Historical code
is retained only to make previously archived checkpoints interpretable.

The package contains aggregate paper sources, reference images, public figure
renderers and portable execution components. Raw KMA observations, prepared
arrays, checkpoints, full predictions and restricted station-resolved values
are excluded from this software package. Authorized-input construction of
prepared tuples and upstream exPreCast training/export remain external.
The separate derived-data archive includes the permitted station-resolved
verification sheets recorded in DATA_REDISTRIBUTION_DECISION.md.

New version publication requires uploading the prepared software archive and
derived-data bundle as new versions of the existing Zenodo records. Their
new version-specific DOIs must then replace the pending archive information
in release and manuscript metadata. A DOI for the old version must not be
presented as containing the CNN revision.

Verification results and exact archive hashes are written to the release
artifact directory after final synchronization and testing.
