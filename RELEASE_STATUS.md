# Release status

Status: **version 1.1.0 prepared; new Zenodo version not yet published**.

The final paper compares pySTEPS + CNN, exPreCast + CNN and Direct R2P.
Both field-first readouts predict normalized RN60 with mean squared error.
Each uses three independently fitted readouts; exPreCast uses one fixed
upstream checkpoint. Epochs are selected using 2023-only month/station-fold
out-of-fold macro CSI and are 6 for pySTEPS and 4 for exPreCast.

The source includes the forecast-route CNN model, normalized-MSE loss,
prepared-data validation, 2023 OOF epoch selection, refitting and prediction. The Fig. 2
valid-time diagnostic retains the center-cell MLP and local-patch CNN results
in the public aggregate sources and renderer; its model-training workflow is
not distributed.

The package contains aggregate paper sources, reference images, public figure
renderers and portable execution components. Raw KMA observations, prepared
arrays, checkpoints, full predictions and restricted station-resolved values
are excluded from this software package. Authorized-input construction of
prepared tuples and upstream exPreCast training/export remain external.
The prepared separate derived-data bundle includes the permitted
station-resolved verification sheets recorded in DATA_REDISTRIBUTION_DECISION.md.

The version 1.1.0 software archive and matching derived-data bundle have not
yet been published to Zenodo. Their version-specific DOIs and publication
dates remain pending.
