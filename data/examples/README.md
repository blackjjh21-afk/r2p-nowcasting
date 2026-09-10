# Aggregate example output

`radar_only_csi_by_lead_threshold.csv` contains only pooled, three-member
summary statistics for the same-checkpoint radar-only sensitivity. It includes
standard and radar-only CSI, their difference, and the paired issuance-date
block-bootstrap interval at five leads and four RN60 thresholds. The values
come from the three Direct R2P checkpoints selected by the final
unweighted 52-cell RN60 macro-CSI rule (13 leads × 4 thresholds).

`radar_only_audit_manifest.json` records the contract version, selected epoch
and checkpoint hash for each seed, common evaluation axes, pooling and missing
rules, paired-bootstrap settings, and hashes of the public outputs.

`station_dropout_csi_by_lead_threshold.csv` contains the corresponding pooled,
three-member summary for the matched training-time station-dropout comparison.
It reports CSI for Direct R2P and the separately trained no-station-dropout
control, their difference, and paired issuance-date block-bootstrap intervals
at the same five leads and four RN60 thresholds.

`station_dropout_audit_manifest.json` records the two training contracts,
selected epochs and checkpoint hashes, exact truth and axis checks,
paired-bootstrap settings, and hashes of the public Fig. 4b outputs.

Both examples and their manifests omit station identifiers, coordinates,
station-time observations and private file locations.
