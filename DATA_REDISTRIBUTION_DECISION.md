# KMA-derived artifact redistribution decisions

repository_decision: public_redistribution_excluded
supplementary_data_1_decision: derived_station_resolved_archive_approved
supplementary_data_1_contents: Distance_station_groups (station_id, nearest-fitting-station distance, distance group); FigS2_station_timeseries and FigS2_timeseries_summary (station_id, exact valid times, gauge truth, route predictions, member ranges); FigS3_station_members, FigS3_station_CSI and FigS3_pairwise_counts (station_id, name, lat, lon, member contingency counts and CSI, mean CSI, pairwise station tallies)
supplementary_data_1_destination: journal supplementary material and permanent data archive
approved_by: Joon-Woo Roh
date: 2026-08-28
source_product: KMA Radar Precipitation (HSR) Query Service; KMA Ground (Disaster Prevention, AWS) Meteorological Observation Data Query Service; KMA Ground-observation Station Information Query Service
source_url: https://apihub.kma.go.kr/
radar_source_url: https://www.data.go.kr/data/15139446/openapi.do
aws_source_url: https://www.data.go.kr/data/15057084/openapi.do
station_metadata_source_url: https://www.data.go.kr/data/15139439/openapi.do
accessed: 2026-08-28
terms_reviewed: 2026-08-28
terms_reference: KMA API Hub terms, https://apihub.kma.go.kr/policy.do; Korea Open Government License Type 1 (Attribution), https://www.kogl.or.kr/info/license.do; KMA Open Data Portal copyright policy, https://data.kma.go.kr/cmmn/static/staticPage.do?page=pageCr

## Scope of this decision

The study's HSR, AWS minute observations and station information were obtained
from KMA API Hub with an issued authentication key and no separately notified
access restriction. The public GitHub repository contains schemas, synthetic
examples, non-station aggregate audit results and rendering code, but does not
redistribute the API key, raw KMA files, full prediction arrays, station
identifiers, names, coordinates, the 514/128 membership table,
station-to-grid mappings, stationwise metrics or case time series.

The separate `derived_station_resolved_archive_approved` decision authorizes
distribution of only the named Supplementary Data 1 sheets through the journal
and/or a permanent data archive. Those sheets contain the derived
station-resolved material needed to verify the manuscript's transfer, case and
stationwise analyses. This decision does not authorize redistribution of the
raw KMA gauge or radar archives, complete dense prediction stores, the API
authentication key or those derived sheets in the GitHub source repository.

These decisions do not relicense KMA observations or provider metadata and do
not restrict users from obtaining the source products directly from KMA under
the applicable terms. The repository's BSD-3-Clause license covers only
author-generated software unless a file states a different approved license.
