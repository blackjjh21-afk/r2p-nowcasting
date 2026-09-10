# Third-party notices

This document records provenance and exclusions. It is not a license grant and
does not replace the terms of any data provider or third-party project.

## Korea Meteorological Administration data

The experiments use Korea Meteorological Administration (KMA) radar and gauge
products. The Korean Public Data Portal identifies the KMA **Radar
Precipitation (HSR) Query Service**, **Ground (Disaster Prevention, AWS)
Meteorological Observation Data Query Service**, and **Ground-observation
Station Information Query Service** as Korea Open Government License Type 1
(KOGL Type 1: Attribution). The study obtained these products through
[KMA API Hub](https://apihub.kma.go.kr/).

- HSR service: <https://www.data.go.kr/data/15139446/openapi.do>
- AWS observation service: <https://www.data.go.kr/data/15057084/openapi.do>
- ground-observation station-information service: <https://www.data.go.kr/data/15139439/openapi.do>
- KOGL Type 1: <https://www.kogl.or.kr/info/license.do>
- KMA Open Data Portal copyright policy: <https://data.kma.go.kr/cmmn/static/staticPage.do?page=pageCr>
- KMA API Hub terms: <https://apihub.kma.go.kr/policy.do>

Use of these public products does not imply KMA endorsement of the analyses,
software or conclusions. Users must obtain the exact source products through
their own account or authorization and comply with the terms that apply to
those products. Raw observations, API credentials and station-resolved outputs
are not included in this source repository; Supplementary Data 1 is distributed
separately. See the [data policy](docs/DATA_POLICY.md) for the distribution
scope. These notices do not relicense provider data or third-party material.

## exPreCast

The exPreCast field-first route derives from the official exPreCast work by
Changhoon Song, Teng Yuan Chang and Youngjoon Hong:

- source: <https://github.com/tony890048/exPreCast>
- paper: <https://arxiv.org/abs/2602.05204>
- audited upstream revision: `092922c126bdf6098fbda2e08b2f1f3b2873bd9a`

The project-specific adaptation in `src/exprecast_adapted/` is shared with
permission from the upstream authors.

The original upstream `model.py` and pretrained or locally trained checkpoints
are not bundled. Obtain the upstream source separately at the revision above;
the loader verifies its hash before use. Source and checkpoint identities are
recorded in `configs/exprecast_adaptation_provenance.json` and
`configs/upstream_artifact_provenance.json`.

The reported permission covers sharing this adaptation; it is not treated as
a grant to relicense upstream-derived material under this repository's
BSD-3-Clause license. At the audited revision the upstream repository did not
provide an explicit license. The repository license applies to the study
authors' original contributions, not to rights in upstream-derived portions.

## pySTEPS

- project: pySTEPS
- version used in the reference environment: 1.21.1
- source: <https://github.com/pySTEPS/pysteps>
- license: BSD-3-Clause
- tested tag/commit: `v1.21.1` / `a1dd4a725d9cd960abce5555ddb4a6971c7cd08e`
- license text: <https://github.com/pySTEPS/pysteps/blob/v1.21.1/LICENSE>

## Python dependencies

Runtime dependencies are recorded in `environment.yml`. Each dependency is
subject to its own license. Listing a dependency does not grant permission to
redistribute its source, binaries, model weights or datasets.
