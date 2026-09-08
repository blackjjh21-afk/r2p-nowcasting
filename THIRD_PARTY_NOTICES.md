# Third-party notices

This document records provenance and exclusions. It is not a license grant and
does not replace the terms of any data provider or third-party project.

## Korea Meteorological Administration data

The experiments use Korea Meteorological Administration (KMA) radar and gauge
products. The Korean Public Data Portal identifies the KMA **Radar
Precipitation (HSR) Query Service**, **Ground (Disaster Prevention, AWS)
Meteorological Observation Data Query Service**, and **Ground-observation
Station Information Query Service** as Korea Open Government License Type 1
(KOGL Type 1: Attribution). The responsible author confirmed the KMA API Hub
acquisition route and approved the distribution scope recorded in
`DATA_REDISTRIBUTION_DECISION.md`.

- HSR service: <https://www.data.go.kr/data/15139446/openapi.do>
- AWS observation service: <https://www.data.go.kr/data/15057084/openapi.do>
- ground-observation station-information service: <https://www.data.go.kr/data/15139439/openapi.do>
- KOGL Type 1: <https://www.kogl.or.kr/info/license.do>
- KMA Open Data Portal copyright policy: <https://data.kma.go.kr/cmmn/static/staticPage.do?page=pageCr>
- KMA API Hub terms: <https://apihub.kma.go.kr/policy.do>

Use of these public products does not imply KMA endorsement of the analyses,
software or conclusions. Users must obtain the exact source products through
their own account or authorization and comply with the terms that apply to
those products. API credentials are never distributed. Raw KMA observations,
prepared 4-km TIFF archives, station-time gauge arrays and full-grid forecast
archives are excluded from this release.

Suggested Korean attribution for HSR:

> 본 연구는 기상청이 공공누리 제1유형(출처표시)으로 개방한
> 「기상청_레이더강수량(HSR) 조회서비스」 자료를 이용하였습니다.
> 원자료는 기상청 공공데이터포털에서 확인할 수 있으며, 본 연구에서는
> 이를 공간 절단, 재격자화 및 정규화하여 사용하였습니다.

Suggested attribution for the derived station-resolved Supplementary Data 1
archive:

> 본 자료는 기상청이 공공누리 제1유형(출처표시)으로 개방한
> 「기상청_레이더강수량(HSR) 조회서비스」, 「기상청_지상(방재,
> AWS)기상관측자료 조회서비스」 및 「기상청_지상기상관측 지점정보
> 조회서비스」를 이용하여 작성한 파생 검증자료입니다. 원자료는
> 기상청 공공데이터포털에서 확인할 수 있습니다. 이들 공공자료의
> 이용은 기상청이 본 분석 결과나 공개 코드를 보증한다는 의미가
> 아닙니다.

Station identifiers, station coordinates and station-resolved derived outputs
remain excluded from GitHub. Only the named derived sheets in Supplementary
Data 1 are approved for journal and permanent-archive distribution, as recorded
in `DATA_REDISTRIBUTION_DECISION.md`. The KOGL notices above do not relicense
third-party material.

## exPreCast

The exPreCast field-first route derives from the official exPreCast work by
Changhoon Song, Teng Yuan Chang and Youngjoon Hong:

- source: <https://github.com/tony890048/exPreCast>
- paper: <https://arxiv.org/abs/2602.05204>
- audited upstream revision: `092922c126bdf6098fbda2e08b2f1f3b2873bd9a`

No official exPreCast source files or pretrained checkpoints are redistributed
in this release. At the audited revision, the upstream repository did not
provide an explicit software or model-artifact license; users must obtain the
materials from the upstream authors and determine whether their intended use
is permitted.

The paper-facing route called **exPreCast + CNN** uses a project-specific
7-input/18-output long-horizon adaptation. It must not be represented as the
unchanged official checkpoint or as a verbatim reproduction of every upstream
training setting. Public provenance records must distinguish the upstream
architecture/source from the project adaptation and its locally trained
weights.

## pySTEPS

- project: pySTEPS
- version used in the reference environment: 1.21.1
- source: <https://github.com/pySTEPS/pysteps>
- license: BSD-3-Clause
- tested tag/commit: `v1.21.1` / `a1dd4a725d9cd960abce5555ddb4a6971c7cd08e`
- license text: <https://github.com/pySTEPS/pysteps/blob/v1.21.1/LICENSE>

The field source internally uses the Lucas-Kanade optical-flow configuration,
but figures and tables use the simplified reader-facing name **pySTEPS**. The
full route is **pySTEPS + CNN**. pySTEPS is independent of and is not
endorsed by this project.

## Python dependencies

Runtime dependencies are recorded in `environment.yml`. Each dependency is
subject to its own license. Listing a dependency does not grant permission to
redistribute its source, binaries, model weights or datasets.
