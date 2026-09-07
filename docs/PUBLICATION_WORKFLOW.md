# Release verification and archive workflow

This repository is the audited source root of the public 4-km/10-min release.
Legacy packages and private-project history remain outside its release
contract.

## 1. Release metadata

The approved BSD-3-Clause `LICENSE` and responsible-author
`DATA_REDISTRIBUTION_DECISION.md` are installed. `CITATION.cff` records the
public repository URL, version 1.0.0, release date and published software DOI
`10.5281/zenodo.22147192`. `preferred-citation` remains omitted until the
associated paper citation is final.

The conservative repository decision matches the published tree: station identifiers,
coordinates, split tables, stationwise values and case time series are not
redistributed through GitHub. The separate Supplementary Data 1 decision covers
the exact named derived sheets intended for the journal/archive, including
held-out-station identifiers/names/coordinates, distance groups, selected-case
time series and stationwise metrics. The public figure code accepts
author-supplied versions of station-resolved inputs.

## 2. Run the local release gate

From the repository root:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name build -o -name dist -o -name '*.egg-info' \) -prune -exec rm -rf {} +
python scripts/refresh_artifact_manifest.py
python scripts/audit_release.py
python scripts/audit_release.py --release-ready
python -m pytest -q -p no:cacheprovider tests
```

Build the wheel from a clean copy, install it without dependency resolution
into a clean environment that already contains the pinned dependencies, and
run every console command with `--help`. One network-independent build route
from that prepared environment is:

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/r2p-wheel .
python -m pip install --no-deps --force-reinstall /tmp/r2p-wheel/direct_r2p_4km10min-*.whl
```

Run `r2p-paper-figures` from the repository root (or pass explicit aggregate
paths) because manuscript data and editable Figure 6 assets are checkout
artifacts rather than wheel package data. Do not commit `build/`, `dist/`,
`*.egg-info`, caches or local environments.

## 3. Verify the published repository and archive

Version 1.0.0 is published at
<https://github.com/blackjjh21-afk/r2p-nowcasting/releases/tag/v1.0.0> and
archived at <https://doi.org/10.5281/zenodo.22147192>. The version in
`pyproject.toml`, the CFF version, the Git tag and the Zenodo record must remain
identical. The Zenodo software record was populated manually, so automatic
GitHub--Zenodo archiving remains disabled to avoid creating a duplicate record.

Generate the two Zenodo files from the exact audited tag in a clean working
directory:

```bash
git archive --format=zip --prefix=r2p-nowcasting-1.0.0/ \
  --output=r2p-nowcasting-v1.0.0.zip v1.0.0
sha256sum r2p-nowcasting-v1.0.0.zip \
  > r2p-nowcasting-v1.0.0.zip.sha256
```

After an approved documentation-only correction, rerun every gate above,
update the mutable `v1.0.0` tag to the audited commit, regenerate both files,
replace them through Zenodo's published-file editing workflow, and republish
the existing record without changing its DOI or version. GitHub and Zenodo
publication actions require the authors' authenticated accounts and are not
automated by this repository.
