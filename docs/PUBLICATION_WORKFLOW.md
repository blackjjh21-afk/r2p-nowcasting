# Publication workflow

The contents of `release_4km10min/` are the root of the new public
repository. Do not publish the parent `github/` directory: it contains the
legacy 2-km/5-min package and private-project history that are outside this
paper's release contract.

## 1. Complete release metadata

The approved BSD-3-Clause `LICENSE` and responsible-author
`DATA_REDISTRIBUTION_DECISION.md` are installed. Reserve the archived software
DOI, then create `CITATION.cff` from `docs/templates/CITATION.cff.in` with the
final repository URL, version, release date and DOI. Omit `preferred-citation`
until the paper citation is final.

The conservative repository decision matches the staged tree: station identifiers,
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

## 3. Create the public repository and archive

1. Copy the *contents* of this directory into the existing public repository
   at <https://github.com/blackjjh21-afk/r2p-nowcasting>; this directory, not
   its parent `github/`, is the repository root.
2. Keep the version in `pyproject.toml`, the CFF version and the intended tag
   identical. Commit the audit-clean tree, create the annotated `v1.0.0` tag,
   push the branch and tag, and create the GitHub release.
3. Create an archive from that exact tag. Upload it manually to the existing
   Zenodo software draft that reserved DOI `10.5281/zenodo.22147192`; do not
   enable automatic GitHub--Zenodo archiving for this first release because it
   could create a second record instead of using the reserved DOI.
4. Publish the software and data Zenodo drafts and verify that both DOI links
   resolve before submitting the manuscript.

Illustrative commands after cloning the existing repository into a clean
working directory and copying this release tree into its root:

```bash
git add .
git commit -m "Release v1.0.0 reproducibility package"
git push origin main
git tag -a v1.0.0 -m "v1.0.0"
git push origin v1.0.0
git archive --format=zip --prefix=r2p-nowcasting-1.0.0/ \
  --output=r2p-nowcasting-v1.0.0.zip v1.0.0
```

External publication is deliberately not automated by this repository because
it changes public state and requires the authors' GitHub and Zenodo accounts.
