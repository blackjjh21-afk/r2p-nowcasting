# CNN revision publication workflow

Version 1.1.0 contains new CNN experiments, runtime code, figures and tables.
Version 1.0.0 and its historical tag and DOI remain unchanged.

1. Synchronize the current manuscript workbook and reference figures with
   `scripts/sync_public_paper_outputs.py`.
2. Run `python scripts/refresh_artifact_manifest.py`,
   `python scripts/audit_release.py`, and
   `pytest -q -p no:cacheprovider tests`. Staging may report the pending
   version-specific DOI; all file and scientific-contract errors must be zero.
3. Commit the audited source and push the normal GitHub branch.
4. Create a new version of software record 22147192 in Zenodo. Reserve its new
   DOI, then add that DOI to CITATION and manuscript metadata.
5. Create a new version of derived-data record 22146749 and reserve its new
   DOI. It receives the updated Supplementary Data 1 workbook and checksum.
6. Freeze the final metadata commit, tag it `v1.1.0`, create a GitHub release,
   and archive that exact tag as `r2p-nowcasting-v1.1.0.zip` with its SHA256.
7. Upload the new software and derived-data files to their new Zenodo versions,
   verify metadata and both authors' Earth & Tech Inc. and WIZAI Co., Ltd.
   affiliations, and publish.
8. Verify public archive hashes and manuscript version-specific DOI links.

The software concept DOI is 10.5281/zenodo.22147191. The data concept DOI is
10.5281/zenodo.22146748. These links cover version histories; until the new
version is published, they do not contain the current CNN artifacts.

The public software package excludes raw KMA arrays, checkpoints, full
predictions and restricted station-resolved data. The separate derived-data
archive includes only the approved verification sheets recorded in
DATA_REDISTRIBUTION_DECISION.md.
