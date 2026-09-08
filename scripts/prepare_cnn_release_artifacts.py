#!/usr/bin/env python3
"""Prepare a versioned software snapshot and separate derived-data uploads.

Only an explicitly specified Git commit is archived. Nothing is published,
tagged or pushed by this command. Restricted derived-data files are written
outside the public software repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--supplementary-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="1.1.0")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination == repo or repo in destination.parents:
        raise ValueError("release uploads must be outside the public repository")
    commit = subprocess.check_output(["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=repo, text=True).strip()
    software = destination / "software"
    data = destination / "derived_data"
    software.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    archive = software / f"r2p-nowcasting-v{args.version}.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=r2p-nowcasting-{args.version}/", f"--output={archive}", commit], cwd=repo, check=True)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    workbook = data / "Supplementary_Data_1.xlsx"
    shutil.copyfile(args.supplementary_data, workbook)
    (data / "SHA256SUMS.txt").write_text(f"{digest(workbook)}  {workbook.name}\n", encoding="utf-8")
    (data / "README.md").write_text(
        "# Derived verification data for gauge-referenced point nowcasting\n\n"
        f"Version {args.version}: CNN field-first routes and Direct R2P.\n\n"
        "Supplementary_Data_1.xlsx contains the current aggregate and permitted "
        "station-resolved verification values, source tables and experiment contracts. "
        "The README sheet explains each sheet. CNNs use normalized RN60 mean squared error. "
        "Raw KMA observations, full prediction arrays, checkpoints and credentials are absent.\n\n"
        "Source observations: Korea Meteorological Administration. Derived-data "
        "redistribution follows the responsible-author scope recorded in the software "
        "repository DATA_REDISTRIBUTION_DECISION.md.\n\n"
        "Authors: Jaehee Jung and Joon-Woo Roh; Earth & Tech Inc.; WIZAI Co., Ltd.\n"
        f"Corresponding software snapshot: {commit}.\n"
        "Upload as a new version of https://zenodo.org/records/22146749.\n",
        encoding="utf-8")
    metadata = {
        "version": args.version, "software_commit": commit,
        "software_zip_sha256": digest(archive), "supplementary_data_sha256": digest(workbook),
        "software_existing_record": "https://zenodo.org/records/22147192",
        "data_existing_record": "https://zenodo.org/records/22146749",
        "software_concept_doi": "10.5281/zenodo.22147191",
        "data_concept_doi": "10.5281/zenodo.22146748",
        "new_version_dois": "not yet assigned; old version DOIs are not reused",
        "creators": [
            {"name": "Jung, Jaehee", "orcid": "0009-0000-2112-0771", "affiliation": "Earth & Tech Inc.; WIZAI Co., Ltd."},
            {"name": "Roh, Joon-Woo", "orcid": "0000-0003-0337-4019", "affiliation": "Earth & Tech Inc.; WIZAI Co., Ltd."}],
        "publication_status": "prepared, not published"}
    (destination / "SNAPSHOT.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
