#!/usr/bin/env python3
"""Refresh hashes for every registered public artifact and paper workflow."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/artifact_manifest.csv"

ROLES = {
    "configs/cnn_readout_provenance.json": "CNN checkpoint identities and target contract",
    "src/cnn_readout/model.py": "CNN architecture matching final research checkpoints",
    "src/cnn_readout/loss.py": "normalized RN60 MSE and exact sampling correction",
    "src/cnn_readout/sampling.py": "deterministic target-stratified sampling",
    "src/cnn_readout/workflow.py": "prepared-tuple CNN selection fitting and prediction",
    "configs/upstream_artifact_provenance.json": "non-redistributed upstream checkpoint provenance",
    "docs/PAPER_OUTPUT_REPRODUCIBILITY.md": "paper-output reproducibility scope and commands",
    "scripts/finalize_fig6_architecture.py": "exact editable-source Figure 6 finalizer",
    "scripts/refresh_artifact_manifest.py": "artifact-manifest hash refresh workflow",
    "scripts/scrub_office_metadata.py": "deterministic OOXML metadata scrubber",
    "scripts/sync_public_paper_outputs.py": "exact-schema public aggregate synchronization workflow",
    "src/paper_outputs/__init__.py": "paper-output package marker",
    "src/paper_outputs/render_publication.py": "public aggregate figure and table renderer",
    "src/paper_outputs/render_restricted.py": "authorized-input station-resolved display renderer",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_public_artifacts() -> dict[str, str]:
    result: dict[str, str] = {}
    roots = (
        (ROOT / "data/examples", {".csv", ".json"}),
        (ROOT / "data/paper_aggregates", {".csv", ".json", ".xlsx"}),
        (ROOT / "figures", {".csv", ".json", ".pdf", ".png", ".pptx", ".txt"}),
    )
    for directory, suffixes in roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("data/paper_aggregates/"):
                if path.suffix.lower() == ".xlsx":
                    role = "deterministic public-safe aggregate workbook"
                elif path.name == "manifest.json":
                    role = "public paper aggregate and reference provenance manifest"
                else:
                    role = f"public-safe aggregate sheet: {path.stem}"
            elif relative.startswith("figures/reference/"):
                role = f"hash-bound manuscript reference rendering: {path.stem}"
            elif relative.startswith("figures/main/"):
                role = f"editable-source Figure 6 artifact: {path.name}"
            elif relative.startswith("data/examples/"):
                role = f"public evaluation audit artifact: {path.name}"
            else:
                role = f"registered manuscript figure artifact: {path.name}"
            result[relative] = role
    return result


def main() -> None:
    existing: dict[str, str] = {}
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            existing[str(row["path"])] = str(row["role"])

    roles = {**existing, **discover_public_artifacts(), **ROLES}
    missing = sorted(relative for relative in roles if not (ROOT / relative).is_file())
    if missing:
        raise FileNotFoundError(f"manifested files are missing: {missing}")

    rows = [
        {"path": relative, "sha256": sha256(ROOT / relative), "role": roles[relative]}
        for relative in sorted(roles)
    ]
    temporary = MANIFEST.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "role"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(MANIFEST)
    print(f"wrote {len(rows)} artifact records: {MANIFEST}")


if __name__ == "__main__":
    main()
