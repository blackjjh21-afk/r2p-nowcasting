#!/usr/bin/env python3
"""Audit exPreCast field exports and create CNN station-patch caches.

The adapted exPreCast training code and weights are not redistributed here.
This adapter consumes its audited HDF5 export: normalized-reflectivity fields
at +10,...,+180 min on the frozen 256 x 256 grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


FIELD_LEADS = np.arange(10, 181, 10, dtype=np.int16)
TARGET_LEADS = np.arange(60, 181, 10, dtype=np.int16)
WINDOW_INDICES = np.arange(len(TARGET_LEADS), dtype=np.int64)[:, None] + np.arange(6)[None]
GRID_SHAPE = (256, 256)
PATCH_SIZE = 3
SCHEMA = "exprecast-patch-cache-4km10min-v1"


def _validate_existing_cache_for_overwrite(output: Path) -> None:
    """Require an unmistakable adapter-owned cache before recursive deletion."""

    if output.is_symlink() or not output.is_dir():
        raise RuntimeError(
            f"refusing --overwrite for a symlink or non-directory output: {output}"
        )
    resolved = output.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected:
        raise RuntimeError(f"refusing --overwrite for protected path: {output}")
    metadata_path = output / "metadata.json"
    completed_path = output / "COMPLETED.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "refusing --overwrite because the existing directory is not a "
            f"completed {SCHEMA} cache: {output}"
        ) from exc
    if metadata.get("schema") != SCHEMA or completed.get("completed") is not True:
        raise RuntimeError(
            "refusing --overwrite because the existing directory has no valid "
            f"{SCHEMA} ownership markers: {output}"
        )
    expected_metadata_sha256 = completed.get("metadata_sha256")
    if expected_metadata_sha256 != sha256_file(metadata_path):
        raise RuntimeError(
            "refusing --overwrite because the cache metadata hash does not match "
            f"COMPLETED.json: {output}"
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_field_axes(
    shape: tuple[int, ...], lead_minutes: np.ndarray, issue_times_ns: np.ndarray
) -> None:
    if len(shape) != 4 or tuple(shape[1:]) != (len(FIELD_LEADS), *GRID_SHAPE):
        raise ValueError(
            "forecast_normalized_dbz must have shape [issue,18,256,256]; "
            f"received {shape}"
        )
    leads = np.asarray(lead_minutes, dtype=np.int16)
    issues = np.asarray(issue_times_ns, dtype=np.int64)
    if not np.array_equal(leads, FIELD_LEADS):
        raise ValueError("field lead axis must be +10,...,+180 min")
    if issues.shape != (shape[0],):
        raise ValueError("issue-time axis length does not match the field store")
    if len(np.unique(issues)) != len(issues) or (
        len(issues) > 1 and np.any(np.diff(issues) <= 0)
    ):
        raise ValueError("issue times must be unique and strictly increasing")


def inspect_hdf5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        required = {"forecast_normalized_dbz", "issue_time_ns", "lead_minutes"}
        missing = required.difference(source.keys())
        if missing:
            raise ValueError("field store lacks datasets: " + ", ".join(sorted(missing)))
        fields = source["forecast_normalized_dbz"]
        issues = np.asarray(source["issue_time_ns"], dtype=np.int64)
        leads = np.asarray(source["lead_minutes"], dtype=np.int16)
        validate_field_axes(tuple(fields.shape), leads, issues)
        if "completed" in source:
            completed = np.asarray(source["completed"], dtype=bool)
            if completed.shape != (fields.shape[0],) or not completed.all():
                raise ValueError("field export is incomplete")
        sample_rows = np.unique(
            np.linspace(0, max(fields.shape[0] - 1, 0), min(fields.shape[0], 8), dtype=int)
        )
        if len(sample_rows):
            sample = np.asarray(fields[sample_rows], dtype=np.float32)
            if (
                not np.isfinite(sample).all()
                or np.any(sample < 0.0)
                or np.any(sample > 1.0)
            ):
                raise ValueError(
                    "normalized-reflectivity sample is outside [0,1] or non-finite"
                )
        contract_text = source.attrs.get("contract_json")
        contract = {}
        if isinstance(contract_text, bytes):
            contract_text = contract_text.decode("utf-8")
        if isinstance(contract_text, str) and contract_text.strip():
            contract = json.loads(contract_text)
        return {
            "schema": "exprecast-field-export-audit-v1",
            "field_path": path.name,
            "field_sha256": sha256_file(path),
            "field_shape": list(fields.shape),
            "field_dtype": str(fields.dtype),
            "field_units": "normalized reflectivity (dBZ = 100 * value)",
            "issue_count": int(fields.shape[0]),
            "lead_minutes": leads.astype(int).tolist(),
            "checkpoint_sha256": contract.get("checkpoint_sha256"),
        }


def load_station_mapping(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    aliases = (
        ("exprecast_y", "exprecast_x"),
        ("grid_y", "grid_x"),
        ("station_y", "station_x"),
    )
    coordinate_columns = next(
        ((y, x) for y, x in aliases if y in frame.columns and x in frame.columns), None
    )
    if "station_id" not in frame.columns or coordinate_columns is None:
        raise ValueError(
            "mapping CSV requires station_id and exprecast_y/exprecast_x "
            "(grid_y/grid_x aliases are accepted)"
        )
    y_column, x_column = coordinate_columns
    station_ids = frame["station_id"].to_numpy(dtype=np.int64)
    station_y = frame[y_column].to_numpy(dtype=np.int64)
    station_x = frame[x_column].to_numpy(dtype=np.int64)
    if len(np.unique(station_ids)) != len(station_ids):
        raise ValueError("station_id values must be unique")
    margin = PATCH_SIZE // 2
    if (
        np.any(station_y < margin)
        or np.any(station_y >= GRID_SHAPE[0] - margin)
        or np.any(station_x < margin)
        or np.any(station_x >= GRID_SHAPE[1] - margin)
    ):
        raise ValueError("a station patch crosses the 256 x 256 grid boundary")
    return station_ids, station_y, station_x


def patch_indices(
    station_y: np.ndarray, station_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    offset_y, offset_x = np.meshgrid(
        np.arange(-1, 2), np.arange(-1, 2), indexing="ij"
    )
    return station_y[:, None, None] + offset_y, station_x[:, None, None] + offset_x


def extract_station_patches(
    fields: np.ndarray, station_y: np.ndarray, station_x: np.ndarray
) -> np.ndarray:
    """Extract [issue,station,18,3,3] patches from normalized fields."""

    values = np.asarray(fields)
    if values.ndim != 4 or tuple(values.shape[1:]) != (18, 256, 256):
        raise ValueError(f"invalid field tensor: {values.shape}")
    patch_y, patch_x = patch_indices(station_y, station_x)
    result = values[:, :, patch_y, patch_x].transpose(0, 2, 1, 3, 4)
    expected = (len(values), len(station_y), 18, 3, 3)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"invalid station-patch tensor: {result.shape}")
    return np.ascontiguousarray(result)


def gather_scene_windows(patches18: np.ndarray) -> np.ndarray:
    """Return [issue,station,13,6,3,3] lead-aligned CNN inputs."""

    patches = np.asarray(patches18, dtype=np.float32)
    if patches.ndim != 5 or tuple(patches.shape[2:]) != (18, 3, 3):
        raise ValueError(f"invalid 18-frame station-patch tensor: {patches.shape}")
    result = patches[:, :, WINDOW_INDICES]
    expected = (len(patches), patches.shape[1], 13, 6, 3, 3)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"invalid lead-aligned patch tensor: {result.shape}")
    return np.ascontiguousarray(result)


def create_patch_cache(
    field_path: Path,
    mapping_csv: Path,
    output: Path,
    *,
    batch_size: int,
    overwrite: bool,
) -> None:
    audit = inspect_hdf5(field_path)
    station_ids, station_y, station_x = load_station_mapping(mapping_csv)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        _validate_existing_cache_for_overwrite(output)
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent)
    )
    try:
        with h5py.File(field_path, "r") as source:
            fields = source["forecast_normalized_dbz"]
            issue_times = np.asarray(source["issue_time_ns"], dtype=np.int64)
            patches = open_memmap(
                temporary / "patches18_f16.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(fields), len(station_ids), 18, 3, 3),
            )
            for start in range(0, len(fields), batch_size):
                stop = min(start + batch_size, len(fields))
                patches[start:stop] = extract_station_patches(
                    fields[start:stop], station_y, station_x
                ).astype(np.float16)
            patches.flush()
            del patches
        np.save(temporary / "issue_times_ns.npy", issue_times, allow_pickle=False)
        np.save(temporary / "station_ids.npy", station_ids, allow_pickle=False)
        np.save(temporary / "field_lead_minutes.npy", FIELD_LEADS, allow_pickle=False)
        np.save(temporary / "target_lead_minutes.npy", TARGET_LEADS, allow_pickle=False)
        metadata = {
            "schema": SCHEMA,
            "source": audit,
            "mapping_csv_sha256": sha256_file(mapping_csv),
            "station_count": int(len(station_ids)),
            "patch_shape": [3, 3],
            "patches_shape": [len(issue_times), len(station_ids), 18, 3, 3],
            "window_indices": WINDOW_INDICES.astype(int).tolist(),
            "window_contract": "six fields at L-50,...,L for RN60 ending at L",
            "patches_sha256": sha256_file(temporary / "patches18_f16.npy"),
        }
        write_json(temporary / "metadata.json", metadata)
        write_json(
            temporary / "COMPLETED.json",
            {
                "completed": True,
                "metadata_sha256": sha256_file(temporary / "metadata.json"),
            },
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-field")
    audit.add_argument("--field-h5", type=Path, required=True)
    audit.add_argument("--report-json", type=Path)
    extract = subparsers.add_parser("extract-patches")
    extract.add_argument("--field-h5", type=Path, required=True)
    extract.add_argument("--mapping-csv", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit-field":
        report = inspect_hdf5(args.field_h5)
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "extract-patches":
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        create_patch_cache(
            args.field_h5,
            args.mapping_csv,
            args.output,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
