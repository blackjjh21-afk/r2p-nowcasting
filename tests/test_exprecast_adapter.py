from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from exprecast_adapter.adapter import (
    FIELD_LEADS,
    SCHEMA,
    TARGET_LEADS,
    _validate_existing_cache_for_overwrite,
    extract_station_patches,
    gather_scene_windows,
    validate_field_axes,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_validate_field_axes() -> None:
    issues = np.arange(3, dtype=np.int64)
    validate_field_axes((3, 18, 256, 256), FIELD_LEADS, issues)
    with pytest.raises(ValueError):
        validate_field_axes((3, 17, 256, 256), FIELD_LEADS, issues)


def test_patch_extraction_and_lead_windows() -> None:
    fields = np.zeros((2, 18, 256, 256), dtype=np.float32)
    for lead_index in range(18):
        fields[:, lead_index] = lead_index
    patches = extract_station_patches(
        fields, np.asarray([10, 20]), np.asarray([30, 40])
    )
    assert patches.shape == (2, 2, 18, 3, 3)
    windows = gather_scene_windows(patches)
    assert windows.shape == (2, 2, len(TARGET_LEADS), 6, 3, 3)
    np.testing.assert_array_equal(windows[0, 0, 0, :, 1, 1], np.arange(6))
    np.testing.assert_array_equal(windows[0, 0, -1, :, 1, 1], np.arange(12, 18))


def test_five_by_five_patches_preserve_center_crop_and_lead_windows() -> None:
    fields = np.arange(18 * 256 * 256, dtype=np.float32).reshape(1, 18, 256, 256)
    y, x = np.array([2, 20, 253]), np.array([30, 40, 253])
    patches5 = extract_station_patches(fields, y, x, patch_size=5)
    patches3 = extract_station_patches(fields, y, x)
    assert patches5.shape == (1, 3, 18, 5, 5)
    np.testing.assert_array_equal(patches5[:, :, :, 1:4, 1:4], patches3)
    windows5 = gather_scene_windows(patches5)
    assert windows5.shape == (1, 3, 13, 6, 5, 5)
    np.testing.assert_array_equal(windows5[:, :, -1], patches5[:, :, 12:18])


@pytest.mark.parametrize("coordinate", [0, 1, 254, 255])
def test_five_by_five_rejects_boundary_crossing(coordinate) -> None:
    with pytest.raises(ValueError, match="boundary"):
        extract_station_patches(np.zeros((1, 18, 256, 256)), np.array([coordinate]), np.array([10]), patch_size=5)


def test_five_by_five_cache_from_field_export(tmp_path) -> None:
    import h5py
    from exprecast_adapter.adapter import create_patch_cache, build_parser

    field = tmp_path / "fields.h5"
    with h5py.File(field, "w") as target:
        target.create_dataset("forecast_normalized_dbz", data=np.full((2, 18, 256, 256), .25, dtype=np.float32))
        target.create_dataset("issue_time_ns", data=np.array([0, 600_000_000_000], dtype=np.int64))
        target.create_dataset("lead_minutes", data=FIELD_LEADS)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("station_id,exprecast_y,exprecast_x\n9991,20,30\n")
    output = tmp_path / "cache5"
    create_patch_cache(field, mapping, output, batch_size=1, overwrite=False, patch_size=5)
    assert np.load(output / "patches18_f16.npy", mmap_mode="r").shape == (2, 1, 18, 5, 5)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["patch_shape"] == [5, 5]
    _validate_existing_cache_for_overwrite(output)
    args = build_parser().parse_args(["extract-patches", "--field-h5", str(field), "--mapping-csv", str(mapping), "--output", str(output), "--patch-size", "5"])
    assert args.patch_size == 5


def test_overwrite_rejects_unmarked_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "important"
    output.mkdir()
    sentinel = output / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a completed"):
        _validate_existing_cache_for_overwrite(output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_overwrite_accepts_only_hash_verified_owned_cache(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    output.mkdir()
    metadata = output / "metadata.json"
    _write_json(metadata, {"schema": SCHEMA})
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    _write_json(
        output / "COMPLETED.json",
        {"completed": True, "metadata_sha256": digest},
    )

    _validate_existing_cache_for_overwrite(output)

    _write_json(output / "COMPLETED.json", {"completed": True, "metadata_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="metadata hash"):
        _validate_existing_cache_for_overwrite(output)
