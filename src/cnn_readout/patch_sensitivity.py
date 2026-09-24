"""Portable 3 x 3 versus 5 x 5 Patch-CNN sensitivity workflow.

The workflow begins from externally prepared exPreCast patch arrays.  It
locks those arrays in a manifest, performs the 2023 month-by-station-fold OOF
selection, fits three final seeds, predicts on another prepared period, and
computes paired route contrasts.  It starts after field-patch and gauge-context
export and never opens raw provider data.

Run ``python -m cnn_readout.patch_sensitivity --help`` for the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .loss import importance_corrected_mse, millimetres_from_normalized_prediction, normalize_target
from .model import CNN
from .sampling import TARGET_STRATIFIED_V1, build_target_stratum_pools, sample_target_stratified


SOURCE_SCHEMA = "cnn_patch_sensitivity_source_v1"
PREPARED_SCHEMA = "cnn_patch_sensitivity_prepared_v1"
SELECTION_SCHEMA = "cnn_patch_sensitivity_oof_selection_v1"
CHECKPOINT_SCHEMA = "cnn_patch_sensitivity_checkpoint_v1"
CHECKPOINT_SET_SCHEMA = "cnn_patch_sensitivity_checkpoint_set_v1"
PREDICTION_SCHEMA = "cnn_patch_sensitivity_predictions_v1"
EXTERNAL_PREDICTION_SCHEMA = "cnn_patch_sensitivity_external_predictions_v1"
COMPARISON_SCHEMA = "cnn_patch_sensitivity_comparison_v1"

FIELD_LEADS = np.arange(10, 181, 10, dtype=np.int16)
TARGET_LEADS = np.arange(60, 181, 10, dtype=np.int16)
REPORT_LEADS = np.asarray((60, 90, 120, 150, 180), dtype=np.int16)
THRESHOLDS = np.asarray((1.0, 5.0, 10.0, 20.0), dtype=np.float32)
HELD_MONTHS = np.asarray(("2023-06", "2023-07", "2023-08", "2023-09"), dtype="datetime64[M]")
STATION_FOLDS = (0, 1, 2)
WINDOW_INDICES = np.arange(13, dtype=np.int64)[:, None] + np.arange(6, dtype=np.int64)[None]
FINAL_SEEDS = (21040, 21041, 21042)
OOF_CELL_SEED_BASE = 7_131_000
SAMPLES_PER_EPOCH = 2_097_152
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 4096
PURGE_HOURS = 6
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 20_260_916
AUX_DIM = 39
PATCH_NORMALIZATION = "nonnegative_dBZ_divided_by_100"
ISSUE_TIME_CONVENTION = "KST-naive_int64_nanoseconds"


class ContractError(ValueError):
    """Raised when an external or generated artifact violates its contract."""


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: _sha256(root / name)
        for name in ("patch_sensitivity.py", "model.py", "loss.py", "sampling.py")
    }


def _resolve(base: Path, value: str | Path) -> Path:
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (base / result).resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _array_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _open_array(record: dict[str, Any], *, verify_hash: bool) -> np.ndarray:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_hash and _sha256(path) != record.get("sha256"):
        raise ContractError(f"array hash changed: {path}")
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(result.shape) != record.get("shape") or str(result.dtype) != record.get("dtype"):
        raise ContractError(f"array shape/dtype changed: {path}")
    return result


@dataclass
class PreparedArrays:
    manifest_path: Path
    manifest_sha256: str
    patches18: np.ndarray
    issue_times_ns: np.ndarray
    station_ids: np.ndarray
    xy_norm: np.ndarray
    station_folds: np.ndarray | None
    targets_mm: np.ndarray | None
    context36: np.ndarray | None
    oof_context36: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.issue_times_ns), len(self.station_ids), len(TARGET_LEADS))

    def require_targets(self) -> np.ndarray:
        if self.targets_mm is None:
            raise ContractError("this stage requires targets_mm")
        return self.targets_mm

    def require_context(self) -> np.ndarray:
        if self.context36 is None:
            raise ContractError("this stage requires context36")
        return self.context36

    def require_oof(self) -> tuple[np.ndarray, np.ndarray]:
        if self.station_folds is None or self.oof_context36 is None:
            raise ContractError("OOF selection requires station_folds and oof_context36")
        return self.station_folds, self.oof_context36


def _validate_source_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {"patches18", "issue_times_ns", "station_ids", "xy_norm"}
    missing = required.difference(arrays)
    if missing:
        raise ContractError(f"source config lacks required arrays: {sorted(missing)}")
    patches = arrays["patches18"]
    if not np.issubdtype(patches.dtype, np.floating):
        raise ContractError("patches18 must use a floating-point dtype")
    if patches.ndim != 5 or tuple(patches.shape[2:]) != (18, 5, 5):
        raise ContractError("patches18 must have shape [issue,station,18,5,5]")
    issues, stations = map(int, patches.shape[:2])
    if issues < 1 or stations < 1:
        raise ContractError("issue and station axes must be non-empty")
    if not np.issubdtype(arrays["issue_times_ns"].dtype, np.integer):
        raise ContractError("issue_times_ns must use an integer dtype")
    if not np.issubdtype(arrays["station_ids"].dtype, np.integer):
        raise ContractError("station_ids must use an integer dtype")
    if not np.issubdtype(arrays["xy_norm"].dtype, np.floating):
        raise ContractError("xy_norm must use a floating-point dtype")
    if arrays["issue_times_ns"].shape != (issues,):
        raise ContractError("issue_times_ns shape does not match patches18")
    if arrays["station_ids"].shape != (stations,):
        raise ContractError("station_ids shape does not match patches18")
    if arrays["xy_norm"].shape != (stations, 2):
        raise ContractError("xy_norm must have shape [station,2]")
    times = np.asarray(arrays["issue_times_ns"], dtype=np.int64)
    stations_id = np.asarray(arrays["station_ids"], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ContractError("issue_times_ns must be strictly increasing")
    if len(np.unique(stations_id)) != stations:
        raise ContractError("station_ids must be unique")
    if np.any(stations_id < 0):
        raise ContractError("station_ids must be nonnegative")
    if not np.isfinite(np.asarray(arrays["xy_norm"], dtype=np.float32)).all():
        raise ContractError("xy_norm must be finite")
    optional_shapes = {
        "station_folds": (stations,),
        "targets_mm": (issues, stations, 13),
        "context36": (issues, stations, 36),
        "oof_context36": (3, issues, stations, 36),
    }
    for name, shape in optional_shapes.items():
        if name in arrays and arrays[name].shape != shape:
            raise ContractError(f"{name} must have shape {shape}; found {arrays[name].shape}")
    if "station_folds" in arrays:
        if not np.issubdtype(arrays["station_folds"].dtype, np.integer):
            raise ContractError("station_folds must use an integer dtype")
        folds = set(np.asarray(arrays["station_folds"], dtype=np.int64).tolist())
        if not folds.issubset(STATION_FOLDS):
            raise ContractError("station_folds may contain only 0, 1 and 2")
    for name in ("targets_mm", "context36", "oof_context36"):
        if name in arrays and not np.issubdtype(arrays[name].dtype, np.floating):
            raise ContractError(f"{name} must use a floating-point dtype")


def prepare(args: argparse.Namespace) -> None:
    source_path = args.config.resolve()
    config = _read_json(source_path)
    if config.get("schema") != SOURCE_SCHEMA or not isinstance(config.get("arrays"), dict):
        raise ContractError(f"expected source schema {SOURCE_SCHEMA!r}")
    if config.get("patch_normalization") != PATCH_NORMALIZATION:
        raise ContractError(f"patch_normalization must be {PATCH_NORMALIZATION!r}")
    if config.get("issue_time_convention") != ISSUE_TIME_CONVENTION:
        raise ContractError(f"issue_time_convention must be {ISSUE_TIME_CONVENTION!r}")
    if "targets_mm" in config["arrays"] and config.get("target_units") != "mm":
        raise ContractError("target_units must be 'mm' when targets_mm is supplied")
    arrays: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for name, value in config["arrays"].items():
        path = _resolve(source_path.parent, value)
        if not path.is_file() or path.suffix != ".npy":
            raise FileNotFoundError(f"{name}: expected an existing .npy file: {path}")
        paths[name] = path
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    _validate_source_arrays(arrays)
    records = {name: _array_record(paths[name], array) for name, array in arrays.items()}
    manifest = {
        "schema": PREPARED_SCHEMA,
        "source_config_sha256": _sha256(source_path),
        "arrays": records,
        "field_lead_minutes": FIELD_LEADS.astype(int).tolist(),
        "target_lead_minutes": TARGET_LEADS.astype(int).tolist(),
        "window_indices": WINDOW_INDICES.astype(int).tolist(),
        "patch_normalization": PATCH_NORMALIZATION,
        "issue_time_convention": ISSUE_TIME_CONVENTION,
        "target_units": config.get("target_units"),
        "feature_contract": (
            "six consecutive 10-min forecast fields ending at each RN60 target lead; "
            "3x3 uses the centered crop of the same 5x5 source"
        ),
        "auxiliary_contract": "normalized x/y, lead/180, and 36-D issuance-time gauge context",
        "provenance": config.get("provenance", {}),
    }
    _atomic_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "shape": list(arrays["patches18"].shape)}, indent=2))


def load_prepared(path: Path, *, verify_hashes: bool = True) -> PreparedArrays:
    manifest_path = path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != PREPARED_SCHEMA:
        raise ContractError(f"expected prepared schema {PREPARED_SCHEMA!r}")
    if manifest.get("field_lead_minutes") != FIELD_LEADS.astype(int).tolist():
        raise ContractError("field lead contract changed")
    if manifest.get("target_lead_minutes") != TARGET_LEADS.astype(int).tolist():
        raise ContractError("target lead contract changed")
    if manifest.get("patch_normalization") != PATCH_NORMALIZATION:
        raise ContractError("patch normalization contract changed")
    if manifest.get("issue_time_convention") != ISSUE_TIME_CONVENTION:
        raise ContractError("issue-time convention changed")
    records = manifest.get("arrays", {})
    arrays = {name: _open_array(record, verify_hash=verify_hashes) for name, record in records.items()}
    _validate_source_arrays(arrays)
    return PreparedArrays(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        patches18=arrays["patches18"],
        issue_times_ns=np.asarray(arrays["issue_times_ns"], dtype=np.int64),
        station_ids=np.asarray(arrays["station_ids"], dtype=np.int64),
        xy_norm=np.asarray(arrays["xy_norm"], dtype=np.float32),
        station_folds=None if "station_folds" not in arrays else np.asarray(arrays["station_folds"], dtype=np.int8),
        targets_mm=arrays.get("targets_mm"),
        context36=arrays.get("context36"),
        oof_context36=arrays.get("oof_context36"),
    )


def _decode_local(flat: np.ndarray, issue_rows: np.ndarray, station_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lead_count = len(TARGET_LEADS)
    station_local = (flat // lead_count) % len(station_rows)
    issue_local = flat // (lead_count * len(station_rows))
    return issue_rows[issue_local], station_rows[station_local], flat % lead_count


def _gather_patches(data: PreparedArrays, issues: np.ndarray, stations: np.ndarray, leads: np.ndarray, patch_size: int) -> np.ndarray:
    full = np.asarray(data.patches18[issues, stations], dtype=np.float32)
    result = full[np.arange(len(full))[:, None], WINDOW_INDICES[leads]]
    if patch_size == 3:
        result = result[:, :, 1:4, 1:4]
    elif patch_size != 5:
        raise ValueError("patch_size must be 3 or 5")
    expected = (len(issues), 6, patch_size, patch_size)
    if (
        result.shape != expected
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or np.any(result > 1)
    ):
        raise ContractError(f"invalid gathered patch batch: {result.shape}")
    return np.ascontiguousarray(result, dtype=np.float32)


def _gather_aux(data: PreparedArrays, issues: np.ndarray, stations: np.ndarray, leads: np.ndarray, context: np.ndarray) -> np.ndarray:
    result = np.concatenate(
        (
            np.asarray(data.xy_norm[stations], dtype=np.float32),
            (TARGET_LEADS[leads].astype(np.float32) / 180.0)[:, None],
            np.asarray(context[issues, stations], dtype=np.float32),
        ),
        axis=1,
    )
    if result.shape != (len(issues), AUX_DIM) or not np.isfinite(result).all():
        raise ContractError("invalid auxiliary-feature batch")
    return np.ascontiguousarray(result, dtype=np.float32)


def _finite_pools(
    data: PreparedArrays, issue_rows: np.ndarray, station_rows: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    truth = data.require_targets()
    local = np.asarray(
        truth[np.ix_(issue_rows, station_rows, np.arange(len(TARGET_LEADS)))],
        dtype=np.float32,
    )
    finite = np.flatnonzero(np.isfinite(local.ravel()) & (local.ravel() >= 0))
    if not len(finite):
        raise ContractError("training subset contains no finite nonnegative target")
    return local, build_target_stratum_pools(local.ravel(), finite)


def _train_epoch(
    model: CNN,
    optimizer: torch.optim.Optimizer,
    data: PreparedArrays,
    issue_rows: np.ndarray,
    station_rows: np.ndarray,
    pools: tuple[np.ndarray, ...],
    context: np.ndarray,
    *,
    patch_size: int,
    sample_count: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    order, quotas, replacement = sample_target_stratified(pools, sample_count, rng)
    sizes = np.asarray([len(pool) for pool in pools], dtype=np.float64)
    natural = sizes / sizes.sum()
    sampled = quotas.astype(np.float64) / quotas.sum()
    truth = data.require_targets()
    model.train()
    total = 0.0
    seen = 0
    for start in range(0, len(order), batch_size):
        flat = order[start : start + batch_size]
        issues, stations, leads = _decode_local(flat, issue_rows, station_rows)
        patch = torch.from_numpy(
            _gather_patches(data, issues, stations, leads, patch_size)
        ).to(device)
        auxiliary = torch.from_numpy(
            _gather_aux(data, issues, stations, leads, context)
        ).to(device)
        target_mm = torch.from_numpy(
            np.asarray(truth[issues, stations, leads], dtype=np.float32)
        ).to(device)
        prediction = model(patch, auxiliary)
        loss = importance_corrected_mse(
            prediction,
            normalize_target(target_mm),
            target_mm,
            natural_fractions=natural,
            sampled_fractions=sampled,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(flat)
        seen += len(flat)
    if seen != sample_count:
        raise RuntimeError("training epoch did not consume its exact sample")
    return total / seen, quotas, replacement


@torch.inference_mode()
def _predict_local(
    model: CNN,
    data: PreparedArrays,
    issue_rows: np.ndarray,
    station_rows: np.ndarray,
    context: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = data.require_targets()
    total = len(issue_rows) * len(station_rows) * len(TARGET_LEADS)
    prediction = np.empty(total, dtype=np.float32)
    observation = np.empty(total, dtype=np.float32)
    lead_values = np.empty(total, dtype=np.int16)
    model.eval()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = np.arange(start, stop, dtype=np.int64)
        issues, stations, leads = _decode_local(flat, issue_rows, station_rows)
        patch = torch.from_numpy(
            _gather_patches(data, issues, stations, leads, patch_size)
        ).to(device)
        auxiliary = torch.from_numpy(
            _gather_aux(data, issues, stations, leads, context)
        ).to(device)
        values = millimetres_from_normalized_prediction(model(patch, auxiliary))
        prediction[start:stop] = values.detach().cpu().numpy().astype(np.float32)
        observation[start:stop] = np.asarray(
            truth[issues, stations, leads], dtype=np.float32
        )
        lead_values[start:stop] = TARGET_LEADS[leads]
    return prediction, observation, lead_values


def _categorical_counts(prediction: np.ndarray, truth: np.ndarray, leads: np.ndarray) -> np.ndarray:
    valid = np.isfinite(prediction) & np.isfinite(truth) & (truth >= 0)
    result = np.zeros((len(TARGET_LEADS), len(THRESHOLDS), 4), dtype=np.int64)
    for lead_index, lead in enumerate(TARGET_LEADS):
        at_lead = valid & (leads == lead)
        for threshold_index, threshold in enumerate(THRESHOLDS):
            observed = truth >= threshold
            forecast = prediction >= threshold
            result[lead_index, threshold_index] = (
                np.sum(at_lead & observed & forecast),
                np.sum(at_lead & observed & ~forecast),
                np.sum(at_lead & ~observed & forecast),
                np.sum(at_lead & ~observed & ~forecast),
            )
    return result


def _csi(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    denominator = value[..., 0] + value[..., 1] + value[..., 2]
    return np.divide(
        value[..., 0],
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _purged_issue_mask(issue_times_ns: np.ndarray) -> np.ndarray:
    times = np.asarray(issue_times_ns, dtype=np.int64).astype("datetime64[ns]")
    months = np.unique(times.astype("datetime64[M]"))
    keep = np.ones(len(times), dtype=bool)
    margin = np.timedelta64(PURGE_HOURS, "h")
    for boundary in months[1:]:
        keep &= np.abs(times - boundary.astype("datetime64[ns]")) >= margin
    return keep


def _parse_int_subset(value: str, allowed: Sequence[int]) -> tuple[int, ...]:
    if value == "all":
        return tuple(allowed)
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or not set(result).issubset(set(allowed)):
        raise ValueError(f"invalid subset {value!r}; allowed={tuple(allowed)}")
    return result


def _oof_cell_path(work_dir: Path, patch_size: int, month_index: int, fold: int) -> Path:
    return work_dir / f"patch{patch_size}" / str(HELD_MONTHS[month_index]) / f"fold_{fold}.npz"


def _run_oof_cell(
    data: PreparedArrays,
    *,
    patch_size: int,
    month_index: int,
    fold: int,
    max_epochs: int,
    samples_per_epoch: int,
    train_batch_size: int,
    validation_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    station_folds, oof_context = data.require_oof()
    month = HELD_MONTHS[month_index]
    months = data.issue_times_ns.astype("datetime64[ns]").astype("datetime64[M]")
    keep = _purged_issue_mask(data.issue_times_ns)
    train_issue = np.flatnonzero(keep & (months != month)).astype(np.int64)
    valid_issue = np.flatnonzero(keep & (months == month)).astype(np.int64)
    train_station = np.flatnonzero(station_folds != fold).astype(np.int64)
    valid_station = np.flatnonzero(station_folds == fold).astype(np.int64)
    if not all(len(value) for value in (train_issue, valid_issue, train_station, valid_station)):
        raise ContractError(f"empty OOF axis for month={month}, fold={fold}")
    local_truth, pools = _finite_pools(data, train_issue, train_station)
    finite_count = sum(len(pool) for pool in pools)
    sample_count = min(int(samples_per_epoch), finite_count)
    if sample_count < 5:
        raise ContractError("target-stratified training requires at least five samples")
    cell_seed = OOF_CELL_SEED_BASE + month_index * 100 + fold
    _set_seed(cell_seed)
    model = CNN(patch_steps=6, patch_size=patch_size, aux_dim=AUX_DIM, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    counts = np.zeros((max_epochs, len(TARGET_LEADS), len(THRESHOLDS), 4), dtype=np.int64)
    losses = np.zeros(max_epochs, dtype=np.float64)
    quotas = np.zeros((max_epochs, 5), dtype=np.int64)
    replacement = np.zeros((max_epochs, 5), dtype=bool)
    del local_truth
    context = oof_context[fold]
    for epoch in range(1, max_epochs + 1):
        rng = np.random.default_rng(cell_seed + epoch * 1_000_003)
        losses[epoch - 1], quotas[epoch - 1], replacement[epoch - 1] = _train_epoch(
            model,
            optimizer,
            data,
            train_issue,
            train_station,
            pools,
            context,
            patch_size=patch_size,
            sample_count=sample_count,
            batch_size=train_batch_size,
            rng=rng,
            device=device,
        )
        prediction, truth, leads = _predict_local(
            model,
            data,
            valid_issue,
            valid_station,
            context,
            patch_size=patch_size,
            batch_size=validation_batch_size,
            device=device,
        )
        counts[epoch - 1] = _categorical_counts(prediction, truth, leads)
        print(
            f"OOF patch={patch_size} month={month} fold={fold} "
            f"epoch={epoch}/{max_epochs} loss={losses[epoch - 1]:.7f}",
            flush=True,
        )
    return {
        "counts": counts,
        "training_loss": losses,
        "quotas": quotas,
        "replacement": replacement,
        "cell_seed": np.int64(cell_seed),
        "month_index": np.int8(month_index),
        "held_station_fold": np.int8(fold),
        "training_issues": np.int64(len(train_issue)),
        "training_stations": np.int64(len(train_station)),
        "validation_issues": np.int64(len(valid_issue)),
        "validation_stations": np.int64(len(valid_station)),
        "eligible_training_tuples": np.int64(finite_count),
        "sample_count": np.int64(sample_count),
        "requested_samples_per_epoch": np.int64(samples_per_epoch),
        "train_batch_size": np.int64(train_batch_size),
        "validation_batch_size": np.int64(validation_batch_size),
        "learning_rate": np.float64(LEARNING_RATE),
        "weight_decay": np.float64(WEIGHT_DECAY),
        "prepared_manifest_sha256": np.asarray(data.manifest_sha256),
        "patch_size": np.int8(patch_size),
        "max_epochs": np.int16(max_epochs),
        "implementation_sha256": np.asarray(_json_sha256(_implementation_hashes())),
    }


def select_epoch(args: argparse.Namespace) -> None:
    if args.patch_size not in (3, 5) or args.max_epochs < 1:
        raise ValueError("patch size must be 3 or 5 and max epochs must be positive")
    data = load_prepared(args.input, verify_hashes=not args.skip_hash_verification)
    data.require_targets()
    folds, _ = data.require_oof()
    months = data.issue_times_ns.astype("datetime64[ns]").astype("datetime64[M]")
    if not np.array_equal(np.unique(months), HELD_MONTHS):
        raise ContractError("OOF selection requires June--September 2023 only")
    if set(np.unique(folds).tolist()) != set(STATION_FOLDS):
        raise ContractError("OOF selection requires all three station folds")
    month_indices = _parse_int_subset(args.execution_months, range(4))
    selected_folds = _parse_int_subset(args.execution_folds, STATION_FOLDS)
    work_dir = args.work_dir or args.output.with_suffix("").with_name(args.output.stem + "_cells")
    implementation = _implementation_hashes()
    implementation_sha = _json_sha256(implementation)
    for month_index in month_indices:
        for fold in selected_folds:
            destination = _oof_cell_path(work_dir, args.patch_size, month_index, fold)
            if destination.is_file():
                with np.load(destination, allow_pickle=False) as archive:
                    if (
                        int(archive["patch_size"]) != args.patch_size
                        or int(archive["max_epochs"]) != args.max_epochs
                        or str(archive["prepared_manifest_sha256"]) != data.manifest_sha256
                        or int(archive["month_index"]) != month_index
                        or int(archive["held_station_fold"]) != fold
                        or int(archive["cell_seed"]) != OOF_CELL_SEED_BASE + month_index * 100 + fold
                        or int(archive["requested_samples_per_epoch"]) != args.samples_per_epoch
                        or int(archive["train_batch_size"]) != args.batch_size
                        or int(archive["validation_batch_size"]) != args.validation_batch_size
                        or float(archive["learning_rate"]) != LEARNING_RATE
                        or float(archive["weight_decay"]) != WEIGHT_DECAY
                        or str(archive["implementation_sha256"]) != implementation_sha
                    ):
                        raise ContractError(f"stale OOF cell: {destination}")
                print(f"OOF cell already complete: {destination}", flush=True)
                continue
            values = _run_oof_cell(
                data,
                patch_size=args.patch_size,
                month_index=month_index,
                fold=fold,
                max_epochs=args.max_epochs,
                samples_per_epoch=args.samples_per_epoch,
                train_batch_size=args.batch_size,
                validation_batch_size=args.validation_batch_size,
                device=_device(args.device),
            )
            _atomic_npz(destination, **values)
    paths = [
        _oof_cell_path(work_dir, args.patch_size, month_index, fold)
        for month_index in range(4)
        for fold in STATION_FOLDS
    ]
    if not all(path.is_file() for path in paths):
        print("requested cells completed; selection awaits all 12 OOF cells", flush=True)
        return
    pooled = np.zeros((args.max_epochs, len(TARGET_LEADS), len(THRESHOLDS), 4), dtype=np.int64)
    cells: list[dict[str, Any]] = []
    for month_index in range(4):
        for fold in STATION_FOLDS:
            path = _oof_cell_path(work_dir, args.patch_size, month_index, fold)
            with np.load(path, allow_pickle=False) as archive:
                if (
                    int(archive["patch_size"]) != args.patch_size
                    or int(archive["max_epochs"]) != args.max_epochs
                    or str(archive["prepared_manifest_sha256"]) != data.manifest_sha256
                    or int(archive["month_index"]) != month_index
                    or int(archive["held_station_fold"]) != fold
                    or int(archive["cell_seed"]) != OOF_CELL_SEED_BASE + month_index * 100 + fold
                    or int(archive["requested_samples_per_epoch"]) != args.samples_per_epoch
                    or int(archive["train_batch_size"]) != args.batch_size
                    or int(archive["validation_batch_size"]) != args.validation_batch_size
                    or float(archive["learning_rate"]) != LEARNING_RATE
                    or float(archive["weight_decay"]) != WEIGHT_DECAY
                    or str(archive["implementation_sha256"]) != implementation_sha
                ):
                    raise ContractError(f"OOF cell contract mismatch: {path}")
                pooled += archive["counts"]
                cells.append({
                    "held_month": str(HELD_MONTHS[month_index]),
                    "held_station_fold": fold,
                    "cell_seed": int(archive["cell_seed"]),
                    "sha256": _sha256(path),
                })
    scores = np.nanmean(_csi(pooled), axis=(1, 2))
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        raise RuntimeError("all OOF macro-CSI values are undefined")
    selected_zero = int(finite[np.argmax(scores[finite])])
    result = {
        "schema": SELECTION_SCHEMA,
        "prepared_manifest_sha256": data.manifest_sha256,
        "patch_size": args.patch_size,
        "model_parameters": sum(
            value.numel() for value in CNN(patch_size=args.patch_size, aux_dim=AUX_DIM).parameters()
        ),
        "selection_period": "2023-06 through 2023-09",
        "fold_contract": "4 held months x 3 held station folds",
        "month_boundary_purge_hours": PURGE_HOURS,
        "selection_leads_min": TARGET_LEADS.astype(int).tolist(),
        "selection_thresholds_mm": THRESHOLDS.astype(float).tolist(),
        "selection_rule": "maximum pooled OOF macro CSI; earliest exact tie",
        "macro_csi_by_epoch": scores.tolist(),
        "selected_epoch": selected_zero + 1,
        "maximum_epoch": args.max_epochs,
        "effective_cell_seed_formula": "7131000 + 100*held_month_index + held_station_fold",
        "sampling_recipe": TARGET_STRATIFIED_V1,
        "samples_per_epoch": args.samples_per_epoch,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "objective": "exact-p/q importance-corrected normalized-RN60 MSE",
        "implementation": implementation,
        "cells": cells,
    }
    _atomic_json(args.output, result)
    print(f"selected epoch {selected_zero + 1}: macro CSI={scores[selected_zero]:.8f}", flush=True)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result) or any(seed < 0 for seed in result):
        raise ValueError("--seeds must contain unique nonnegative integers")
    return result


def _fit_one_final(
    data: PreparedArrays,
    *,
    patch_size: int,
    epochs: int,
    seed: int,
    samples_per_epoch: int,
    batch_size: int,
    device: torch.device,
) -> tuple[CNN, list[float]]:
    targets = data.require_targets()
    context = data.require_context()
    issue_rows = np.arange(len(data.issue_times_ns), dtype=np.int64)
    station_rows = np.arange(len(data.station_ids), dtype=np.int64)
    _, pools = _finite_pools(data, issue_rows, station_rows)
    finite_count = sum(len(pool) for pool in pools)
    sample_count = min(int(samples_per_epoch), finite_count)
    del targets
    _set_seed(seed)
    model = CNN(patch_steps=6, patch_size=patch_size, aux_dim=AUX_DIM, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        loss, quotas, replacement = _train_epoch(
            model,
            optimizer,
            data,
            issue_rows,
            station_rows,
            pools,
            context,
            patch_size=patch_size,
            sample_count=sample_count,
            batch_size=batch_size,
            rng=rng,
            device=device,
        )
        losses.append(loss)
        print(
            f"final patch={patch_size} seed={seed} epoch={epoch}/{epochs} "
            f"loss={loss:.7f} quotas={quotas.tolist()} replacement={replacement.tolist()}",
            flush=True,
        )
    return model, losses


def _checkpoint_contract(
    data: PreparedArrays,
    selection_path: Path,
    selection: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    samples_per_epoch: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "prepared_manifest_sha256": data.manifest_sha256,
        "selection_sha256": _sha256(selection_path),
        "patch_size": int(selection["patch_size"]),
        "epochs": int(selection["selected_epoch"]),
        "seeds": list(seeds),
        "sampling_recipe": TARGET_STRATIFIED_V1,
        "samples_per_epoch": int(samples_per_epoch),
        "train_batch_size": int(batch_size),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "objective": "exact-p/q importance-corrected normalized-RN60 MSE",
        "model": {
            "class": "cnn_readout.model.CNN",
            "patch_steps": 6,
            "patch_size": int(selection["patch_size"]),
            "aux_dim": AUX_DIM,
            "dropout": 0.1,
        },
        "implementation": _implementation_hashes(),
    }


def fit_final(args: argparse.Namespace) -> None:
    data = load_prepared(args.input, verify_hashes=not args.skip_hash_verification)
    data.require_targets()
    data.require_context()
    selection_path = args.selection.resolve()
    selection = _read_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ContractError("unsupported OOF selection schema")
    if selection.get("prepared_manifest_sha256") != data.manifest_sha256:
        raise ContractError("selection and fitting data differ")
    if selection.get("implementation") != _implementation_hashes():
        raise ContractError("implementation changed after OOF selection")
    patch_size = int(selection.get("patch_size", -1))
    epochs = int(selection.get("selected_epoch", 0))
    if patch_size not in (3, 5) or not 1 <= epochs <= int(selection.get("maximum_epoch", 0)):
        raise ContractError("invalid selected patch size or epoch")
    seeds = _parse_seeds(args.seeds)
    contract = _checkpoint_contract(
        data,
        selection_path,
        selection,
        seeds=seeds,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "contract.json"
    if contract_path.is_file() and _read_json(contract_path) != contract:
        raise ContractError("existing final-fit contract differs")
    _atomic_json(contract_path, contract)
    records: list[dict[str, Any]] = []
    device = _device(args.device)
    for seed in seeds:
        checkpoint = args.output_dir / f"seed_{seed}.pt"
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                payload.get("schema") != CHECKPOINT_SCHEMA
                or payload.get("contract") != contract
                or int(payload.get("seed", -1)) != seed
            ):
                raise ContractError(f"existing checkpoint contract differs: {checkpoint}")
        else:
            model, losses = _fit_one_final(
                data,
                patch_size=patch_size,
                epochs=epochs,
                seed=seed,
                samples_per_epoch=args.samples_per_epoch,
                batch_size=args.batch_size,
                device=device,
            )
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "contract": contract,
                "seed": seed,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "training_loss_by_epoch": losses,
            }
            _atomic_torch(checkpoint, payload)
        records.append({"seed": seed, "file": checkpoint.name, "sha256": _sha256(checkpoint)})
    frozen = {
        "schema": CHECKPOINT_SET_SCHEMA,
        "contract": contract,
        "contract_sha256": _sha256(contract_path),
        "checkpoints": records,
    }
    _atomic_json(args.output_dir / "FROZEN.json", frozen)
    print(f"frozen {len(records)} final checkpoints in {args.output_dir}", flush=True)


def _load_checkpoint_set(path: Path) -> tuple[dict[str, Any], list[tuple[int, Path]]]:
    root = path.resolve()
    frozen_path = root / "FROZEN.json"
    frozen = _read_json(frozen_path)
    if frozen.get("schema") != CHECKPOINT_SET_SCHEMA:
        raise ContractError("unsupported checkpoint-set schema")
    contract_path = root / "contract.json"
    if frozen.get("contract") != _read_json(contract_path):
        raise ContractError("checkpoint-set contract differs from contract.json")
    if frozen.get("contract_sha256") != _sha256(contract_path):
        raise ContractError("checkpoint-set contract hash changed")
    records: list[tuple[int, Path]] = []
    for record in frozen.get("checkpoints", []):
        checkpoint = root / record["file"]
        if _sha256(checkpoint) != record.get("sha256"):
            raise ContractError(f"checkpoint hash changed: {checkpoint}")
        records.append((int(record["seed"]), checkpoint))
    if not records:
        raise ContractError("checkpoint set is empty")
    return frozen, records


def _save_axis(path: Path, value: np.ndarray) -> None:
    if path.is_file():
        existing = np.load(path, allow_pickle=False)
        if not np.array_equal(existing, value):
            raise ContractError(f"existing prediction axis changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _prediction_contract(
    data: PreparedArrays, checkpoint_set: Path, frozen: dict[str, Any]
) -> dict[str, Any]:
    return {
        "prepared_manifest_sha256": data.manifest_sha256,
        "checkpoint_set_sha256": _sha256(checkpoint_set / "FROZEN.json"),
        "checkpoint_contract": frozen["contract"],
        "shape": list(data.shape),
        "storage_dtype": "float16",
        "precision_contract": "millimetre predictions round-tripped through float16",
    }


def predict(args: argparse.Namespace) -> None:
    data = load_prepared(args.input, verify_hashes=not args.skip_hash_verification)
    context = data.require_context()
    frozen, records = _load_checkpoint_set(args.checkpoint_dir)
    if frozen["contract"].get("implementation") != _implementation_hashes():
        raise ContractError("implementation changed after final fitting")
    patch_size = int(frozen["contract"]["patch_size"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = _prediction_contract(data, args.checkpoint_dir.resolve(), frozen)
    contract_path = output / "contract.json"
    if contract_path.is_file() and _read_json(contract_path) != contract:
        raise ContractError("existing prediction contract differs")
    _atomic_json(contract_path, contract)
    _save_axis(output / "issue_times_ns.npy", np.asarray(data.issue_times_ns, dtype=np.int64))
    _save_axis(output / "station_ids.npy", np.asarray(data.station_ids, dtype=np.int64))
    _save_axis(output / "lead_minutes.npy", TARGET_LEADS)
    completed = output / "COMPLETED.json"
    if completed.is_file():
        marker = _read_json(completed)
        if marker.get("contract_sha256") != _sha256(contract_path):
            raise ContractError("stale prediction completion marker")
        for record in marker.get("members", []):
            if _sha256(output / record["file"]) != record.get("sha256"):
                raise ContractError("completed prediction member changed")
        print(f"predictions already complete: {output}", flush=True)
        return
    issues_total, stations_total, leads_total = data.shape
    device = _device(args.device)
    member_records: list[dict[str, Any]] = []
    for member, (seed, checkpoint) in enumerate(records):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("contract") != frozen["contract"]
            or int(payload.get("seed", -1)) != seed
        ):
            raise ContractError(f"checkpoint payload contract changed: {checkpoint}")
        model = CNN(patch_steps=6, patch_size=patch_size, aux_dim=AUX_DIM, dropout=0.1).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        destination = output / f"member_{member}.npy"
        done_path = output / f"member_{member}_done.npy"
        if destination.is_file():
            store = np.load(destination, mmap_mode="r+", allow_pickle=False)
            done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
            if store.shape != data.shape or store.dtype != np.float16 or done.shape != (issues_total,):
                raise ContractError(f"incompatible resumable prediction store: {destination}")
        else:
            store = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16, shape=data.shape)
            done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(issues_total,))
            done[:] = 0
            done.flush()
        with torch.inference_mode():
            for first in range(0, issues_total, args.issue_batch_size):
                last = min(first + args.issue_batch_size, issues_total)
                if np.asarray(done[first:last]).all():
                    continue
                batch_issues = last - first
                issues = np.repeat(np.arange(first, last, dtype=np.int64), stations_total * leads_total)
                stations = np.tile(
                    np.repeat(np.arange(stations_total, dtype=np.int64), leads_total),
                    batch_issues,
                )
                leads = np.tile(np.arange(leads_total, dtype=np.int64), batch_issues * stations_total)
                patch = torch.from_numpy(
                    _gather_patches(data, issues, stations, leads, patch_size)
                ).to(device)
                auxiliary = torch.from_numpy(
                    _gather_aux(data, issues, stations, leads, context)
                ).to(device)
                values = millimetres_from_normalized_prediction(model(patch, auxiliary))
                values_f32 = values.detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(values_f32).all():
                    raise RuntimeError("model produced a non-finite millimetre prediction")
                values_np = values_f32.astype(np.float16)
                if not np.isfinite(values_np).all():
                    raise RuntimeError("prediction overflowed during the float16 round trip")
                store[first:last] = values_np.reshape(batch_issues, stations_total, leads_total)
                store.flush()
                done[first:last] = 1
                done.flush()
                print(
                    f"predict member={member} seed={seed} issues={last}/{issues_total}",
                    flush=True,
                )
        if not np.asarray(done).all():
            raise RuntimeError(f"prediction member {member} is incomplete")
        member_records.append(
            {"member": member, "seed": seed, "file": destination.name, "sha256": _sha256(destination)}
        )
        del model
    marker = {
        "schema": PREDICTION_SCHEMA,
        "contract_sha256": _sha256(contract_path),
        "members": member_records,
        "axis_sha256": {
            name: _sha256(output / name)
            for name in ("issue_times_ns.npy", "station_ids.npy", "lead_minutes.npy")
        },
    }
    _atomic_json(completed, marker)
    print(f"completed {len(member_records)} prediction members in {output}", flush=True)


@dataclass
class PredictionRoute:
    label: str
    members: list[np.ndarray]
    issue_times_ns: np.ndarray
    station_ids: np.ndarray
    lead_minutes: np.ndarray
    source_hash: str


def _external_array(
    config_path: Path, config: dict[str, Any], name: str, *, verify_hashes: bool
) -> np.ndarray:
    if name not in config:
        raise ContractError(f"external prediction config lacks {name}")
    path = _resolve(config_path.parent, config[name])
    if not path.is_file():
        raise FileNotFoundError(path)
    hashes = config.get("sha256", {})
    if verify_hashes:
        if name not in hashes:
            raise ContractError(f"external prediction config lacks sha256 for {name}")
        if _sha256(path) != hashes[name]:
            raise ContractError(f"external prediction array hash changed: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _load_prediction_route(label: str, path: Path, *, verify_hashes: bool) -> PredictionRoute:
    source = path.resolve()
    if source.is_dir():
        marker_path = source / "COMPLETED.json"
        marker = _read_json(marker_path)
        if marker.get("schema") != PREDICTION_SCHEMA:
            raise ContractError(f"unsupported prediction directory: {source}")
        contract_path = source / "contract.json"
        if marker.get("contract_sha256") != _sha256(contract_path):
            raise ContractError(f"prediction contract hash changed: {source}")
        members: list[np.ndarray] = []
        for record in marker.get("members", []):
            member_path = source / record["file"]
            if verify_hashes and _sha256(member_path) != record.get("sha256"):
                raise ContractError(f"prediction member hash changed: {member_path}")
            members.append(np.load(member_path, mmap_mode="r", allow_pickle=False))
        issue = np.load(source / "issue_times_ns.npy", allow_pickle=False)
        station = np.load(source / "station_ids.npy", allow_pickle=False)
        lead = np.load(source / "lead_minutes.npy", allow_pickle=False)
        if verify_hashes:
            for name in ("issue_times_ns.npy", "station_ids.npy", "lead_minutes.npy"):
                if _sha256(source / name) != marker.get("axis_sha256", {}).get(name):
                    raise ContractError(f"prediction axis hash changed: {source / name}")
        source_hash = _sha256(marker_path)
    elif source.is_file() and source.suffix == ".json":
        config = _read_json(source)
        if config.get("schema") != EXTERNAL_PREDICTION_SCHEMA:
            raise ContractError(f"unsupported external prediction schema: {source}")
        values = _external_array(source, config, "predictions", verify_hashes=verify_hashes)
        if values.ndim != 4:
            raise ContractError("external predictions must have shape [member,issue,station,lead]")
        members = [values[index] for index in range(len(values))]
        issue = _external_array(source, config, "issue_times_ns", verify_hashes=verify_hashes)
        station = _external_array(source, config, "station_ids", verify_hashes=verify_hashes)
        lead = _external_array(source, config, "lead_minutes", verify_hashes=verify_hashes)
        source_hash = _sha256(source)
    else:
        raise ContractError("route path must be a prediction directory or external JSON")
    if not members:
        raise ContractError(f"prediction route has no members: {label}")
    if not all(np.issubdtype(axis.dtype, np.integer) for axis in (issue, station, lead)):
        raise ContractError(f"prediction axes must use integer dtypes for {label}")
    expected = (len(issue), len(station), len(lead))
    if any(member.shape != expected or not np.issubdtype(member.dtype, np.floating) for member in members):
        raise ContractError(f"prediction member shape/dtype changed for {label}")
    return PredictionRoute(
        label=label,
        members=members,
        issue_times_ns=np.asarray(issue, dtype=np.int64),
        station_ids=np.asarray(station, dtype=np.int64),
        lead_minutes=np.asarray(lead, dtype=np.int16),
        source_hash=source_hash,
    )


def _route_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--route must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError("--route must be LABEL=PATH")
    return label, Path(path)


def _contrast_spec(value: str) -> tuple[str, str, str]:
    fields = value.split(":")
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError("--contrast must be NAME:LEFT:RIGHT (computed as LEFT minus RIGHT)")
    return fields[0], fields[1], fields[2]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator_array,
        denominator_array,
        out=np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan),
        where=denominator_array > 0,
    )
    return float(result) if result.shape == () else result


def compare(args: argparse.Namespace) -> None:
    truth_data = load_prepared(args.truth_input, verify_hashes=not args.skip_hash_verification)
    truth = truth_data.require_targets()
    route_specs = [_route_spec(value) for value in args.route]
    if len(route_specs) < 2 or len({label for label, _ in route_specs}) != len(route_specs):
        raise ValueError("provide at least two routes with unique labels")
    routes = [
        _load_prediction_route(label, path, verify_hashes=not args.skip_hash_verification)
        for label, path in route_specs
    ]
    member_count = len(routes[0].members)
    if member_count != args.expected_members or any(len(route.members) != member_count for route in routes):
        raise ContractError(f"every route must contain exactly {args.expected_members} members")
    for route in routes:
        if (
            not np.array_equal(route.issue_times_ns, truth_data.issue_times_ns)
            or not np.array_equal(route.station_ids, truth_data.station_ids)
            or not np.array_equal(route.lead_minutes, TARGET_LEADS)
        ):
            raise ContractError(f"prediction axes differ from truth input for {route.label}")
    labels = [route.label for route in routes]
    contrast_specs = [_contrast_spec(value) for value in args.contrast]
    if not contrast_specs:
        contrast_specs = [(f"{labels[1]}_minus_{labels[0]}", labels[1], labels[0])]
    if len({name for name, _, _ in contrast_specs}) != len(contrast_specs):
        raise ValueError("contrast names must be unique")
    label_index = {label: index for index, label in enumerate(labels)}
    for _, left, right in contrast_specs:
        if left not in label_index or right not in label_index:
            raise ValueError(f"contrast references an unknown route: {left}, {right}")
    dates, date_index = np.unique(
        truth_data.issue_times_ns.astype("datetime64[ns]").astype("datetime64[D]"),
        return_inverse=True,
    )
    route_count = len(routes)
    counts = np.zeros(
        (route_count, member_count, len(dates), len(TARGET_LEADS), len(THRESHOLDS), 3),
        dtype=np.int64,
    )
    errors = np.zeros(
        (route_count, member_count, len(dates), len(TARGET_LEADS), 4),
        dtype=np.float64,
    )
    common_pairs = 0
    for day in range(len(dates)):
        rows = date_index == day
        observed = np.asarray(truth[rows], dtype=np.float32)
        predictions = [
            [np.asarray(member[rows], dtype=np.float16).astype(np.float32) for member in route.members]
            for route in routes
        ]
        valid = np.isfinite(observed) & (observed >= 0)
        for route_values in predictions:
            for member_values in route_values:
                valid &= np.isfinite(member_values)
        common_pairs += int(valid.sum())
        for route_index, route_values in enumerate(predictions):
            for member_index, forecast in enumerate(route_values):
                delta = np.where(valid, forecast - observed, 0.0)
                errors[route_index, member_index, day, :, 0] = valid.sum(axis=(0, 1))
                errors[route_index, member_index, day, :, 1] = np.abs(delta).sum(axis=(0, 1))
                errors[route_index, member_index, day, :, 2] = (delta * delta).sum(axis=(0, 1))
                errors[route_index, member_index, day, :, 3] = delta.sum(axis=(0, 1))
                for threshold_index, threshold in enumerate(THRESHOLDS):
                    observed_event = observed >= threshold
                    forecast_event = forecast >= threshold
                    counts[route_index, member_index, day, :, threshold_index] = np.stack(
                        (
                            (valid & observed_event & forecast_event).sum(axis=(0, 1)),
                            (valid & observed_event & ~forecast_event).sum(axis=(0, 1)),
                            (valid & ~observed_event & forecast_event).sum(axis=(0, 1)),
                        ),
                        axis=-1,
                    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output / "daily_statistics.npz", counts=counts, errors=errors, dates=dates)
    pooled = counts.sum(axis=2)
    summed_errors = errors.sum(axis=2)
    member_rows: list[dict[str, Any]] = []
    for route_index, label in enumerate(labels):
        for member in range(member_count):
            for lead_index, lead in enumerate(TARGET_LEADS):
                if lead not in REPORT_LEADS:
                    continue
                n, absolute, squared, signed = summed_errors[route_index, member, lead_index]
                for threshold_index, threshold in enumerate(THRESHOLDS):
                    hits, misses, false_alarms = pooled[route_index, member, lead_index, threshold_index]
                    member_rows.append({
                        "route": label,
                        "member": member,
                        "lead_min": int(lead),
                        "threshold_mm": float(threshold),
                        "finite_pairs": int(n),
                        "CSI": _safe_ratio(hits, hits + misses + false_alarms),
                        "POD": _safe_ratio(hits, hits + misses),
                        "FAR": _safe_ratio(false_alarms, hits + false_alarms),
                        "frequency_bias": _safe_ratio(hits + false_alarms, hits + misses),
                        "MAE_mm": _safe_ratio(absolute, n),
                        "RMSE_mm": float(np.sqrt(_safe_ratio(squared, n))),
                        "mean_error_mm": _safe_ratio(signed, n),
                    })
    _write_csv(output / "metrics_per_member.csv", member_rows)
    mean_rows: list[dict[str, Any]] = []
    for label in labels:
        for lead in REPORT_LEADS:
            for threshold in THRESHOLDS:
                selected = [
                    row for row in member_rows
                    if row["route"] == label and row["lead_min"] == int(lead)
                    and row["threshold_mm"] == float(threshold)
                ]
                mean_rows.append({
                    "route": label,
                    "lead_min": int(lead),
                    "threshold_mm": float(threshold),
                    **{
                        key: float(np.mean([row[key] for row in selected]))
                        for key in ("CSI", "POD", "FAR", "frequency_bias", "MAE_mm", "RMSE_mm", "mean_error_mm")
                    },
                })
    _write_csv(output / "metrics_mean_of_members.csv", mean_rows)
    point_member = _csi(pooled)
    point_route = np.mean(point_member, axis=1)
    point_contrasts = np.stack(
        [point_route[label_index[left]] - point_route[label_index[right]] for _, left, right in contrast_specs]
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    samples = np.empty(
        (args.bootstrap_resamples, len(contrast_specs), len(TARGET_LEADS), len(THRESHOLDS)),
        dtype=np.float64,
    )
    for replicate in range(args.bootstrap_resamples):
        weights = np.bincount(rng.integers(0, len(dates), len(dates)), minlength=len(dates))
        sampled_counts = np.tensordot(weights, counts, axes=(0, 2))
        sampled_route = np.mean(_csi(sampled_counts), axis=1)
        for contrast_index, (_, left, right) in enumerate(contrast_specs):
            samples[replicate, contrast_index] = (
                sampled_route[label_index[left]] - sampled_route[label_index[right]]
            )
    bounds = np.nanquantile(samples, (0.025, 0.975), axis=0)
    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, (name, left, right) in enumerate(contrast_specs):
        for lead_index, lead in enumerate(TARGET_LEADS):
            if lead not in REPORT_LEADS:
                continue
            for threshold_index, threshold in enumerate(THRESHOLDS):
                contrast_rows.append({
                    "contrast": name,
                    "left_route": left,
                    "right_route": right,
                    "lead_min": int(lead),
                    "threshold_mm": float(threshold),
                    "CSI_difference": float(point_contrasts[contrast_index, lead_index, threshold_index]),
                    "lower95": float(bounds[0, contrast_index, lead_index, threshold_index]),
                    "upper95": float(bounds[1, contrast_index, lead_index, threshold_index]),
                })
    _write_csv(output / "paired_CSI_intervals.csv", contrast_rows)
    manifest = {
        "schema": COMPARISON_SCHEMA,
        "truth_prepared_manifest_sha256": truth_data.manifest_sha256,
        "routes": [
            {"label": route.label, "source_hash": route.source_hash, "members": len(route.members)}
            for route in routes
        ],
        "contrasts": [
            {"name": name, "left": left, "right": right} for name, left, right in contrast_specs
        ],
        "common_finite_pairs_all_13_leads": common_pairs,
        "dates": len(dates),
        "reported_leads_min": REPORT_LEADS.astype(int).tolist(),
        "thresholds_mm": THRESHOLDS.astype(float).tolist(),
        "aggregation": "member metrics first, then arithmetic route mean; predictions are not ensembled",
        "precision": "every route prediction is round-tripped through float16 before common-support metrics",
        "bootstrap": {
            "unit": "paired KST issuance date; stations and members retained together",
            "repetitions": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "interval": "pointwise percentile 95%; no seed bootstrap",
        },
        "files": {
            name: _sha256(output / name)
            for name in (
                "daily_statistics.npz",
                "metrics_per_member.csv",
                "metrics_mean_of_members.csv",
                "paired_CSI_intervals.csv",
            )
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    print(f"comparison complete: {output}", flush=True)


def _add_hash_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="skip re-hashing prepared arrays (shape and contract checks still run)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="validate and lock external NPY arrays")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    validate_parser = commands.add_parser("validate", help="validate a prepared manifest")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument("--require-targets", action="store_true")
    validate_parser.add_argument("--require-context", action="store_true")
    validate_parser.add_argument("--require-oof", action="store_true")
    _add_hash_flag(validate_parser)

    select_parser = commands.add_parser("select-epoch", help="run 2023 4x3 OOF epoch selection")
    select_parser.add_argument("--input", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--work-dir", type=Path)
    select_parser.add_argument("--patch-size", type=int, choices=(3, 5), default=5)
    select_parser.add_argument("--max-epochs", type=int, default=50)
    select_parser.add_argument("--samples-per-epoch", type=int, default=SAMPLES_PER_EPOCH)
    select_parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    select_parser.add_argument("--validation-batch-size", type=int, default=65_536)
    select_parser.add_argument("--device", default="cpu")
    select_parser.add_argument("--execution-months", default="all", help="0,1,2,3 or all")
    select_parser.add_argument("--execution-folds", default="all", help="0,1,2 or all")
    _add_hash_flag(select_parser)

    fit_parser = commands.add_parser("fit-final", help="fit the selected epoch for final seeds")
    fit_parser.add_argument("--input", type=Path, required=True)
    fit_parser.add_argument("--selection", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--seeds", default=",".join(str(seed) for seed in FINAL_SEEDS))
    fit_parser.add_argument("--samples-per-epoch", type=int, default=SAMPLES_PER_EPOCH)
    fit_parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    fit_parser.add_argument("--device", default="cpu")
    _add_hash_flag(fit_parser)

    predict_parser = commands.add_parser("predict", help="predict all members on prepared arrays")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser.add_argument("--issue-batch-size", type=int, default=8)
    predict_parser.add_argument("--device", default="cpu")
    _add_hash_flag(predict_parser)

    compare_parser = commands.add_parser("compare", help="compute member-first paired route comparisons")
    compare_parser.add_argument("--truth-input", type=Path, required=True)
    compare_parser.add_argument("--route", action="append", required=True, help="LABEL=prediction-directory-or-JSON")
    compare_parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="NAME:LEFT:RIGHT, evaluated as LEFT minus RIGHT; repeat as needed",
    )
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--expected-members", type=int, default=3)
    compare_parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_REPETITIONS)
    compare_parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    _add_hash_flag(compare_parser)
    return parser


def _validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in (
        "max_epochs",
        "samples_per_epoch",
        "batch_size",
        "validation_batch_size",
        "issue_batch_size",
        "expected_members",
        "bootstrap_resamples",
    ):
        if hasattr(args, name) and int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_positive_arguments(args)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "validate":
        data = load_prepared(args.input, verify_hashes=not args.skip_hash_verification)
        if args.require_targets:
            data.require_targets()
        if args.require_context:
            data.require_context()
        if args.require_oof:
            data.require_oof()
        print(json.dumps({"schema": PREPARED_SCHEMA, "shape": list(data.shape)}, indent=2))
    elif args.command == "select-epoch":
        select_epoch(args)
    elif args.command == "fit-final":
        fit_final(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "compare":
        compare(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
