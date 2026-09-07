#!/usr/bin/env python3
"""Portable safety and scientific-contract audit for the 4-km/10-min release.

The default *staging* mode treats unresolved publication permissions as warnings so
that development can continue.  ``--release-ready`` turns those warnings into
hard errors.  The script uses only the Python standard library and can therefore
be run before the project environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


EXPECTED_INPUT_OFFSETS = list(range(-60, 1, 10))
EXPECTED_FIELD_LEADS = list(range(10, 181, 10))
EXPECTED_POINT_LEADS = list(range(5, 181, 5))
EXPECTED_REPORT_LEADS = [60, 90, 120, 150, 180]
EXPECTED_THRESHOLDS = [1, 5, 10, 20]
EXPECTED_CORE_ROUTES = {
    "pysteps_patch_mlp",
    "exprecast_patch_mlp",
    "direct_r2p",
}
EXPECTED_AUXILIARY_ROUTES = {
    "direct_r2p_radar_only",
    "direct_r2p_no_station_dropout",
}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
CACHE_DIR_NAMES = {
    "." + "pytest_cache",
    "." + "mypy_cache",
    "." + "ruff_cache",
    "__" + "pycache__",
    ".ipynb_checkpoints",
}
GENERATED_DIR_NAMES = {"build", "dist"}
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_SECRET_FILENAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    path: str
    message: str


class Report:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, level: str, code: str, path: str | Path, message: str) -> None:
        self.issues.append(Issue(level, code, str(path), message))

    def error(self, code: str, path: str | Path, message: str) -> None:
        self.add("ERROR", code, path, message)

    def warning(self, code: str, path: str | Path, message: str) -> None:
        self.add("WARNING", code, path, message)

    @property
    def errors(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "WARNING"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect(
    report: Report,
    condition: bool,
    path: str | Path,
    code: str,
    message: str,
) -> None:
    if not condition:
        report.error(code, path, message)


def _mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _read_json(path: Path, report: Report) -> Any | None:
    if not path.is_file():
        report.error("missing-json", path, "required JSON file is missing")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.error("invalid-json", path, f"could not parse JSON: {exc}")
        return None


def validate_scientific_contract(data: Any, path: str | Path = "scientific_contract") -> Report:
    """Validate the frozen shared 4-km/10-min comparison contract."""

    report = Report()
    if not _mapping(data):
        report.error("contract-type", path, "top level must be a JSON object")
        return report

    _expect(
        report,
        data.get("contract_id") == "r2p_4km10min_release_v1",
        path,
        "contract-id",
        "contract_id must be 'r2p_4km10min_release_v1'",
    )
    _expect(
        report,
        data.get("schema") == "r2p-4km10min-scientific-contract-v1",
        path,
        "contract-schema",
        "schema must be 'r2p-4km10min-scientific-contract-v1'",
    )

    time = data.get("time")
    _expect(report, _mapping(time), path, "time-schema", "time must be an object")
    if _mapping(time):
        checks = (
            (time.get("project_timezone") == "Asia/Seoul", "project_timezone must equal 'Asia/Seoul'"),
            (time.get("evaluation_years") == [2024, 2025], "evaluation_years must equal [2024, 2025]"),
            (time.get("evaluation_months") == [6, 7, 8, 9], "evaluation_months must equal [6, 7, 8, 9]"),
            (time.get("issue_time_count") == 35088, "issue_time_count must equal 35088"),
            (time.get("bootstrap_block") == "issuance_date", "bootstrap_block must equal 'issuance_date'"),
        )
        for ok, message in checks:
            _expect(report, ok, path, "time-value", message)

    stations = data.get("station_support")
    _expect(
        report, _mapping(stations), path, "stations-schema", "station_support must be an object"
    )
    if _mapping(stations):
        checks = (
            (stations.get("total_station_count") == 642, "total_station_count must equal 642"),
            (stations.get("fitting_station_count") == 514, "fitting_station_count must equal 514"),
            (stations.get("heldout_station_count") == 128, "heldout_station_count must equal 128"),
            (
                stations.get("primary_evaluation_support") == "heldout128",
                "primary_evaluation_support must equal 'heldout128'",
            ),
            (
                stations.get("heldout_excluded_from_fitting_and_checkpoint_selection") is True,
                "heldout_excluded_from_fitting_and_checkpoint_selection must be true",
            ),
        )
        for ok, message in checks:
            _expect(report, ok, path, "stations-value", message)
        counts = [stations.get(key) for key in ("total_station_count", "fitting_station_count", "heldout_station_count")]
        if all(isinstance(value, int) for value in counts):
            _expect(
                report,
                counts[1] + counts[2] == counts[0],
                path,
                "stations-sum",
                "fitting_station_count + heldout_station_count must equal total_station_count",
            )

    radar = data.get("radar_input")
    _expect(report, _mapping(radar), path, "radar-schema", "radar_input must be an object")
    if _mapping(radar):
        checks = (
            (radar.get("grid_spacing_km") == 4, "grid_spacing_km must equal 4"),
            (radar.get("frame_interval_minutes") == 10, "frame_interval_minutes must equal 10"),
            (radar.get("frame_count") == 7, "frame_count must equal 7"),
            (
                radar.get("relative_minutes") == EXPECTED_INPUT_OFFSETS,
                f"relative_minutes must equal {EXPECTED_INPUT_OFFSETS}",
            ),
        )
        for ok, message in checks:
            _expect(report, ok, path, "radar-value", message)

    field = data.get("field_forecast")
    _expect(
        report, _mapping(field), path, "field-schema", "field_forecast must be an object"
    )
    if _mapping(field):
        checks = (
            (field.get("frame_count") == 18, "frame_count must equal 18"),
            (
                field.get("lead_minutes") == EXPECTED_FIELD_LEADS,
                f"lead_minutes must equal {EXPECTED_FIELD_LEADS}",
            ),
            (field.get("frame_interval_minutes") == 10, "frame_interval_minutes must equal 10"),
            (field.get("native_quantity") == "instantaneous_precipitation_rate", "native_quantity must equal 'instantaneous_precipitation_rate'"),
            (field.get("native_unit") == "mm h-1", "native_unit must equal 'mm h-1'"),
        )
        for ok, message in checks:
            _expect(report, ok, path, "field-value", message)

    target = data.get("direct_r2p_target")
    _expect(
        report, _mapping(target), path, "target-schema", "direct_r2p_target must be an object"
    )
    if _mapping(target):
        checks = (
            (target.get("quantity") == "gauge_RN60", "quantity must equal 'gauge_RN60'"),
            (target.get("unit") == "mm", "unit must equal 'mm'"),
            (target.get("lead_count") == 36, "lead_count must equal 36"),
            (
                target.get("lead_minutes") == EXPECTED_POINT_LEADS,
                f"lead_minutes must equal {EXPECTED_POINT_LEADS}",
            ),
        )
        for ok, message in checks:
            _expect(report, ok, path, "target-value", message)

    selection = data.get("direct_r2p_checkpoint_selection")
    _expect(
        report,
        _mapping(selection),
        path,
        "selection-schema",
        "direct_r2p_checkpoint_selection must be an object",
    )
    if _mapping(selection):
        expected_selection = {
            "period": "June-September 2023 at fitting stations",
            "metric": "unweighted RN60 macro CSI",
            "lead_minutes": list(range(60, 181, 10)),
            "thresholds_mm": EXPECTED_THRESHOLDS,
            "cell_count": 52,
            "validation_support": (
                "all issues with any fitting-station RN60 target >=0.1 mm, "
                "plus dry issues within 180 min of such an issue"
            ),
            "validation_input_condition": {
                "query_site_gauge_history": "available through issuance",
                "station_dropout": "disabled",
            },
            "smoothing": "none",
            "excluded_epochs": [1, 2, 3],
            "tie_break": "earliest epoch",
        }
        for key, expected_value in expected_selection.items():
            _expect(
                report,
                selection.get(key) == expected_value,
                path,
                "selection-value",
                f"direct_r2p_checkpoint_selection.{key} must equal {expected_value!r}",
            )

    evaluation = data.get("paper_evaluation")
    _expect(
        report,
        _mapping(evaluation),
        path,
        "evaluation-schema",
        "paper_evaluation must be an object",
    )
    if _mapping(evaluation):
        checks = (
            (evaluation.get("truth_quantity") == "gauge_RN60", "truth_quantity must equal 'gauge_RN60'"),
            (evaluation.get("truth_unit") == "mm", "truth_unit must equal 'mm'"),
            (evaluation.get("lead_minutes") == EXPECTED_REPORT_LEADS, f"lead_minutes must equal {EXPECTED_REPORT_LEADS}"),
            (evaluation.get("thresholds_mm") == EXPECTED_THRESHOLDS, f"thresholds_mm must equal {EXPECTED_THRESHOLDS}"),
            (evaluation.get("primary_metric") == "CSI", "primary_metric must equal 'CSI'"),
            (
                evaluation.get("csi_pooling") == "pool_hit_miss_false_alarm_over_common_issue_time_and_station_support",
                "csi_pooling must encode pooled categorical counts on common support",
            ),
            (evaluation.get("seed_summary") == "arithmetic_mean_of_seed_level_CSI", "seed_summary must equal 'arithmetic_mean_of_seed_level_CSI'"),
        )
        for ok, message in checks:
            _expect(report, ok, path, "evaluation-value", message)

        uncertainty = evaluation.get("uncertainty")
        _expect(report, _mapping(uncertainty), path, "uncertainty-schema", "paper_evaluation.uncertainty must be an object")
        if _mapping(uncertainty):
            checks = (
                (uncertainty.get("method") == "paired_issuance_date_block_bootstrap", "uncertainty method must be paired_issuance_date_block_bootstrap"),
                (uncertainty.get("confidence_level") == 0.95, "confidence_level must equal 0.95"),
                (uncertainty.get("pairing_axes") == ["issue_time", "station", "lead", "threshold"], "pairing_axes must preserve issue_time/station/lead/threshold"),
            )
            for ok, message in checks:
                _expect(report, ok, path, "uncertainty-value", message)

    _expect(
        report,
        data.get("primary_routes") == ["pysteps_patch_mlp", "exprecast_patch_mlp", "direct_r2p"],
        path,
        "primary-routes",
        "primary_routes must contain the frozen three routes in display order",
    )
    radar_only = data.get("radar_only_ablation")
    _expect(report, _mapping(radar_only), path, "radar-only-schema", "radar_only_ablation must be an object")
    if _mapping(radar_only):
        expected = {
            "route_id": "direct_r2p_radar_only",
            "training": "none",
            "checkpoint_relation": "same_checkpoint_as_direct_r2p_within_seed",
            "intervention": "mask_all_issuance_time_context_gauges",
            "required_equalities": [
                "truth_values",
                "issue_time_axis",
                "station_axis",
                "lead_axis",
                "checkpoint_sha256_within_seed",
            ],
        }
        for key, expected_value in expected.items():
            _expect(report, radar_only.get(key) == expected_value, path, "radar-only-contract", f"radar_only_ablation.{key} must equal {expected_value!r}")

    no_dropout = data.get("no_station_dropout_control")
    _expect(
        report,
        _mapping(no_dropout),
        path,
        "no-dropout-schema",
        "no_station_dropout_control must be an object",
    )
    if _mapping(no_dropout):
        expected_no_dropout = {
            "route_id": "direct_r2p_no_station_dropout",
            "training_change": "station masking probabilities set to zero",
            "other_intended_changes": "none",
            "comparison": "Direct R2P minus no-station-dropout control",
        }
        for key, expected_value in expected_no_dropout.items():
            _expect(
                report,
                no_dropout.get(key) == expected_value,
                path,
                "no-dropout-contract",
                f"no_station_dropout_control.{key} must equal {expected_value!r}",
            )

    return report


def _route_by_id(routes: Any, report: Report, path: str | Path) -> dict[str, Mapping[str, Any]]:
    if not isinstance(routes, list):
        report.error("routes-schema", path, "routes must be a list")
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for index, route in enumerate(routes):
        if not _mapping(route):
            report.error("route-schema", path, f"routes[{index}] must be an object")
            continue
        route_id = route.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            report.error("route-id", path, f"routes[{index}].route_id must be a string")
            continue
        if route_id in result:
            report.error("route-duplicate", path, f"duplicate route_id: {route_id}")
            continue
        result[route_id] = route
    return result


def validate_route_registry(data: Any, path: str | Path = "route_registry") -> Report:
    """Validate canonical route names and the same-checkpoint radar-only ablation."""

    report = Report()
    if not _mapping(data):
        report.error("registry-type", path, "top level must be a JSON object")
        return report
    _expect(
        report,
        data.get("registry_id") == "field_first_vs_direct_4km10min_v1",
        path,
        "registry-id",
        "registry_id must be 'field_first_vs_direct_4km10min_v1'",
    )
    _expect(
        report,
        data.get("schema") == "r2p-4km10min-route-registry-v1",
        path,
        "registry-schema",
        "schema must be 'r2p-4km10min-route-registry-v1'",
    )

    routes = _route_by_id(data.get("routes"), report, path)
    _expect(
        report,
        EXPECTED_CORE_ROUTES.issubset(routes),
        path,
        "core-routes",
        f"routes must include {sorted(EXPECTED_CORE_ROUTES)}",
    )

    expected = {
        "pysteps_patch_mlp": {
            "display_name": "pySTEPS + Patch MLP",
            "role": "field_first_comparator",
            "field_source": "pySTEPS",
            "point_readout": "Patch MLP",
            "patch_shape": [3, 3],
            "uses_permitted_issuance_time_gauge_context": True,
            "evaluated_station_support": "heldout128",
        },
        "exprecast_patch_mlp": {
            "display_name": "exPreCast + Patch MLP",
            "role": "field_first_comparator",
            "field_source": "project_adapted_exprecast_i7_o18",
            "point_readout": "Patch MLP",
            "patch_shape": [3, 3],
            "uses_permitted_issuance_time_gauge_context": True,
            "evaluated_station_support": "heldout128",
        },
        "direct_r2p": {
            "display_name": "Direct R2P",
            "role": "primary_direct_route",
            "field_source": None,
            "point_readout": None,
            "uses_permitted_issuance_time_gauge_context": True,
            "evaluated_station_support": "heldout128",
        },
        "direct_r2p_radar_only": {
            "display_name": "Direct R2P (radar-only)",
            "role": "matched_input_ablation",
            "parent_route_id": "direct_r2p",
            "training": "none",
            "same_checkpoint_within_seed": True,
            "same_truth": True,
            "same_axes": ["issue_time", "station", "lead"],
            "masked_inputs": ["all_issuance_time_context_gauges"],
            "uses_permitted_issuance_time_gauge_context": False,
            "evaluated_station_support": "heldout128",
        },
        "direct_r2p_no_station_dropout": {
            "display_name": "Direct R2P (no station dropout)",
            "role": "matched_training_control",
            "parent_route_id": "direct_r2p",
            "training": "separate fit with station masking probabilities set to zero",
            "same_architecture": True,
            "same_fitting_stations": True,
            "same_objective_and_optimizer": True,
            "uses_permitted_issuance_time_gauge_context": True,
            "evaluated_station_support": "heldout128",
        },
    }
    for route_id, fields in expected.items():
        route = routes.get(route_id)
        if route is None:
            continue
        for key, expected_value in fields.items():
            _expect(
                report,
                route.get(key) == expected_value,
                path,
                "route-value",
                f"{route_id}.{key} must equal {expected_value!r}",
            )

    banned_display_fragments = (
        "selected long",
        "long exprecast",
        "pysteps-lk",
        "context patch",
        "stride 2",
    )
    for route_id, route in routes.items():
        display = str(route.get("display_name", "")).lower()
        for fragment in banned_display_fragments:
            _expect(
                report,
                fragment not in display,
                path,
                "route-terminology",
                f"{route_id}.display_name contains obsolete term {fragment!r}",
            )

    _expect(
        report,
        set(routes) == EXPECTED_CORE_ROUTES | EXPECTED_AUXILIARY_ROUTES,
        path,
        "route-count",
        "registry must contain exactly the three primary routes and two registered Direct R2P controls",
    )
    return report


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if any(part == ".git" for part in path.parts):
            continue
        if path.is_file() or path.is_symlink():
            yield path


def _check_release_gates(root: Path, report: Report, release_ready: bool) -> None:
    required = {
        "LICENSE": "choose and install the repository license",
        "CITATION.cff": "finalize citation metadata",
        "DATA_REDISTRIBUTION_DECISION.md": (
            "record dated author decisions for both the public-repository station-artifact "
            "scope and the station-resolved Supplementary Data 1 archive scope, with KMA "
            "source and terms provenance"
        ),
    }
    for relative, message in required.items():
        path = root / relative
        valid = path.is_file() and path.stat().st_size > 0
        text = ""
        if valid:
            text = path.read_text(encoding="utf-8", errors="replace")
            unresolved = re.search(
                r"(?i)\[(?:required|insert|choose|todo)[^\]]*\]|"
                r"this file is (?:a )?(?:decision )?template|example\.invalid",
                text,
            )
            valid = unresolved is None
        if valid and relative == "LICENSE":
            # The audit cannot decide which license is appropriate, but it can
            # prevent a decision worksheet or placeholder from satisfying the
            # publication gate accidentally.
            valid = len(text.strip()) >= 100
        if valid and relative == "CITATION.cff":
            required_cff_patterns = (
                r"(?m)^cff-version:\s*['\"]?1\.2\.0['\"]?\s*$",
                r"(?m)^title:\s*['\"]?\S.+$",
                r"(?m)^authors:\s*$",
                r"(?m)^\s*-\s*family-names:\s*['\"]?\S.+$",
                r"(?m)^\s+given-names:\s*['\"]?\S.+$",
                r"(?m)^repository-code:\s*['\"]?https://\S+",
                r"(?m)^version:\s*['\"]?\S.+$",
                r"(?m)^date-released:\s*['\"]?\d{4}-\d{2}-\d{2}['\"]?\s*$",
                r"(?m)^doi:\s*['\"]?10\.\S+",
            )
            valid = all(re.search(pattern, text) for pattern in required_cff_patterns)
        if valid and relative == "DATA_REDISTRIBUTION_DECISION.md":
            repository_decision = re.search(
                r"(?im)^repository_decision:\s*"
                r"(minimal_redistribution_approved|public_redistribution_excluded)\s*$",
                text,
            )
            supplementary_decision = re.search(
                r"(?im)^supplementary_data_1_decision:\s*"
                r"derived_station_resolved_archive_approved\s*$",
                text,
            )
            supplementary_contents = re.search(
                r"(?im)^supplementary_data_1_contents:\s*(\S.+)$", text
            )
            supplementary_destination = re.search(
                r"(?im)^supplementary_data_1_destination:\s*(\S.+)$", text
            )
            contents_text = (
                supplementary_contents.group(1).lower()
                if supplementary_contents
                else ""
            )
            destination_text = (
                supplementary_destination.group(1).lower()
                if supplementary_destination
                else ""
            )
            approver = re.search(r"(?im)^approved_by:\s*\S.+$", text)
            date = re.search(r"(?im)^date:\s*\d{4}-\d{2}-\d{2}\s*$", text)
            source_product = re.search(r"(?im)^source_product:\s*\S.+$", text)
            source_url = re.search(r"(?im)^source_url:\s*https?://\S+\s*$", text)
            accessed = re.search(r"(?im)^accessed:\s*\d{4}-\d{2}-\d{2}\s*$", text)
            terms_reference = re.search(r"(?im)^terms_reference:\s*\S.+$", text)
            valid = bool(
                repository_decision
                and supplementary_decision
                and supplementary_contents
                and supplementary_destination
                and all(
                    fragment in contents_text
                    for fragment in (
                        "distance_station_groups (station_id, "
                        "nearest-fitting-station distance, distance group)",
                        "figs2_station_timeseries and figs2_timeseries_summary "
                        "(station_id, exact valid times, gauge truth, route "
                        "predictions, member ranges)",
                        "figs3_station_members, figs3_station_csi and "
                        "figs3_pairwise_counts (station_id, name, lat, lon, "
                        "member contingency counts and csi, mean csi, pairwise "
                        "station tallies)",
                    )
                )
                and "journal" in destination_text
                and "archive" in destination_text
                and approver
                and date
                and source_product
                and source_url
                and accessed
                and terms_reference
            )
        if valid:
            continue
        if release_ready:
            report.error("release-gate", relative, message)
        else:
            report.warning("release-gate", relative, message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_artifact_manifest(root: Path, report: Report) -> None:
    manifest = root / "configs" / "artifact_manifest.csv"
    if not manifest.is_file():
        report.error("artifact-manifest", "configs/artifact_manifest.csv", "manifest is missing")
        return
    try:
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError) as exc:
        report.error("artifact-manifest", "configs/artifact_manifest.csv", str(exc))
        return
    if not rows or set(rows[0]) != {"path", "sha256", "role"}:
        report.error("artifact-manifest", "configs/artifact_manifest.csv", "expected path,sha256,role columns")
        return
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        relative = str(row.get("path", ""))
        expected = str(row.get("sha256", "")).lower()
        role = str(row.get("role", "")).strip()
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in seen
        ):
            report.error("artifact-manifest", f"configs/artifact_manifest.csv:{index}", "path is empty, duplicate, absolute, or escapes the root")
            continue
        seen.add(relative)
        if not role:
            report.error(
                "artifact-role",
                f"configs/artifact_manifest.csv:{index}",
                "role must be non-empty",
            )
        path = root / candidate
        if not path.is_file():
            report.error("artifact-missing", relative, "manifested artifact is missing")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            report.error("artifact-hash", relative, "manifest SHA-256 is malformed")
            continue
        actual = _sha256(path)
        if actual != expected:
            report.error("artifact-hash", relative, f"SHA-256 mismatch: {actual} != {expected}")

    public_suffixes = {".csv", ".json", ".pdf", ".png", ".pptx", ".txt", ".xlsx"}
    for directory in (
        root / "data" / "examples",
        root / "data" / "paper_aggregates",
        root / "figures",
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in public_suffixes:
                continue
            relative = _relative(path, root)
            if relative not in seen:
                report.error(
                    "artifact-unmanifested",
                    relative,
                    "public aggregate/figure is not listed in configs/artifact_manifest.csv",
                )

    paper_workflows = (
        "configs/upstream_artifact_provenance.json",
        "docs/PAPER_OUTPUT_REPRODUCIBILITY.md",
        "scripts/finalize_fig6_architecture.py",
        "scripts/refresh_artifact_manifest.py",
        "scripts/scrub_office_metadata.py",
        "scripts/sync_public_paper_outputs.py",
        "src/paper_outputs/__init__.py",
        "src/paper_outputs/render_publication.py",
        "src/paper_outputs/render_restricted.py",
    )
    for relative in paper_workflows:
        path = root / relative
        if not path.is_file():
            report.error("paper-workflow-missing", relative, "required paper-output workflow is missing")
        elif relative not in seen:
            report.error(
                "artifact-unmanifested",
                relative,
                "paper-output workflow is not listed in configs/artifact_manifest.csv",
            )


def _check_notebook(path: Path, relative: str, report: Report) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.error("notebook-json", relative, f"invalid notebook JSON: {exc}")
        return
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        report.error("notebook-schema", relative, "notebook must use nbformat 4 with a cells list")
        return
    if notebook.get("metadata", {}).get("widgets"):
        report.error("notebook-widgets", relative, "widget state must be removed")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            report.error("notebook-execution", relative, f"code cell {index} has execution_count")
        if cell.get("outputs") not in ([], None):
            report.error("notebook-output", relative, f"code cell {index} contains outputs")


def _check_text(path: Path, relative: str, report: Report) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return

    # Build path expressions from fragments so the audit does not flag its own source.
    unix_roots = ("home", "Users", "wind" + "lidar", "data" + "01", "mnt")
    host_path = re.compile(r"/(?:" + "|".join(unix_roots) + r")/[A-Za-z0-9_.~-]+(?:/[A-Za-z0-9_.~+@%=-]+)+")
    windows_path = re.compile(r"[A-Za-z]:\\(?:Users|Documents and Settings)\\")
    uri_prefix = "file" + "://"
    if host_path.search(text) or windows_path.search(text) or uri_prefix in text:
        report.error("absolute-host-path", relative, "contains a machine-specific absolute path")

    secret_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{24,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(
            r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
        ),
    )
    if any(pattern.search(text) for pattern in secret_patterns):
        report.error("secret-content", relative, "contains a credential-like value")


def _check_office_container(path: Path, relative: str, report: Report) -> None:
    """Inspect OOXML internals that are invisible to ordinary text scans."""

    try:
        with zipfile.ZipFile(path) as archive:
            name_list = archive.namelist()
            names = set(name_list)
            if len(names) != len(name_list):
                report.error(
                    "office-member",
                    relative,
                    "OOXML archive contains duplicate member names",
                )
            text_parts: list[str] = []
            for name in sorted(names):
                member = Path(name)
                if (
                    name.startswith(("/", "\\"))
                    or "\\" in name
                    or ".." in member.parts
                ):
                    report.error(
                        "office-member",
                        relative,
                        f"OOXML archive contains an unsafe member path: {name}",
                    )
                lower_name = name.lower()
                hidden_parts = (
                    "comments",
                    "notesslides/",
                    "customxml/",
                    "embeddings/",
                    "externallinks/",
                    "vbaproject.bin",
                )
                if any(fragment in lower_name for fragment in hidden_parts):
                    report.error(
                        "office-hidden-content",
                        relative,
                        f"OOXML archive retains hidden or embedded content: {name}",
                    )
                if lower_name.endswith((".xml", ".rels", ".txt", ".csv")):
                    info = archive.getinfo(name)
                    if info.file_size > 10 * 1024 * 1024:
                        report.error(
                            "office-member-size",
                            relative,
                            f"OOXML text member exceeds 10 MiB: {name}",
                        )
                        continue
                    text = archive.read(name).decode("utf-8", errors="replace")
                    text_parts.append(text)
                    if lower_name.endswith(".rels") and re.search(
                        r"\bTargetMode\s*=\s*['\"]External['\"]", text, re.I
                    ):
                        report.error(
                            "office-external-relationship",
                            relative,
                            f"OOXML relationship points outside the archive: {name}",
                        )

            core_name = "docProps/core.xml"
            if core_name in names:
                root = ET.fromstring(archive.read(core_name))
                metadata_fields = {
                    "title": "{http://purl.org/dc/elements/1.1/}title",
                    "subject": "{http://purl.org/dc/elements/1.1/}subject",
                    "creator": "{http://purl.org/dc/elements/1.1/}creator",
                    "lastModifiedBy": (
                        "{http://schemas.openxmlformats.org/package/2006/metadata/"
                        "core-properties}lastModifiedBy"
                    ),
                }
                for label, tag in metadata_fields.items():
                    element = root.find(tag)
                    if element is not None and (element.text or "").strip():
                        report.error(
                            "office-personal-metadata",
                            relative,
                            f"docProps/core.xml retains non-empty {label}",
                        )
                for label in ("created", "modified"):
                    tag = f"{{http://purl.org/dc/terms/}}{label}"
                    element = root.find(tag)
                    if element is not None and element.text != "2000-01-01T00:00:00Z":
                        report.error(
                            "office-volatile-timestamp",
                            relative,
                            f"{label} is not the frozen public timestamp",
                        )

            app_name = "docProps/app.xml"
            if app_name in names:
                root = ET.fromstring(archive.read(app_name))
                namespace = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
                for label in ("Company", "Manager", "HyperlinkBase"):
                    element = root.find(f"{{{namespace}}}{label}")
                    if element is not None and (element.text or "").strip():
                        report.error(
                            "office-personal-metadata",
                            relative,
                            f"docProps/app.xml retains non-empty {label}",
                        )

            custom_name = "docProps/custom.xml"
            if custom_name in names:
                root = ET.fromstring(archive.read(custom_name))
                if list(root):
                    report.error(
                        "office-custom-metadata",
                        relative,
                        "custom document properties have not been removed",
                    )
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        report.error("office-container", relative, f"could not audit OOXML: {exc}")
        return

    combined = "\n".join(text_parts)
    unix_roots = ("home", "Users", "wind" + "lidar", "data" + "01", "mnt")
    host_path = re.compile(
        r"/(?:(?:" + "|".join(unix_roots) + r"))/[A-Za-z0-9_.~-]+"
        r"(?:/[A-Za-z0-9_.~+@%=-]+)+"
    )
    windows_path = re.compile(r"[A-Za-z]:\\(?:Users|Documents and Settings)\\")
    if host_path.search(combined) or windows_path.search(combined) or ("file" + "://") in combined:
        report.error(
            "office-absolute-host-path",
            relative,
            "OOXML internals contain a machine-specific path",
        )
    credential_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{24,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(
            r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)"
            r"\s*[:=]\s*['\"][^'\"]{12,}['\"]"
        ),
    )
    if any(pattern.search(combined) for pattern in credential_patterns):
        report.error("office-secret-content", relative, "OOXML internals contain a credential-like value")


def audit_tree(root: Path, release_ready: bool = False) -> Report:
    root = root.resolve()
    report = Report()
    if not root.is_dir():
        report.error("root", root, "release root does not exist")
        return report

    contract_path = root / "configs" / "scientific_contract_4km10min.json"
    registry_path = root / "configs" / "route_registry.json"
    contract = _read_json(contract_path, report)
    registry = _read_json(registry_path, report)
    if contract is not None:
        report.issues.extend(validate_scientific_contract(contract, "configs/scientific_contract_4km10min.json").issues)
    if registry is not None:
        report.issues.extend(validate_route_registry(registry, "configs/route_registry.json").issues)

    _check_release_gates(root, report, release_ready)
    _check_artifact_manifest(root, report)

    for path in _iter_files(root):
        relative = _relative(path, root)
        if path.is_symlink():
            report.error("symlink", relative, "release trees must not contain symlinks")
            continue
        if any(part in CACHE_DIR_NAMES for part in path.parts):
            report.error("cache", relative, "cache artifacts must not be released")
        if any(
            part in GENERATED_DIR_NAMES or part.endswith(".egg-info")
            for part in path.parts
        ):
            report.error(
                "generated-build-artifact",
                relative,
                "build, dist and *.egg-info directories must not be released",
            )
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if lower_name in FORBIDDEN_SECRET_FILENAMES or suffix in {".pem", ".key"}:
            report.error("secret-file", relative, "credential-like file is forbidden")
        if suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
            report.error("binary-artifact", relative, "model/data archives are not part of the source release")
        try:
            if path.stat().st_size > 25 * 1024 * 1024:
                report.error("large-file", relative, "file exceeds the 25 MiB source-release limit")
        except OSError as exc:
            report.error("stat", relative, f"could not stat file: {exc}")
            continue
        if suffix == ".ipynb":
            _check_notebook(path, relative, report)
        if suffix in {".pptx", ".xlsx"}:
            _check_office_container(path, relative, report)
        if suffix in TEXT_SUFFIXES or path.name in {
            "LICENSE",
            "CITATION.cff",
            "DATA_REDISTRIBUTION_DECISION.md",
        }:
            _check_text(path, relative, report)

    return report


def _print_report(report: Report, release_ready: bool) -> None:
    for issue in report.issues:
        print(f"{issue.level:7s} {issue.code:24s} {issue.path}: {issue.message}")
    mode = "release-ready" if release_ready else "staging"
    print(f"audit mode={mode}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--release-ready",
        action="store_true",
        help="treat license, citation, and data-redistribution gates as errors",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = audit_tree(args.root, release_ready=args.release_ready)
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "release-ready" if args.release_ready else "staging",
                    "errors": [item.__dict__ for item in report.errors],
                    "warnings": [item.__dict__ for item in report.warnings],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_report(report, args.release_ready)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
