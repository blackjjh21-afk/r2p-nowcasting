from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules while the class body is evaluated.
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module("release_audit", ROOT / "scripts" / "audit_release.py")
CONTRACT = json.loads(
    (ROOT / "configs" / "scientific_contract_4km10min.json").read_text(encoding="utf-8")
)
REGISTRY = json.loads(
    (ROOT / "configs" / "route_registry.json").read_text(encoding="utf-8")
)


def test_frozen_scientific_contract_passes() -> None:
    report = AUDIT.validate_scientific_contract(CONTRACT)
    assert report.errors == []


def test_contract_rejects_temporal_and_station_drift() -> None:
    value = copy.deepcopy(CONTRACT)
    value["radar_input"]["relative_minutes"] = [-50, -40, -30, -20, -10, 0]
    value["station_support"]["heldout_station_count"] = 127
    report = AUDIT.validate_scientific_contract(value)
    codes = {item.code for item in report.errors}
    assert "radar-value" in codes
    assert "stations-value" in codes
    assert "stations-sum" in codes


def test_frozen_route_registry_passes() -> None:
    report = AUDIT.validate_route_registry(REGISTRY)
    assert report.errors == []


def test_contract_rejects_cnn_loss_or_replication_drift() -> None:
    value = copy.deepcopy(CONTRACT)
    value["cnn_readout"]["loss"] = "another objective"
    value["cnn_readout"]["exprecast_upstream_members"] = 3
    report = AUDIT.validate_scientific_contract(value)
    assert len([issue for issue in report.errors if issue.code == "cnn-contract"]) == 2


def test_radar_only_requires_matched_checkpoint_truth_axes_and_mask() -> None:
    value = copy.deepcopy(REGISTRY)
    radar = next(
        route for route in value["routes"] if route["route_id"] == "direct_r2p_radar_only"
    )
    radar["same_checkpoint_within_seed"] = False
    radar["same_truth"] = False
    radar["same_axes"] = ["issue_time", "lead"]
    radar["masked_inputs"] = ["heldout_target_history"]
    report = AUDIT.validate_route_registry(value)
    messages = "\n".join(item.message for item in report.errors)
    assert "same_checkpoint_within_seed" in messages
    assert "same_truth" in messages
    assert "same_axes" in messages
    assert "masked_inputs" in messages


def test_no_station_dropout_requires_matched_training_control() -> None:
    value = copy.deepcopy(REGISTRY)
    control = next(
        route
        for route in value["routes"]
        if route["route_id"] == "direct_r2p_no_station_dropout"
    )
    control["same_architecture"] = False
    control["same_objective_and_optimizer"] = False
    report = AUDIT.validate_route_registry(value)
    messages = "\n".join(item.message for item in report.errors)
    assert "same_architecture" in messages
    assert "same_objective_and_optimizer" in messages


def _write_minimal_candidate(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "scientific_contract_4km10min.json").write_text(
        json.dumps(CONTRACT), encoding="utf-8"
    )
    (configs / "route_registry.json").write_text(json.dumps(REGISTRY), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("synthetic audit candidate\n", encoding="utf-8")
    workflow_paths = (
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
    for relative in workflow_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "{}\n" if path.suffix == ".json" else "synthetic workflow fixture\n",
            encoding="utf-8",
        )
    manifested = ("README.md", *workflow_paths)
    rows = []
    for relative in manifested:
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        rows.append(f"{relative},{digest},synthetic test fixture")
    (configs / "artifact_manifest.csv").write_text(
        "path,sha256,role\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_legal_gates_warn_in_staging_and_fail_release_ready(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    staging = AUDIT.audit_tree(tmp_path, release_ready=False)
    assert staging.errors == []
    assert {item.path for item in staging.warnings} == {
        "LICENSE",
        "CITATION.cff",
        "DATA_REDISTRIBUTION_DECISION.md",
    }

    release = AUDIT.audit_tree(tmp_path, release_ready=True)
    assert release.warnings == []
    assert {item.path for item in release.errors} == {
        "LICENSE",
        "CITATION.cff",
        "DATA_REDISTRIBUTION_DECISION.md",
    }


def test_release_gates_accept_explicit_approval(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "LICENSE").write_text(
        "Synthetic License\n\nCopyright (c) 2026 Synthetic Authors\n\n"
        "Permission is granted to use, copy, modify, and distribute this "
        "synthetic fixture for testing. This text is not installed in the "
        "real release candidate.\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: Cite this software.\n"
        "type: software\n"
        "title: Synthetic release\n"
        "authors:\n"
        "  - family-names: Author\n"
        "    given-names: Synthetic\n"
        "repository-code: https://github.com/synthetic/release\n"
        "version: 1.0.0\n"
        "date-released: 2026-08-28\n"
        "doi: 10.0000/synthetic.release\n",
        encoding="utf-8",
    )
    (tmp_path / "DATA_REDISTRIBUTION_DECISION.md").write_text(
        "repository_decision: public_redistribution_excluded\n"
        "supplementary_data_1_decision: derived_station_resolved_archive_approved\n"
        "supplementary_data_1_contents: Distance_station_groups (station_id, "
        "nearest-fitting-station distance, distance group); FigS2_station_timeseries and "
        "FigS2_timeseries_summary (station_id, exact valid times, gauge truth, route "
        "predictions, member ranges); FigS3_station_members, FigS3_station_CSI and "
        "FigS3_pairwise_counts (station_id, name, lat, lon, member contingency counts "
        "and CSI, mean CSI, pairwise station tallies)\n"
        "supplementary_data_1_destination: journal supplement and data archive\n"
        "approved_by: Synthetic Author\n"
        "date: 2026-08-28\n"
        "source_product: Synthetic KMA product\n"
        "source_url: https://data.go.kr/kma-product\n"
        "accessed: 2026-08-28\n"
        "terms_reference: Synthetic provider terms reviewed\n",
        encoding="utf-8",
    )
    report = AUDIT.audit_tree(tmp_path, release_ready=True)
    assert report.errors == []
    assert report.warnings == []


def test_release_gate_accepts_minimal_redistribution_decision(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "LICENSE").write_text(
        "Synthetic License\n\nCopyright (c) 2026 Synthetic Authors\n\n"
        "Permission is granted to use, copy, modify, and distribute this "
        "synthetic fixture for testing. This text is not installed in the "
        "real release candidate.\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: Cite this software.\n"
        "type: software\n"
        "title: Synthetic release\n"
        "authors:\n"
        "  - family-names: Author\n"
        "    given-names: Synthetic\n"
        "repository-code: https://github.com/synthetic/release\n"
        "version: 1.0.0\n"
        "date-released: 2026-08-28\n"
        "doi: 10.0000/synthetic.release\n",
        encoding="utf-8",
    )
    (tmp_path / "DATA_REDISTRIBUTION_DECISION.md").write_text(
        "repository_decision: minimal_redistribution_approved\n"
        "supplementary_data_1_decision: derived_station_resolved_archive_approved\n"
        "supplementary_data_1_contents: Distance_station_groups (station_id, "
        "nearest-fitting-station distance, distance group); FigS2_station_timeseries and "
        "FigS2_timeseries_summary (station_id, exact valid times, gauge truth, route "
        "predictions, member ranges); FigS3_station_members, FigS3_station_CSI and "
        "FigS3_pairwise_counts (station_id, name, lat, lon, member contingency counts "
        "and CSI, mean CSI, pairwise station tallies)\n"
        "supplementary_data_1_destination: journal supplement and data archive\n"
        "approved_by: Synthetic Author\n"
        "date: 2026-08-28\n"
        "source_product: Synthetic KMA product\n"
        "source_url: https://data.go.kr/kma-product\n"
        "accessed: 2026-08-28\n"
        "terms_reference: Synthetic provider terms reviewed\n",
        encoding="utf-8",
    )
    report = AUDIT.audit_tree(tmp_path, release_ready=True)
    assert report.errors == []


def test_release_gate_rejects_unfilled_templates(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "LICENSE").write_text(
        "THIS FILE IS A TEMPLATE AND IS NOT A LICENSE GRANT.\n"
        "Approved license: [REQUIRED: CHOOSE WITH AUTHORS]\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "title: '[REQUIRED: TITLE]'\n"
        "authors:\n"
        "  - family-names: '[REQUIRED: FAMILY NAME]'\n",
        encoding="utf-8",
    )
    (tmp_path / "DATA_REDISTRIBUTION_DECISION.md").write_text(
        "repository_decision: public_redistribution_excluded\n"
        "supplementary_data_1_decision: [REQUIRED: APPROVAL]\n"
        "approved_by: [REQUIRED: AUTHOR]\n"
        "date: [REQUIRED: YYYY-MM-DD]\n",
        encoding="utf-8",
    )
    report = AUDIT.audit_tree(tmp_path, release_ready=True)
    assert {item.path for item in report.errors} == {
        "LICENSE",
        "CITATION.cff",
        "DATA_REDISTRIBUTION_DECISION.md",
    }


def test_release_gate_rejects_legacy_single_scope_data_decision(
    tmp_path: Path,
) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "LICENSE").write_text(
        "Synthetic License\n\nCopyright (c) 2026 Synthetic Authors\n\n"
        "Permission is granted to use, copy, modify, and distribute this "
        "synthetic fixture for testing. This text is not installed in the "
        "real release candidate.\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: Cite this software.\n"
        "type: software\n"
        "title: Synthetic release\n"
        "authors:\n"
        "  - family-names: Author\n"
        "    given-names: Synthetic\n"
        "repository-code: https://github.com/synthetic/release\n"
        "version: 1.0.0\n"
        "date-released: 2026-08-28\n"
        "doi: 10.0000/synthetic.release\n",
        encoding="utf-8",
    )
    (tmp_path / "DATA_REDISTRIBUTION_DECISION.md").write_text(
        "decision: public_redistribution_excluded\n"
        "approved_by: Synthetic Author\n"
        "date: 2026-08-28\n"
        "source_product: Synthetic KMA product\n"
        "source_url: https://data.go.kr/kma-product\n"
        "accessed: 2026-08-28\n"
        "terms_reference: Synthetic provider terms reviewed\n",
        encoding="utf-8",
    )

    report = AUDIT.audit_tree(tmp_path, release_ready=True)
    assert any(
        item.code == "release-gate"
        and item.path == "DATA_REDISTRIBUTION_DECISION.md"
        for item in report.errors
    )


def test_release_gate_requires_exact_supplementary_data_scope(
    tmp_path: Path,
) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "DATA_REDISTRIBUTION_DECISION.md").write_text(
        "repository_decision: public_redistribution_excluded\n"
        "supplementary_data_1_decision: derived_station_resolved_archive_approved\n"
        "supplementary_data_1_contents: case RN60 time series; stationwise metrics\n"
        "supplementary_data_1_destination: journal supplement and data archive\n"
        "approved_by: Synthetic Author\n"
        "date: 2026-08-28\n"
        "source_product: Synthetic KMA product\n"
        "source_url: https://data.go.kr/kma-product\n"
        "accessed: 2026-08-28\n"
        "terms_reference: Synthetic provider terms reviewed\n",
        encoding="utf-8",
    )

    report = AUDIT.Report()
    AUDIT._check_release_gates(tmp_path, report, release_ready=True)
    assert any(
        item.code == "release-gate"
        and item.path == "DATA_REDISTRIBUTION_DECISION.md"
        for item in report.errors
    )


def test_artifact_manifest_detects_content_drift(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "README.md").write_text("changed after freeze\n", encoding="utf-8")
    report = AUDIT.audit_tree(tmp_path, release_ready=False)
    assert "artifact-hash" in {item.code for item in report.errors}


def test_artifact_manifest_requires_public_result_coverage(tmp_path: Path) -> None:
    _write_minimal_candidate(tmp_path)
    figure = tmp_path / "figures" / "unlisted.png"
    figure.parent.mkdir()
    figure.write_bytes(b"synthetic")
    report = AUDIT.audit_tree(tmp_path, release_ready=False)
    assert "artifact-unmanifested" in {item.code for item in report.errors}


def test_audit_rejects_host_paths_archives_caches_and_dirty_notebooks(
    tmp_path: Path,
) -> None:
    _write_minimal_candidate(tmp_path)
    (tmp_path / "README.md").write_text(
        "private=" + "/" + "home/example/private/data\n", encoding="utf-8"
    )
    (tmp_path / ("weights" + ".pt")).write_bytes(b"synthetic")
    cache = tmp_path / ("__" + "pycache__")
    cache.mkdir()
    (cache / "generated.pyc").write_bytes(b"synthetic")
    egg_info = tmp_path / "src" / "synthetic.egg-info"
    egg_info.mkdir(parents=True)
    (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["x"]}],
                "source": ["print('x')"],
            }
        ],
    }
    (tmp_path / "dirty.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    report = AUDIT.audit_tree(tmp_path, release_ready=False)
    codes = {item.code for item in report.errors}
    assert "absolute-host-path" in codes
    assert "binary-artifact" in codes
    assert "cache" in codes
    assert "generated-build-artifact" in codes
    assert "notebook-execution" in codes
    assert "notebook-output" in codes


def test_office_audit_scans_nested_metadata_paths_and_secrets(
    tmp_path: Path,
) -> None:
    office = tmp_path / "unsafe.pptx"
    private_path = "/" + "home" + "/example/private/source"
    core = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/">'
        '<dc:creator>Private Author</dc:creator>'
        '<cp:lastModifiedBy>Private Editor</cp:lastModifiedBy>'
        '<dcterms:created>2026-01-01T00:00:00Z</dcterms:created>'
        '</cp:coreProperties>'
    )
    credential_label = "pass" + "word"
    slide = (
        '<slide><text>'
        + private_path
        + f'</text><config {credential_label}="this-is-not-public-123"/></slide>'
    )
    with zipfile.ZipFile(office, "w") as archive:
        archive.writestr("docProps/core.xml", core)
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/notesSlides/notesSlide1.xml", "<notes>hidden</notes>")

    report = AUDIT.Report()
    AUDIT._check_office_container(office, "unsafe.pptx", report)
    codes = {item.code for item in report.errors}
    assert "office-personal-metadata" in codes
    assert "office-volatile-timestamp" in codes
    assert "office-absolute-host-path" in codes
    assert "office-secret-content" in codes
    assert "office-hidden-content" in codes


def test_office_audit_rejects_external_relationships(tmp_path: Path) -> None:
    office = tmp_path / "external.xlsx"
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="hyperlink" Target="https://example.invalid" '
        'TargetMode="External"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(office, "w") as archive:
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)

    report = AUDIT.Report()
    AUDIT._check_office_container(office, "external.xlsx", report)
    assert "office-external-relationship" in {
        item.code for item in report.errors
    }
