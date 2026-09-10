"""The source distribution contains the supported manuscript workflows."""

import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_obsolete_standalone_figures_are_not_bundled() -> None:
    for stem in (
        "fig4a_radar_only_same_checkpoint_4km10min",
        "fig4b_station_dropout_training_ablation",
    ):
        for suffix in (".png", ".pdf"):
            assert not (ROOT / "figures" / f"{stem}{suffix}").exists()
    assert not (ROOT / "figures" / "main").exists()


def test_example_audit_outputs_resolve_and_match_hashes() -> None:
    for name in ("radar_only", "station_dropout"):
        manifest = json.loads(
            (ROOT / "data/examples" / f"{name}_audit_manifest.json").read_text()
        )
        outputs = manifest["outputs"]
        assert outputs, name
        for relative, expected_sha256 in outputs.items():
            candidate = Path(relative)
            assert not candidate.is_absolute() and ".." not in candidate.parts
            path = ROOT / candidate
            assert path.is_file(), relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, relative


def test_author_release_preparation_is_not_distributed() -> None:
    for relative in (
        "scripts/prepare_cnn_release_artifacts.py",
        "scripts/sync_public_paper_outputs.py",
        "scripts/refresh_artifact_manifest.py",
        "scripts/audit_release.py",
        "scripts/scrub_office_metadata.py",
        "scripts/finalize_fig6_architecture.py",
        "tests/test_release_audit.py",
        "docs/PUBLICATION_WORKFLOW.md",
        "docs/ZENODO_SOFTWARE_FORM_VALUES.md",
    ):
        assert not (ROOT / relative).exists(), relative
    assert not list((ROOT / "docs/templates").glob("*.in"))


def test_scientific_package_has_no_office_documents_or_dependencies() -> None:
    for directory in ("data", "figures", "src", "tests"):
        assert not [
            path for path in (ROOT / directory).rglob("*")
            if path.suffix.lower() in {".xlsx", ".xls", ".pptx", ".docx", ".doc"}
        ]
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = metadata["project"]["dependencies"]
    assert not any(
        dependency.startswith(("openpyxl", "python-docx", "python-pptx", "xlsxwriter"))
        for dependency in dependencies
    )


def test_distributed_packages_match_manuscript_workflows() -> None:
    packages = {
        path.parent.name for path in (ROOT / "src").glob("*/__init__.py")
    }
    assert packages == {
        "cnn_readout", "common", "evaluation", "exprecast_adapter", "exprecast_adapted",
        "paper_outputs", "preprocessing", "pysteps_adapter", "r2p_4km10min",
    }


def test_console_commands_match_supported_workflows() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == "1.1.0"
    assert metadata["project"]["scripts"] == {
        "r2p-4km10min": "r2p_4km10min.run_vanilla_r2p:main",
        "r2p-radar-only": "evaluation.radar_only_comparison:main",
        "r2p-station-dropout": "evaluation.station_dropout_comparison:main",
        "prepare-kma-hsr-4km10min": "preprocessing.prepare_kma_hsr_4km10min:main",
        "pysteps-4km10min": "pysteps_adapter.cli:main",
        "cnn-readout-4km10min": "cnn_readout.workflow:main",
        "exprecast-export-adapter": "exprecast_adapter.adapter:main",
        "exprecast-adapted": "exprecast_adapted.workflow:main",
        "exprecast-select-rn60": "exprecast_adapted.selection:main",
        "r2p-paper-figures": "paper_outputs.render_publication:main",
        "r2p-restricted-figures": "paper_outputs.render_restricted:main",
    }
