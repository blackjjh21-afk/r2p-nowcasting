"""The source distribution contains the supported manuscript workflows."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_author_release_preparation_is_not_distributed() -> None:
    for relative in (
        "scripts/prepare_cnn_release_artifacts.py",
        "scripts/sync_public_paper_outputs.py",
        "scripts/refresh_artifact_manifest.py",
        "docs/PUBLICATION_WORKFLOW.md",
        "docs/ZENODO_SOFTWARE_FORM_VALUES.md",
    ):
        assert not (ROOT / relative).exists(), relative
    assert not list((ROOT / "docs/templates").glob("*.in"))


def test_distributed_packages_match_manuscript_workflows() -> None:
    packages = {
        path.parent.name for path in (ROOT / "src").glob("*/__init__.py")
    }
    assert packages == {
        "cnn_readout", "common", "evaluation", "exprecast_adapter",
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
        "r2p-paper-figures": "paper_outputs.render_publication:main",
        "r2p-restricted-figures": "paper_outputs.render_restricted:main",
    }
