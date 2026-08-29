from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patch_mlp import PatchMLP, importance_corrected_smooth_l1  # noqa: E402


def test_context_patch_mlp_shape_and_backward() -> None:
    model = PatchMLP()
    patches = torch.rand(8, 6, 3, 3)
    auxiliary = torch.rand(8, 39)
    prediction = model(patches, auxiliary)
    assert prediction.shape == (8,)
    assert torch.all(prediction >= 0)
    prediction.mean().backward()


def test_importance_correction_reduces_to_plain_mean_when_p_equals_q() -> None:
    prediction = torch.tensor([0.0, 0.4, 0.8, 1.2, 2.0])
    target_mm = torch.tensor([0.0, 2.0, 7.0, 15.0, 25.0])
    target_log = torch.log1p(target_mm)
    fractions = torch.full((5,), 0.2)
    corrected = importance_corrected_smooth_l1(
        prediction,
        target_log,
        target_mm,
        natural_fractions=fractions,
        sampled_fractions=fractions,
    )
    expected = torch.nn.functional.smooth_l1_loss(
        prediction, target_log, beta=1.0, reduction="mean"
    )
    torch.testing.assert_close(corrected, expected)
