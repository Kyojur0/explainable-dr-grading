"""Tests for model construction (architectures only; no pretrained weights are downloaded)."""

from __future__ import annotations

import torch
from torch import nn

from drgrade.models import build_model, gradcam_target_layer
from drgrade.run_config import (
    AugConfig,
    DataSection,
    ModelConfig,
    OptimiserConfig,
    RunConfig,
    ScheduleConfig,
)

INPUT = 64


def _run(timm_name: str, tag: str) -> RunConfig:
    aug = AugConfig(180, 0.5, 0.5, 0.85, 1.0, 0.95, 1.05, 0.2, 0.2, 0.8, 1.2)
    return RunConfig(
        run_name="t",
        model=ModelConfig("M", timm_name, tag, INPUT, 0.2),
        loss="ce",
        label_smoothing=0.0,
        optimiser=OptimiserConfig("adamw", 3e-4, 1e-4),
        schedule=ScheduleConfig("cosine", 1),
        max_epochs=2,
        early_stopping_patience=5,
        batch_size=4,
        accumulation_steps=1,
        grad_clip_norm=1.0,
        selection_metric="qwk",
        seed=42,
        num_workers=0,
        aug=aug,
        data=DataSection("splits/split_v1.csv"),
    )


def test_both_architectures_build_and_output_five_logits() -> None:
    """ResNet-18 and EfficientNet-B0 build without weights and map (B, 3, H, W) to (B, 5)."""
    torch.manual_seed(0)
    for timm_name, tag in (("resnet18", "resnet18.a1_in1k"), ("efficientnet_b0", "efficientnet_b0.ra_in1k")):
        model = build_model(_run(timm_name, tag), pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, INPUT, INPUT))
        assert logits.shape == (2, 5)
        assert model.get_classifier().out_features == 5
        assert model.drop_rate == 0.2


def test_gradcam_target_layers() -> None:
    """EfficientNet-B0 targets bn2 (the layer after conv_head); ResNet-18 targets the last block of layer4."""
    effnet = build_model(_run("efficientnet_b0", "efficientnet_b0.ra_in1k"), pretrained=False)
    layer = gradcam_target_layer(effnet, "efficientnet_b0")
    assert layer is effnet.bn2
    assert isinstance(layer, nn.Module)

    resnet = build_model(_run("resnet18", "resnet18.a1_in1k"), pretrained=False)
    block = gradcam_target_layer(resnet, "resnet18")
    assert block is resnet.layer4[-1]
    assert type(block).__name__ == "BasicBlock"


def test_gradcam_target_layer_rejects_unknown_architecture() -> None:
    """An architecture without a defined Grad-CAM layer raises ValueError."""
    import pytest

    effnet = build_model(_run("efficientnet_b0", "efficientnet_b0.ra_in1k"), pretrained=False)
    with pytest.raises(ValueError):
        gradcam_target_layer(effnet, "vit_small")


def test_mismatched_tag_is_rejected() -> None:
    """A pretrained tag that does not belong to the architecture name is refused at config load."""
    import pytest

    from drgrade.run_config import RunConfigError

    with pytest.raises(RunConfigError):
        _run("resnet18", "efficientnet_b0.ra_in1k")
