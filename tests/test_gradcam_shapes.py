"""Tests for Grad-CAM (brief 12.1): shapes, CAM equivalence, hygiene and helpers.

Models are randomly initialised (pretrained=False) at 128 px, so the suite runs on CPU
in seconds. The key check is that Grad-CAM at bn2 (EfficientNet) or layer4[-1] (ResNet)
reproduces the classic CAM formula from the classifier weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import timm
import torch
from PIL import Image
from torch import nn

from drgrade.gradcam import GradCAM, fundus_mask, normalise, upsample
from drgrade.models import gradcam_target_layer
from drgrade.preprocess import apply_circle_mask

ARCHITECTURES = ("efficientnet_b0", "resnet18")
INPUT = 128
BATCH = 3
RTOL = 1e-4
ATOL = 1e-7


def _build(arch: str) -> nn.Module:
    """A randomly initialised 5-class model in eval mode."""
    torch.manual_seed(0)
    model = timm.create_model(arch, pretrained=False, num_classes=5)
    model.eval()
    return model


def _images() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, 3, INPUT, INPUT)


def _classifier_weight(model: nn.Module, arch: str) -> np.ndarray:
    """The linear classifier weight, (5, K), as float64."""
    linear = model.classifier if arch.startswith("efficientnet") else model.fc
    return linear.weight.detach().double().cpu().numpy()


def _capture_layer_output(layer: nn.Module) -> tuple[dict, torch.utils.hooks.RemovableHandle]:
    """Register our own forward hook that records the layer output A (N, K, h, w)."""
    captured: dict[str, np.ndarray] = {}

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured["A"] = output.detach().double().cpu().numpy()

    return captured, layer.register_forward_hook(hook)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_shapes_dtypes_and_default_target(arch: str) -> None:
    """Maps and logits have the documented shapes and dtypes; the default target is the argmax."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    images = _images()
    captured, handle = _capture_layer_output(layer)
    with GradCAM(model, layer) as cam:
        result = cam(images)
    handle.remove()
    with torch.no_grad():
        reference = model(images).double().numpy()

    h, w = captured["A"].shape[2:]
    assert result.cams.shape == (BATCH, h, w)
    assert result.cams.dtype == np.float64
    assert result.logits.shape == (BATCH, 5)
    assert result.logits.dtype == np.float64
    assert result.target.shape == (BATCH,)
    assert result.target.dtype == np.int64
    assert np.all(result.cams >= 0)
    assert np.all(np.isfinite(result.cams))
    assert np.array_equal(result.target, result.logits.argmax(axis=1))
    assert np.allclose(result.logits, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_cam_equivalence_with_classifier_weights(arch: str) -> None:
    """Grad-CAM equals relu(sum_k W[t, k] * A[k]) / (h * w), the CAM formula of Zhou et al. (2016)."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    images = _images()
    captured, handle = _capture_layer_output(layer)
    with GradCAM(model, layer) as cam:
        result = cam(images)
    handle.remove()

    activations = captured["A"]
    weights = _classifier_weight(model, arch)
    _, _, h, w = activations.shape
    expected = np.empty_like(result.cams)
    for i, target in enumerate(result.target):
        expected[i] = np.maximum(np.einsum("k,khw->hw", weights[target], activations[i]), 0.0) / (h * w)

    assert np.any(expected > 0), "random model produced an all-zero CAM; the check would be vacuous"
    assert np.allclose(result.cams, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_batch_independence(arch: str) -> None:
    """Image 0 alone gives the same map as image 0 inside a batch of three (eval mode)."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    images = _images()
    with GradCAM(model, layer) as cam:
        batch = cam(images)
        alone = cam(images[:1], target_class=int(batch.target[0]))
    assert np.allclose(alone.cams[0], batch.cams[0], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_target_class_int_list_and_errors(arch: str) -> None:
    """An int explains one class for all images; a list gives one class per image; bad inputs raise."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    images = _images()
    with GradCAM(model, layer) as cam:
        as_int = cam(images, target_class=2)
        assert np.all(as_int.target == 2)

        as_list = cam(images, target_class=[0, 1, 4])
        assert np.array_equal(as_list.target, np.array([0, 1, 4]))

        with pytest.raises(ValueError):
            cam(images, target_class=[0, 1])
        with pytest.raises(ValueError):
            cam(images, target_class=5)
        with pytest.raises(ValueError):
            cam(images, target_class=[0, 1, 5])


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_hook_removed_on_exit(arch: str) -> None:
    """Leaving the context manager removes the forward hook from the target layer."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    assert len(layer._forward_hooks) == 0
    with GradCAM(model, layer) as cam:
        cam(_images())
        assert len(layer._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_train_mode_is_rejected(arch: str) -> None:
    """Grad-CAM refuses a model in train mode, where BatchNorm couples the batch."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    model.train()
    with GradCAM(model, layer) as cam, pytest.raises(ValueError):
        cam(_images())


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_parameter_gradients_are_cleared(arch: str) -> None:
    """No parameter keeps a .grad after a call."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    with GradCAM(model, layer) as cam:
        cam(_images())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_channels_last_matches_contiguous(arch: str) -> None:
    """A channels_last model with channels_last inputs gives the same maps as the contiguous run."""
    model = _build(arch)
    layer = gradcam_target_layer(model, arch)
    images = _images()
    with GradCAM(model, layer) as cam:
        contiguous = cam(images, target_class=1)

    model = model.to(memory_format=torch.channels_last)
    layer = gradcam_target_layer(model, arch)
    cl_images = images.contiguous(memory_format=torch.channels_last)
    with GradCAM(model, layer) as cam:
        channels_last = cam(cl_images, target_class=1)
    assert np.allclose(channels_last.cams, contiguous.cams, rtol=RTOL, atol=ATOL)


def test_upsample_shape_and_non_negative() -> None:
    """upsample gives (N, size, size), float64, with no negative values."""
    cams = np.random.default_rng(0).normal(size=(3, 4, 4))
    cams[0, 0, 0] = -1.0  # a negative input must not survive the clip
    out = upsample(np.maximum(cams, 0.0), 256)
    assert out.shape == (3, 256, 256)
    assert out.dtype == np.float64
    assert np.all(out >= 0)


def test_fundus_mask_matches_apply_circle_mask() -> None:
    """fundus_mask(64) is exactly the set of pixels that apply_circle_mask keeps."""
    size = 64
    kept = np.asarray(apply_circle_mask(Image.new("RGB", (size, size), (255, 255, 255))))[..., 0] > 0
    assert np.array_equal(fundus_mask(size), kept)


def test_normalise_scales_inside_mask_and_zeros_outside() -> None:
    """normalise gives a maximum of 1 inside the mask and exactly 0 outside; zero maps stay zero."""
    mask = fundus_mask(64)
    cam = np.random.default_rng(1).random((64, 64)) * 3.0
    cam[~mask] = 100.0  # large values outside the mask must be ignored and zeroed
    out = normalise(cam, mask)
    assert out[mask].max() == pytest.approx(1.0)
    assert np.all(out[~mask] == 0.0)
    assert np.all(out >= 0.0)

    zeros = np.zeros((64, 64))
    assert np.all(normalise(zeros, mask) == 0.0)
