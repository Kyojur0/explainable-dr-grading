"""Stage 4: model construction through timm (brief 8.1).

Plain timm construction, so that the Grad-CAM layer names (stage 8) stay valid:
EfficientNet-B0 exposes bn2 and ResNet-18 exposes layer4.
"""

from __future__ import annotations

from torch import nn

from drgrade.io import GRADE_VALUES
from drgrade.run_config import RunConfig

NUM_CLASSES = len(GRADE_VALUES)  # DR grades 0 to 4 (brief 2)
EFFICIENTNET_PREFIX = "efficientnet"
RESNET_PREFIX = "resnet"


def build_model(cfg: RunConfig, pretrained: bool) -> nn.Module:
    """Create the network named by cfg.model.timm_pretrained_tag with a 5-way head.

    The head has dropout cfg.model.dropout before the linear classifier (timm's
    drop_rate). With pretrained=False the same architecture is built with random
    weights, which is what the offline smoke test uses. Imports timm lazily.
    """
    import timm

    return timm.create_model(
        cfg.model.timm_pretrained_tag,
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
        drop_rate=cfg.model.dropout,
    )


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the layer Grad-CAM reads (brief 12.1).

    arch is the timm architecture name, for example 'efficientnet_b0' or 'resnet18'.

    EfficientNet uses bn2, the BatchNorm plus SiLU that timm applies after conv_head.
    Its output is exactly the feature map that global average pooling passes to the
    linear classifier, so Grad-CAM at this layer equals CAM (Zhou et al., 2016): the map
    is the classifier-weighted sum of the final feature maps, scaled by 1/(h*w).
    conv_head's output is before BatchNorm and SiLU, so that equivalence would not hold
    there. ResNet uses the last block of layer4, whose output is post-ReLU and feeds the
    pooling directly.
    """
    if arch.startswith(EFFICIENTNET_PREFIX):
        layer = getattr(model, "bn2", None)
        if not isinstance(layer, nn.Module):
            raise ValueError(f"{arch}: expected a bn2 module, got {type(layer).__name__}")
        return layer
    if arch.startswith(RESNET_PREFIX):
        layer4 = getattr(model, "layer4", None)
        if not isinstance(layer4, nn.Sequential) or len(layer4) == 0:
            raise ValueError(f"{arch}: expected a non-empty layer4 Sequential")
        return layer4[-1]
    raise ValueError(f"no Grad-CAM target layer is defined for architecture {arch!r}")
