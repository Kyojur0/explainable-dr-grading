"""Stage 8: Grad-CAM for the two classifiers (brief 12.1).

Grad-CAM (Selvaraju et al., 2017) weights each channel of a convolutional feature map by
the spatial mean of the gradient of the class score with respect to that map, sums the
weighted channels and keeps the positive part. The gradient is captured with a tensor
hook on the layer output, not register_full_backward_hook, which misbehaves with
in-place activations.

Maps are returned unnormalised. normalise() and upsample() are applied by the caller
when a map is drawn or compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from drgrade.io import GRADE_VALUES

NUM_CLASSES = len(GRADE_VALUES)  # DR grades 0 to 4 (brief 2)


@dataclass(frozen=True)
class GradCAMResult:
    """Grad-CAM output for one batch.

    cams: (N, h, w) float64, ReLU applied, not normalised.
    logits: (N, NUM_CLASSES) float64 model logits for the batch.
    target: (N,) int64, the class each map explains.
    """

    cams: np.ndarray
    logits: np.ndarray
    target: np.ndarray


class GradCAM:
    """Grad-CAM on one target layer of one model.

    Use as a context manager, or call remove() when finished, so the forward hook is
    not left on the model. The model must be in eval mode. Callers must not call the
    object inside torch.inference_mode(): the backward pass needs the autograd graph.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle: RemovableHandle | None = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Keep the layer output and ask autograd to hand back its gradient."""
        self._activations = output
        if output.requires_grad:
            output.register_hook(self._on_gradient)

    def _on_gradient(self, grad: torch.Tensor) -> None:
        """Store the gradient of the score with respect to the layer output."""
        self._gradient = grad

    def remove(self) -> None:
        """Remove the forward hook. Safe to call more than once."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove()

    @staticmethod
    def _requested_targets(target_class: int | Sequence[int] | None, batch_size: int) -> np.ndarray | None:
        """Validate target_class. Returns None when the predicted class should be explained."""
        if target_class is None:
            return None
        if isinstance(target_class, (int, np.integer)):
            targets = np.full(batch_size, int(target_class), dtype=np.int64)
        else:
            targets = np.asarray(list(target_class), dtype=np.int64)
            if targets.shape != (batch_size,):
                raise ValueError(f"target_class has {targets.size} entries for a batch of {batch_size}")
        if targets.size and (targets.min() < 0 or targets.max() >= NUM_CLASSES):
            raise ValueError(f"target classes must be in 0..{NUM_CLASSES - 1}, got {targets.tolist()}")
        return targets

    @staticmethod
    def _cam(activations: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
        """Return ReLU(sum_k alpha_k * A_k), with alpha_k the spatial mean of the gradient."""
        acts = activations.detach().to(device="cpu", dtype=torch.float64)
        grads = gradient.detach().to(device="cpu", dtype=torch.float64)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # alpha_k, shape (N, K, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1))  # shape (N, h, w)
        return np.ascontiguousarray(cam.numpy())

    def __call__(
        self, images: torch.Tensor, target_class: int | Sequence[int] | None = None
    ) -> GradCAMResult:
        """Return Grad-CAM maps for a batch of normalised images, shape (N, 3, H, W).

        target_class: None explains the predicted class of each image. An int explains
        that class for every image. A sequence gives one class per image.

        Do not call this inside torch.inference_mode(); the gradient needs autograd.
        """
        if self.model.training:
            raise ValueError("Grad-CAM needs eval mode (call model.eval()); BatchNorm and dropout would couple the batch")
        if images.ndim != 4:
            raise ValueError(f"images must be (N, C, H, W), got shape {tuple(images.shape)}")
        requested = self._requested_targets(target_class, images.shape[0])
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(images)
                if requested is None:
                    target = logits.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
                else:
                    target = requested
                # One backward pass over the summed scores gives each image its own gradient.
                # In eval mode image i's score depends only on image i, so d(sum)/dA_i equals
                # d(score_i)/dA_i.
                index = torch.as_tensor(target, device=logits.device)
                score = logits.gather(1, index[:, None]).sum()
                score.backward()
            if self._activations is None or self._gradient is None:
                raise RuntimeError("the target layer was not reached in the forward pass")
            cams = self._cam(self._activations, self._gradient)
            logits_np = logits.detach().to(dtype=torch.float64, device="cpu").numpy()
            return GradCAMResult(cams=cams, logits=logits_np, target=target)
        finally:
            # Drop the stored tensors and the parameter gradients left by backward().
            self._activations = None
            self._gradient = None
            self.model.zero_grad(set_to_none=True)


def upsample(cams: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize (N, h, w) maps to (N, size, size), float64, clipped at zero."""
    if cams.ndim != 3:
        raise ValueError(f"cams must be (N, h, w), got shape {cams.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(cams, dtype=np.float64))[:, None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return np.clip(resized[:, 0].numpy(), 0.0, None)


def fundus_mask(size: int) -> np.ndarray:
    """Boolean (size, size) mask of the inscribed circle kept by apply_circle_mask.

    A pixel is inside when (y - c)^2 + (x - c)^2 <= (size / 2)^2 with c = (size - 1) / 2,
    the same rule as drgrade.preprocess.apply_circle_mask.
    """
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    return (yy - centre) ** 2 + (xx - centre) ** 2 <= (size / 2.0) ** 2


def normalise(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale one (H, W) map to [0, 1] by its maximum inside the mask; zero outside.

    Dividing by the maximum, not min-max scaling, keeps zero meaning "no positive
    evidence", because the map is already ReLU'd. An all-zero map stays all zero.
    """
    if cam.shape != mask.shape:
        raise ValueError(f"cam {cam.shape} and mask {mask.shape} must have the same shape")
    out = np.zeros(cam.shape, dtype=np.float64)
    peak = float(cam[mask].max()) if mask.any() else 0.0
    if peak > 0.0:
        out = np.where(mask, cam / peak, 0.0)
    return out
