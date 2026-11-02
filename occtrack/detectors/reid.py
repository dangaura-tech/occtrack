"""Lightweight appearance embedding using a torchvision ImageNet backbone.

This is a generic baseline; for best results plug in a person Re-ID model (e.g. OSNet,
FastReID) that exposes the same ``__call__(frame, boxes) -> (N, D)`` interface.
"""

from __future__ import annotations

import numpy as np


class TorchvisionEmbedder:
    def __init__(self, arch: str = "resnet18", device: str = "cpu", input_size=(256, 128)) -> None:
        try:
            import torch
            import torchvision
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("TorchvisionEmbedder requires torch and torchvision") from e
        self.torch = torch
        weights = torchvision.models.get_model_weights(arch).DEFAULT
        model = torchvision.models.get_model(arch, weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device
        self.size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def __call__(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        torch = self.torch
        if len(boxes) == 0:
            return np.zeros((0, 512))
        h, w = frame_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes.astype(int):
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
            crop = torch.from_numpy(frame_bgr[y1:y2, x1:x2, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            crop = torch.nn.functional.interpolate(crop[None], size=self.size, mode="bilinear", align_corners=False)[0]
            crops.append(crop)
        batch = (torch.stack(crops).to(self.device) - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(batch)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float64)
