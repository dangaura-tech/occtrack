"""Open-vocabulary detection: track whatever a text prompt names.

OWL-ViT / OWLv2 (Minderer et al., ECCV 2022 / NeurIPS 2023) embed every image patch into
CLIP's joint image-text space and predict one box per patch. A detection's class score is the
similarity between that patch embedding and the embedding of each text query, so the
vocabulary is chosen at run time (``["person", "bicycle", "a red backpack"]``) instead of
being fixed by a training set.

The model is loaded lazily from the Hugging Face hub (``pip install occtrack[openvocab]``).
Decoding is implemented here rather than through the processor's post-processing helpers,
whose names and signatures change between ``transformers`` releases:

* scores are ``sigmoid`` logits, maximised over the text queries (argmax = label);
* boxes are ``(cx, cy, w, h)`` normalised to the *network input*. OWLv2 pads images to a
  square at the bottom/right, so its coordinates are relative to ``max(H, W)``; OWL-ViT
  resizes without padding, so they are relative to ``(W, H)``;
* per-class non-maximum suppression removes duplicates from neighbouring patches.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices sorted by decreasing score."""
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        iou = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=int)


def decode_owl(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    image_hw: Tuple[int, int],
    score_threshold: float = 0.1,
    nms_iou: float = 0.5,
    padded_square: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's OWL outputs into ``(boxes_xyxy, scores, labels)`` in pixels.

    Args:
        logits: (Q, T) patch-query logits for T text queries.
        pred_boxes: (Q, 4) normalised ``cx, cy, w, h``.
        image_hw: original ``(height, width)``.
        padded_square: OWLv2 convention (coordinates relative to the padded square).
    """
    h, w = image_hw
    probs = _sigmoid(np.asarray(logits, float))
    scores, labels = probs.max(1), probs.argmax(1)
    keep = scores >= score_threshold
    cx, cy, bw, bh = np.asarray(pred_boxes, float)[keep].T
    sx, sy = (max(h, w), max(h, w)) if padded_square else (w, h)
    boxes = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy], 1)
    boxes = np.clip(boxes, 0, [w, h, w, h])
    scores, labels = scores[keep], labels[keep]
    out = [np.zeros(0, int)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        out.append(idx[nms(boxes[idx], scores[idx], nms_iou)])
    sel = np.concatenate(out)
    sel = sel[np.argsort(-scores[sel])]
    return boxes[sel].reshape(-1, 4), scores[sel], labels[sel]


class OpenVocabDetector:
    """Text-prompted detector with the same call signature as :class:`YOLODetector`.

    ``detector(frame)`` returns ``(boxes, scores)``; :meth:`detect` also returns label indices
    into ``prompts`` for class-aware tracking.
    """

    def __init__(
        self,
        prompts: Sequence[str],
        model_name: str = "google/owlv2-base-patch16-ensemble",
        score_threshold: float = 0.1,
        nms_iou: float = 0.5,
        device: Optional[str] = None,
        model=None,
        processor=None,
    ) -> None:
        if not prompts:
            raise ValueError("at least one text prompt is required")
        self.prompts = [p.strip() for p in prompts]
        self.queries = [p if p.startswith(("a ", "an ", "the ")) else f"a photo of a {p}" for p in self.prompts]
        self.model_name, self.score_threshold, self.nms_iou = model_name, score_threshold, nms_iou
        self.padded_square = "owlv2" in model_name.lower()
        self._model, self._processor, self._device = model, processor, device
        self._text_inputs = None

    def _load(self):
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
        self._model = self._model.to(self._device).eval()
        return torch

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self._load()
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        inputs = self._processor(text=[self.queries], images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        return decode_owl(
            out.logits[0].float().cpu().numpy(),
            out.pred_boxes[0].float().cpu().numpy(),
            frame_bgr.shape[:2],
            self.score_threshold,
            self.nms_iou,
            self.padded_square,
        )

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        boxes, scores, _ = self.detect(frame_bgr)
        return boxes, scores
