import numpy as np
import pytest

from occtrack.detectors.open_vocab import OpenVocabDetector, box_iou, decode_owl, nms
from occtrack.multiclass import MultiClassTracker
from occtrack.tracker import TrackerConfig


def _logit(p):
    return np.log(p / (1 - p))


def test_nms_and_iou():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    assert box_iou(boxes[:1], boxes[1:2])[0, 0] == pytest.approx(81 / 119)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_decode_owl_coordinates_threshold_and_classes():
    # 3 patches x 2 queries; image 200 (h) x 400 (w)
    logits = np.array([[_logit(0.9), _logit(0.05)], [_logit(0.02), _logit(0.6)], [_logit(0.01), _logit(0.03)]])
    pred = np.array([[0.25, 0.25, 0.1, 0.1], [0.5, 0.25, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5]])
    boxes, scores, labels = decode_owl(logits, pred, (200, 400), score_threshold=0.1, padded_square=True)
    assert labels.tolist() == [0, 1] and scores == pytest.approx([0.9, 0.6])
    # OWLv2: normalised to the padded 400 x 400 square
    np.testing.assert_allclose(boxes[0], [80, 80, 120, 120])
    boxes2, _, _ = decode_owl(logits, pred, (200, 400), score_threshold=0.1, padded_square=False)
    np.testing.assert_allclose(boxes2[0], [80, 40, 120, 60])  # OWL-ViT: relative to (W, H)


def test_decode_suppresses_duplicates_per_class_only():
    logits = np.array([[_logit(0.9), -9], [_logit(0.8), -9], [-9, _logit(0.7)]])
    same = [0.5, 0.5, 0.2, 0.2]
    boxes, scores, labels = decode_owl(logits, np.array([same, same, same]), (100, 100), 0.1)
    assert sorted(labels.tolist()) == [0, 1]  # duplicate of class 0 removed, overlapping class 1 kept


class _Out:
    def __init__(self, logits, boxes):
        import torch

        self.logits, self.pred_boxes = torch.tensor(logits)[None], torch.tensor(boxes)[None]


def test_detector_with_injected_model():
    torch = pytest.importorskip("torch")
    seen = {}

    def processor(text, images, return_tensors):
        seen["text"], seen["shape"] = text, images.shape
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    class Model(torch.nn.Module):
        def forward(self, pixel_values):
            return _Out([[_logit(0.95), -9.0], [-9.0, _logit(0.5)]], [[0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]])

    det = OpenVocabDetector(["person", "a red backpack"], model=Model(), processor=processor, device="cpu")
    frame = np.zeros((100, 100, 3), np.uint8)
    boxes, scores, labels = det.detect(frame)
    assert seen["text"] == [["a photo of a person", "a red backpack"]]
    assert seen["shape"] == (100, 100, 3)
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(boxes[0], [40, 30, 60, 70])
    b2, s2 = det(frame)
    np.testing.assert_allclose(b2, boxes)
    with pytest.raises(ValueError):
        OpenVocabDetector([])


def test_multiclass_tracker_keeps_classes_apart():
    mt = MultiClassTracker(TrackerConfig())
    ids = {}
    for f in range(12):
        x = 10.0 + 5 * f
        # a person and a bicycle moving together, boxes nearly identical
        boxes = np.array([[x, 50, x + 40, 150], [x + 2, 52, x + 42, 152]])
        out = mt.update(boxes, np.array([0.9, 0.9]), [0, 1])
        for t in out:
            ids.setdefault(t.label, set()).add(t.track_id)
    assert ids[0] and ids[1]
    assert len(ids[0]) == 1 and len(ids[1]) == 1  # one stable identity per object
    assert ids[0].isdisjoint(ids[1])  # IDs are unique across classes
    assert isinstance(mt.update(np.zeros((0, 4)), np.zeros(0), []), list)  # empty frames age tracks without error
