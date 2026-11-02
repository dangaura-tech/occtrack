import numpy as np
import pytest

from occtrack.eval import evaluate_sequence, summarize
from occtrack.io import load_mot, write_mot


def seq(rows):
    out = {}
    for f, tid, x in rows:
        out.setdefault(f, []).append([tid, x, 0, x + 10, 10])
    return {f: np.array(v, dtype=float) for f, v in out.items()}


def test_perfect_tracking():
    gt = seq([(f, 1, f) for f in range(1, 11)])
    res = evaluate_sequence(gt, gt)
    assert res["MOTA"] == 1.0 and res["IDF1"] == 1.0 and res["IDSW"] == 0
    assert res["MOTP"] == pytest.approx(1.0)


def test_identity_switch_is_counted():
    gt = seq([(f, 1, 0) for f in range(1, 11)])
    pred = seq([(f, 7 if f <= 5 else 8, 0) for f in range(1, 11)])
    res = evaluate_sequence(gt, pred)
    assert res["IDSW"] == 1
    assert res["MOTA"] == pytest.approx(1 - 1 / 10)
    assert res["IDF1"] == pytest.approx(2 * 5 / 20)


def test_false_positives_and_misses():
    gt = seq([(1, 1, 0), (2, 1, 0)])
    pred = seq([(1, 1, 0), (2, 2, 100)])
    res = evaluate_sequence(gt, pred)
    assert res["FP"] == 1 and res["FN"] == 1
    assert res["MOTA"] == 0.0


def test_summarize_sums_counts():
    gt = seq([(f, 1, 0) for f in range(1, 5)])
    r = evaluate_sequence(gt, gt)
    total = summarize({"a": r, "b": r})
    assert total["GT"] == 8 and total["MOTA"] == 1.0


def test_mot_io_roundtrip(tmp_path):
    path = tmp_path / "res.txt"
    write_mot(str(path), [(1, 3, np.array([10, 20, 30, 60]), 0.9), (2, 3, np.array([12, 20, 32, 60]), 0.8)])
    data = load_mot(str(path))
    assert sorted(data) == [1, 2]
    np.testing.assert_allclose(data[1][0, :5], [3, 10, 20, 30, 60])
