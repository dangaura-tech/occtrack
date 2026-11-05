from occtrack.benchmark import as_dict, as_markdown, benchmark


def test_benchmark_runs_and_reports():
    res = benchmark(targets=(3, 6), frames=30, warmup=5)
    assert [r.targets for r in res] == [3, 6]
    assert all(r.frames == 25 and r.fps > 0 and r.ms_per_frame_p95 >= r.ms_per_frame_p50 for r in res)
    assert res[1].mean_detections > res[0].mean_detections
    md = as_markdown(res, "test")
    assert sum(line.startswith("|") for line in md.splitlines()) == 4 and "### test" in md
    assert as_dict(res)[0]["targets"] == 3


def test_benchmark_with_features():
    (r,) = benchmark(targets=(5,), frames=20, with_features=True, warmup=0)
    assert r.frames == 20
