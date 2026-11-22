import json

import pytest

from phasemoe.benchmark import main, measure, summarize


def test_cpu_benchmark_artifact_is_self_describing(tmp_path):
    path = tmp_path / "bench.json"
    main(["--tokens", "1", "0", "--warmup", "1", "--repetitions", "3", "--output", str(path)])
    data = json.loads(path.read_text())
    assert data["status"] == "complete"
    assert data["environment"]["device"] == "cpu"
    assert len(data["results"]) == 2
    for case in data["results"]:
        assert case["correctness"]["passed"]
        assert case["correctness"]["independent_oracle"]["passed"]
        assert len(case["timing"]["wall"]["samples_ms"]) == 3
        assert "cuda_event" not in case["timing"]
        assert case["scope"] == "expert_block_including_dispatch"


def test_measure_counts_warmup_and_repetitions():
    calls = []
    measure(lambda: calls.append(1), device="cpu", warmup=2, repetitions=3)
    assert len(calls) == 5
    with pytest.raises(ValueError):
        measure(lambda: None, device="cpu", repetitions=0)


def test_summary_quantiles_and_invalid_samples():
    s = summarize([4, 1, 3, 2])
    assert s["median_ms"] == 2.5
    assert s["p10_ms"] == pytest.approx(1.3)
    assert s["p90_ms"] == pytest.approx(3.7)
    for invalid in ([], [-1], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            summarize(invalid)
