import json

import pytest

from phasemoe.benchmark import main
from phasemoe.plots import load_report


def test_plot_loader_refuses_incomplete_or_mixed_experiments(tmp_path):
    path = tmp_path / "run.json"
    report = main(
        ["--tokens", "1", "2", "--warmup", "0", "--repetitions", "1", "--output", str(path)]
    )
    assert len(load_report(path)["results"]) == 2
    report["results"][1]["dtype"] = "torch.float16"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="mixed"):
        load_report(path)
    report["status"] = "running"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed"):
        load_report(path)
