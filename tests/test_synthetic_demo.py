from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from synthetic_demo import run_demo


def test_offline_demo_trains_calibrates_loads_and_serves_models():
    report = run_demo(rows=300, seed=7)
    assert report["api_smoke_test"] == "passed"
    assert sum(report["split_rows"].values()) == 300
    assert len(report["metrics"]) == 3
    assert all(value["benchmark_rows"] == 60 for value in report["metrics"].values())


def test_too_small_demo_fails_before_training():
    with pytest.raises(ValueError, match="100"):
        run_demo(rows=10)
