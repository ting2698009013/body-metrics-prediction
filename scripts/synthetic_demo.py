"""Offline engineering demo using random toy data, not human measurements.

The artificial target equations are arbitrary software fixtures. All fitted
bundles are written to an automatically removed temporary directory. This
script never reads data/, an existing model directory, or a network resource.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from body_soft_sensor.inference import EXPECTED_TARGETS, ModelRegistry
from body_soft_sensor.modeling import (
    RAW_MODEL_INPUTS,
    ProductFeatureBuilder,
    finite_sample_absolute_quantile,
)
from body_soft_sensor.webapp import create_app


def make_toy_table(rows: int, seed: int) -> pd.DataFrame:
    """Generate fictional independent records and deliberately simple targets."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "sex_code": rng.integers(1, 3, rows),
        "age_years": rng.integers(20, 55, rows),
        "height_cm": rng.uniform(155, 185, rows),
        "weight_kg": rng.uniform(55, 90, rows),
        "waist_circumference_cm": rng.uniform(75, 100, rows),
    })
    # These equations are only a known learnable signal for regression tests;
    # they do not encode a physiological model or a formula for self-assessment.
    frame["body_fat_percent"] = (
        24 + 0.15 * (frame["waist_circumference_cm"] - 85)
        + 0.06 * (frame["age_years"] - 35) + rng.normal(0, 1, rows)
    )
    frame["total_fat_mass_kg"] = (
        20 + 0.2 * (frame["weight_kg"] - 70)
        + rng.normal(0, 0.7, rows)
    )
    frame["total_lean_mass_kg"] = (
        45 + 0.25 * (frame["height_cm"] - 170)
        + rng.normal(0, 0.9, rows)
    )
    return frame


def run_demo(rows: int = 900, seed: int = 42) -> dict:
    if rows < 100:
        raise ValueError("Use at least 100 toy records")
    table = make_toy_table(rows, seed)
    # No person is measured over time; these are disjoint independent toy rows.
    train_end, calibration_end = int(rows * 0.6), int(rows * 0.8)
    train = table.iloc[:train_end]
    calibration = table.iloc[train_end:calibration_end]
    test = table.iloc[calibration_end:]
    metrics = {}
    with tempfile.TemporaryDirectory(prefix="body-synthetic-demo-") as temporary:
        root = Path(temporary)
        models = root / "models"
        models.mkdir()
        files = {}
        for target in EXPECTED_TARGETS:
            model = make_pipeline(ProductFeatureBuilder(), StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[RAW_MODEL_INPUTS], train[target])
            errors = np.abs(calibration[target] - model.predict(calibration[RAW_MODEL_INPUTS]))
            half_width = finite_sample_absolute_quantile(errors.to_numpy(), coverage=0.9)
            predicted = model.predict(test[RAW_MODEL_INPUTS])
            benchmark = {
                "benchmark_rows": len(test),
                "mae": float(mean_absolute_error(test[target], predicted)),
                "r2": float(r2_score(test[target], predicted)),
                "interval_coverage": float(np.mean(np.abs(test[target] - predicted) <= half_width)),
            }
            baseline_mae = float(mean_absolute_error(test[target], np.full(len(test), train[target].mean())))
            metrics[target] = {**benchmark, "mean_baseline_mae": baseline_mae}
            bundle = {
                "model": model,
                "model_version": "v0.0",
                "target": target,
                "raw_inputs": RAW_MODEL_INPUTS,
                "training_cycles": ["synthetic-train"],
                "model_selection": {"family": "ridge"},
                "prediction_interval": {
                    "nominal_coverage": 0.9,
                    "calibration_key": "global",
                    "group_half_widths": {},
                    "global_fallback_half_width": half_width,
                },
                "benchmark": benchmark,
            }
            path = models / f"v0_0__{target}.joblib"
            joblib.dump(bundle, path)
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        manifest = {
            "model_version": "v0.0",
            "runtime": {"scikit_learn": sklearn.__version__},
            "models": files,
            "dataset": {
                "synthetic": True,
                "scope": "Fictional toy records; no physiological or clinical interpretation",
                "benchmark_cycle": "synthetic-test",
            },
        }
        (root / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        registry = ModelRegistry(models)
        payload = {"sex": "male", "age_years": 35, "height_cm": 170.0,
                   "weight_kg": 70.0, "waist_circumference_cm": 85.0}
        # TestClient makes in-process requests; no service is exposed on a port.
        with TestClient(create_app(registry=registry)) as client:
            response = client.post("/api/v1/predict", json=payload)
            response.raise_for_status()
            info = client.get("/api/v1/model-info").json()
            assert info["synthetic_data"] is True
            assert len(response.json()["results"]) == 3
            assert "纯合成" in response.json()["disclaimer"]
            assert client.get("/").status_code == 200
            assert client.get("/assets/app.js").status_code == 200
    return {
        "data": "synthetic only; metrics have no medical meaning",
        "seed": seed,
        "split_rows": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "metrics": metrics,
        "api_smoke_test": "passed",
        "model_files": "temporary files removed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.rows, args.seed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
