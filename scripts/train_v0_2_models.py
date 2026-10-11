"""Package v0.2 models with sex-by-BMI calibrated prediction intervals."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from body_soft_sensor.modeling import RAW_MODEL_INPUTS

from train_v0_1_models import (
    BENCHMARK_CYCLE,
    DATA_PATH,
    DEVELOPMENT_CYCLES,
    INTERVAL_COVERAGE,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_UNITS,
    analysis_data,
    candidate_configs,
    make_estimator,
    metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = ROOT / "artifacts" / "v0_2_calibration"
VERSION = "v0.2"
FILE_VERSION = "v0_2"


def main() -> None:
    selection = pd.read_csv(CALIBRATION_DIR / "method_selection.csv")
    selected_rows = selection.loc[selection["selected"].astype(str).str.lower().eq("true")]
    if len(selected_rows) != 1:
        raise ValueError("Calibration experiment must select exactly one method")
    selected_method = selected_rows["method"].iat[0]
    if selected_method != "sex_bmi_group":
        raise ValueError(f"v0.2 packaging expects sex_bmi_group, got {selected_method}")

    all_widths = json.loads(
        (CALIBRATION_DIR / "calibration_widths.json").read_text(encoding="utf-8")
    )
    width_spec = all_widths[selected_method]
    benchmark_eval = pd.read_parquet(CALIBRATION_DIR / "benchmark_predictions.parquet")
    benchmark_eval = benchmark_eval.loc[benchmark_eval["method"].eq(selected_method)]
    winners = pd.read_csv(OUTPUT_DIR / "selected_models.csv")
    data = pd.read_parquet(DATA_PATH)
    config_lookup = {item["config_id"]: item for item in candidate_configs()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    packaged_rows = []
    for winner in winners.itertuples(index=False):
        target = winner.target
        config = config_lookup[winner.config_id]
        target_data = analysis_data(data, target)
        development = target_data.loc[target_data["cycle"].isin(DEVELOPMENT_CYCLES)]
        benchmark = target_data.loc[target_data["cycle"].eq(BENCHMARK_CYCLE)]

        model = make_estimator(config)
        model.fit(development[RAW_MODEL_INPUTS], development[target])
        predicted = model.predict(benchmark[RAW_MODEL_INPUTS])
        point_metrics = metrics(
            benchmark[target], predicted, benchmark["exam_sample_weight"]
        )
        target_benchmark = benchmark_eval.loc[benchmark_eval["target"].eq(target)]
        interval_coverage = float(target_benchmark["covered"].mean())
        interval_mean_half_width = float(target_benchmark["half_width"].mean())
        benchmark_summary = {
            "benchmark_rows": int(len(benchmark)),
            "mae": point_metrics["mae"],
            "rmse": point_metrics["rmse"],
            "r2": point_metrics["r2"],
            "interval_coverage": interval_coverage,
            "interval_mean_half_width": interval_mean_half_width,
        }

        bundle = {
            "model_version": VERSION,
            "model": model,
            "raw_inputs": RAW_MODEL_INPUTS,
            "target": target,
            "target_unit": TARGET_UNITS[target],
            "training_cycles": DEVELOPMENT_CYCLES,
            "benchmark_cycle": BENCHMARK_CYCLE,
            "model_selection": {
                "config_id": config["config_id"],
                "family": config["family"],
                "params": config["params"],
                "criterion": "locked v0.1 winner; lowest mean leave-one-cycle-out development MAE",
            },
            "prediction_interval": {
                "method": "sex-by-BMI-group leave-one-cycle-out absolute residual quantile",
                "calibration_key": selected_method,
                "nominal_coverage": INTERVAL_COVERAGE,
                "group_half_widths": width_spec[target]["groups"],
                "global_fallback_half_width": width_spec[target]["fallback"],
            },
            "benchmark": benchmark_summary,
            "intended_use": "personal software feasibility prototype; not medical diagnosis",
        }
        path = MODEL_DIR / f"{FILE_VERSION}__{target}.joblib"
        joblib.dump(bundle, path, compress=3)
        packaged_rows.append({"target": target, **benchmark_summary})

    pd.DataFrame(packaged_rows).to_csv(
        CALIBRATION_DIR / "v0_2_benchmark_summary.csv", index=False
    )
    write_artifact_manifest(selected_method)
    print(pd.DataFrame(packaged_rows).to_string(index=False))


def write_artifact_manifest(calibration_method: str) -> None:
    models = {}
    for path in sorted(MODEL_DIR.glob(f"{FILE_VERSION}__*.joblib")):
        models[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "model_version": VERSION,
        "artifact_format": "trusted local joblib bundles",
        "point_model_source": "v0.1 locked model-selection winners",
        "calibration_method": calibration_method,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "models": models,
    }
    content = json.dumps(manifest, ensure_ascii=False, indent=2)
    (OUTPUT_DIR / "model_manifest_v0_2.json").write_text(content, encoding="utf-8")
    (OUTPUT_DIR / "model_manifest.json").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
