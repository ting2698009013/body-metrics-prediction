"""Evaluate the complete selection/calibration procedure with held-out cycles.

Only 2011--2016 development cycles are read into the experiment. Nothing here
evaluates the historical 2017--2018 benchmark or publishes a model bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from body_soft_sensor.modeling import RAW_MODEL_INPUTS
from run_v0_2_calibration_experiments import (
    CALIBRATION_METHODS,
    add_groups,
    apply_widths,
    fit_widths,
    selection_table,
    summarize,
)
from train_v0_1_models import (
    DATA_PATH,
    DEVELOPMENT_CYCLES,
    INTERVAL_COVERAGE,
    TARGETS,
    analysis_data,
    candidate_configs,
    make_estimator,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "v0_3_nested_calibration"
PROTOCOL_VERSION = "nested-cycle-v1"
SPLIT_SEED = 3103


def configs_for_preset(preset: str) -> list[dict]:
    """Smoke checks the workflow; compact is a smaller declared candidate grid."""
    ids = {
        "smoke": {"ridge_a1", "ridge_a100"},
        "compact": {
            "ridge_a10", "extra_leaf5_mf0.7", "hist_lr0.04_leaf15",
            "hist_lr0.08_leaf31",
        },
    }
    if preset == "full":
        return candidate_configs()
    if preset not in ids:
        raise ValueError(f"Unknown preset: {preset}")
    return [config for config in candidate_configs() if config["config_id"] in ids[preset]]


def split_calibration_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split held-out residuals by participant, without inspecting their errors.

    Ranking hashes makes the split deterministic and balanced, including small
    test fixtures. Target-specific missingness can change the midpoint slightly.
    """
    if len(frame) < 4 or frame["participant_id"].duplicated().any():
        raise ValueError("Each inner cycle/target needs at least four unique participants")
    keys = frame["participant_id"].astype(str) + f"|{SPLIT_SEED}"
    order = np.argsort(pd.util.hash_pandas_object(keys, index=False).to_numpy(), kind="stable")
    midpoint = len(order) // 2
    return frame.iloc[order[:midpoint]].copy(), frame.iloc[order[midpoint:]].copy()


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in [
        Path(__file__),
        ROOT / "scripts" / "train_v0_1_models.py",
        ROOT / "scripts" / "run_v0_2_calibration_experiments.py",
        ROOT / "src" / "body_soft_sensor" / "modeling.py",
    ]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_key(frame: pd.DataFrame, target: str, config: dict, source_digest: str) -> str:
    header = {
        "protocol": PROTOCOL_VERSION,
        "source_sha256": source_digest,
        "target": target,
        "config": config,
        "columns": frame.columns.tolist(),
        "dtypes": [str(value) for value in frame.dtypes],
        "versions": [sklearn.__version__, np.__version__, pd.__version__],
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _new_estimator(config: dict, factory: Callable, workers: int):
    estimator = factory(config)
    if hasattr(estimator, "get_params"):
        changes = {
            name: workers for name in estimator.get_params(deep=True)
            if name == "n_jobs" or name.endswith("__n_jobs")
        }
        if changes:
            estimator.set_params(**changes)
    return estimator


def _inner_predictions(
    training: pd.DataFrame,
    target: str,
    config: dict,
    estimator_factory: Callable,
    workers: int,
    cache_dir: Path | None,
    source_digest: str,
) -> np.ndarray:
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(training, target, config, source_digest)}.npy"
        if cache_path.is_file():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.shape == (len(training),) and np.isfinite(cached).all():
                    return cached
            except (ValueError, OSError):
                pass
    predictions = np.empty(len(training), dtype=float)
    for validation_cycle in sorted(training["cycle"].unique()):
        validation_mask = training["cycle"].eq(validation_cycle)
        model = _new_estimator(config, estimator_factory, workers)
        model.fit(training.loc[~validation_mask, RAW_MODEL_INPUTS], training.loc[~validation_mask, target])
        predictions[validation_mask.to_numpy()] = model.predict(training.loc[validation_mask, RAW_MODEL_INPUTS])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # No model pickle is loaded from the cache. Interrupted writes are retried.
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, predictions, allow_pickle=False)
        temporary.replace(cache_path)
    return predictions


def _prediction_frame(frame: pd.DataFrame, target: str, predicted: np.ndarray) -> pd.DataFrame:
    result = add_groups(frame)
    result["target"] = target
    result["actual"] = result[target]
    result["predicted"] = predicted
    result["absolute_error"] = np.abs(result["actual"] - result["predicted"])
    return result


def _select_method(inner_predictions: pd.DataFrame) -> pd.DataFrame:
    evaluations = []
    # Within each held-out inner cycle the point model has seen neither the
    # calibration half nor the method-validation half of these participants.
    for _, cycle_frame in inner_predictions.groupby("cycle", observed=True):
        calibrations, validations = [], []
        for _, target_frame in cycle_frame.groupby("target", observed=True):
            calibration, validation = split_calibration_selection(target_frame)
            calibrations.append(calibration)
            validations.append(validation)
        calibration = pd.concat(calibrations, ignore_index=True)
        validation = pd.concat(validations, ignore_index=True)
        for method in CALIBRATION_METHODS:
            evaluations.append(apply_widths(validation, method, fit_widths(calibration, method)))
    overall, subgroups = summarize(pd.concat(evaluations, ignore_index=True), "inner_method_selection")
    return selection_table(overall, subgroups)


def metric_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = predictions.groupby(group_columns, observed=True).agg(
        rows=("actual", "size"),
        mae=("absolute_error", "mean"),
        coverage=("covered", "mean"),
        mean_half_width=("half_width", "mean"),
    ).reset_index()
    result["nominal_coverage"] = INTERVAL_COVERAGE
    return result


def run_nested(
    data: pd.DataFrame,
    configs: list[dict],
    *,
    targets: list[str] | None = None,
    estimator_factory: Callable = make_estimator,
    workers: int = 2,
    cache_dir: Path | None = None,
) -> dict:
    """Outer validation data never enter any model fit or selection decision.

    Cache use is restricted to the standard estimator factory so alternate
    estimators cannot accidentally reuse the standard estimator's predictions.
    """
    targets = list(TARGETS if targets is None else targets)
    if not configs or len({config["config_id"] for config in configs}) != len(configs):
        raise ValueError("At least one uniquely identified candidate is required")
    if not targets or workers < 1:
        raise ValueError("At least one target and a positive worker count are required")
    if cache_dir is not None and estimator_factory is not make_estimator:
        raise ValueError("Caching requires the standard estimator factory")
    # Do this before analysis_data, completeness checks, or any label statistics.
    development = data.loc[data["cycle"].isin(DEVELOPMENT_CYCLES)].copy()
    if set(development["cycle"].unique()) != set(DEVELOPMENT_CYCLES):
        raise ValueError("All three development cycles are required")
    if development["participant_id"].duplicated().any():
        raise ValueError("A participant must occur in only one development row/cycle")
    source_digest = _source_digest()
    outer_predictions, config_scores, method_scores, fold_records = [], [], [], []
    with threadpool_limits(limits=workers):
        for outer_cycle in DEVELOPMENT_CYCLES:
            outer_training = development.loc[development["cycle"].ne(outer_cycle)].copy()
            outer_validation = development.loc[development["cycle"].eq(outer_cycle)].copy()
            inner_frames, target_training, selected_configs = [], {}, {}
            for target in targets:
                training = analysis_data(outer_training, target).reset_index(drop=True)
                if set(training["cycle"].unique()) != set(DEVELOPMENT_CYCLES) - {outer_cycle}:
                    raise ValueError(f"Missing complete inner cycle for {target}")
                target_training[target] = training
                candidate_predictions, candidates = {}, []
                for config in configs:
                    predicted = _inner_predictions(training, target, config, estimator_factory, workers, cache_dir, source_digest)
                    candidate_predictions[config["config_id"]] = predicted
                    errors = pd.DataFrame({"cycle": training["cycle"], "error": np.abs(training[target] - predicted)})
                    per_cycle = errors.groupby("cycle")["error"].mean()
                    candidates.append({
                        "outer_cycle": outer_cycle, "target": target,
                        "config_id": config["config_id"],
                        "mean_inner_cycle_mae": float(per_cycle.mean()),
                        "worst_inner_cycle_mae": float(per_cycle.max()),
                        "inner_cycles": "|".join(sorted(per_cycle.index)),
                    })
                ranked = pd.DataFrame(candidates).sort_values(["mean_inner_cycle_mae", "config_id"], kind="stable")
                selected_id = str(ranked.iloc[0]["config_id"])
                ranked["selected"] = ranked["config_id"].eq(selected_id)
                config_scores.append(ranked)
                selected_configs[target] = next(config for config in configs if config["config_id"] == selected_id)
                inner_frames.append(_prediction_frame(training, target, candidate_predictions[selected_id]))
            inner = pd.concat(inner_frames, ignore_index=True)
            method_selection = _select_method(inner)
            method_selection.insert(0, "outer_cycle", outer_cycle)
            method_scores.append(method_selection)
            selected_method = str(method_selection.loc[method_selection["selected"], "method"].iat[0])
            widths = fit_widths(inner, selected_method)
            training_cycles = sorted(outer_training["cycle"].unique())
            fold_records.append({
                "outer_cycle": outer_cycle,
                "point_training_cycles": training_cycles,
                "config_selection_cycles": training_cycles,
                "method_selection_cycles": training_cycles,
                "calibration_fit_cycles": training_cycles,
                "selected_configs": selected_configs,
                "selected_method": selected_method,
                "calibration_widths": widths,
            })
            # Validation labels are consulted only after all decisions are fixed.
            for target in targets:
                training = target_training[target]
                validation = analysis_data(outer_validation, target)
                if validation.empty:
                    raise ValueError(f"No complete outer validation rows for {target}")
                model = _new_estimator(selected_configs[target], estimator_factory, workers)
                model.fit(training[RAW_MODEL_INPUTS], training[target])
                evaluated = apply_widths(_prediction_frame(validation, target, model.predict(validation[RAW_MODEL_INPUTS])), selected_method, widths)
                evaluated["outer_cycle"] = outer_cycle
                evaluated["config_id"] = selected_configs[target]["config_id"]
                outer_predictions.append(evaluated)
            print(f"Outer cycle {outer_cycle}: {selected_method}; " + ", ".join(f"{key}={value['config_id']}" for key, value in selected_configs.items()), flush=True)
    predictions = pd.concat(outer_predictions, ignore_index=True)
    outer_metrics = metric_table(predictions, ["outer_cycle", "target", "method", "config_id"])
    overall = metric_table(predictions, ["target"])
    macro_mae = outer_metrics.groupby("target")["mae"].mean().rename("mean_outer_cycle_mae")
    overall = overall.merge(macro_mae, on="target")
    return {
        "predictions": predictions,
        "outer_metrics": outer_metrics,
        "overall_metrics": overall,
        "subgroup_metrics": metric_table(predictions, ["target", "sex_bmi_group"]),
        "config_scores": pd.concat(config_scores, ignore_index=True),
        "method_scores": pd.concat(method_scores, ignore_index=True),
        "folds": fold_records,
        "source_sha256": source_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--preset", choices=["smoke", "compact", "full"], default="compact")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    configs = configs_for_preset(args.preset)
    # Parquet predicate pushdown avoids loading the historical benchmark rows.
    data = pd.read_parquet(args.data, filters=[("cycle", "in", DEVELOPMENT_CYCLES)])
    result = run_nested(data, configs, workers=args.workers, cache_dir=None if args.no_cache else args.output / "cache")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["outer_metrics", "overall_metrics", "subgroup_metrics", "config_scores", "method_scores"]:
        result[name].to_csv(args.output / f"{name}.csv", index=False)
    # Keep predictions needed to audit metrics; avoid duplicating every target label.
    keep = ["participant_id", "outer_cycle", "sex", "bmi", "sex_bmi_group", "target", "actual", "predicted", "absolute_error", "method", "config_id", "half_width", "covered"]
    result["predictions"][keep].to_parquet(args.output / "outer_predictions.parquet", index=False)
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "preset": args.preset,
        "smoke_only": args.preset == "smoke",
        "candidate_configs": configs,
        "calibration_methods": CALIBRATION_METHODS,
        "nominal_coverage": INTERVAL_COVERAGE,
        "development_cycles": DEVELOPMENT_CYCLES,
        "historical_benchmark_evaluated": False,
        "production_model_modified": False,
        "workers": args.workers,
        "source_sha256": result["source_sha256"],
        "development_rows_sha256": hashlib.sha256(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "folds": result["folds"],
        "limitations": [
            "Only three historical development cycles; this is not a fresh prospective test.",
            "Candidate families, inputs and grouping boundaries were designed after prior development-data exploration.",
            "Inner OOF residuals use one-cycle models; outer point predictions use two-cycle models.",
            "Adaptive OOF calibration and survey-cycle shift do not confer exact conformal coverage guarantees.",
            "Subgroup method selection can be noisy for sparse groups; report all subgroup counts.",
            "Pooled errors and coverage are unweighted individual-sample diagnostics, not survey-population estimates.",
        ],
    }
    (args.output / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["overall_metrics"].to_string(index=False))
    print(f"Saved development-only evaluation to {args.output}")


if __name__ == "__main__":
    main()
