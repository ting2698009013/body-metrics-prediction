"""Select, train, and evaluate the locked-input v0.1 regression models."""

from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from body_soft_sensor.modeling import (
    RAW_MODEL_INPUTS,
    ProductFeatureBuilder,
    finite_sample_absolute_quantile,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nhanes_2011_2018_body_composition.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "model_selection"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
DEVELOPMENT_CYCLES = ["2011-2012", "2013-2014", "2015-2016"]
BENCHMARK_CYCLE = "2017-2018"
TARGETS = ["body_fat_percent", "total_fat_mass_kg", "total_lean_mass_kg"]
TARGET_UNITS = {
    "body_fat_percent": "percentage_points",
    "total_fat_mass_kg": "kg",
    "total_lean_mass_kg": "kg",
}
RANDOM_STATE = 42
INTERVAL_COVERAGE = 0.90
BMI_GROUP_EDGES = [0, 25, 30, 35, 40, np.inf]
BMI_GROUP_LABELS = ["<25", "25-29.9", "30-34.9", "35-39.9", "40+"]


def candidate_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        configs.append(
            {"config_id": f"ridge_a{alpha:g}", "family": "ridge", "params": {"alpha": alpha}}
        )
    for alpha in [0.001, 0.01, 0.1]:
        for l1_ratio in [0.1, 0.7]:
            configs.append(
                {
                    "config_id": f"elastic_a{alpha:g}_l1{l1_ratio:g}",
                    "family": "elastic_net",
                    "params": {"alpha": alpha, "l1_ratio": l1_ratio},
                }
            )
    for min_samples_leaf in [3, 8]:
        for max_features in [0.7, 1.0]:
            configs.append(
                {
                    "config_id": f"rf_leaf{min_samples_leaf}_mf{max_features:g}",
                    "family": "random_forest",
                    "params": {
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                    },
                }
            )
    for min_samples_leaf in [2, 5, 10]:
        for max_features in [0.7, 1.0]:
            configs.append(
                {
                    "config_id": f"extra_leaf{min_samples_leaf}_mf{max_features:g}",
                    "family": "extra_trees",
                    "params": {
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                    },
                }
            )
    for learning_rate in [0.04, 0.08, 0.12]:
        for max_leaf_nodes in [15, 31]:
            configs.append(
                {
                    "config_id": f"hist_lr{learning_rate:g}_leaf{max_leaf_nodes}",
                    "family": "hist_gradient_boosting",
                    "params": {
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                    },
                }
            )
    return configs


def make_estimator(config: dict[str, object]) -> Pipeline:
    family = str(config["family"])
    params = dict(config["params"])
    steps: list[tuple[str, object]] = [("features", ProductFeatureBuilder())]
    if family == "ridge":
        steps.extend(
            [("scaler", StandardScaler()), ("model", Ridge(**params))]
        )
    elif family == "elastic_net":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        **params,
                        max_iter=30_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    elif family == "random_forest":
        steps.append(
            (
                "model",
                RandomForestRegressor(
                    **params,
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
    elif family == "extra_trees":
        steps.append(
            (
                "model",
                ExtraTreesRegressor(
                    **params,
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )
    elif family == "hist_gradient_boosting":
        steps.append(
            (
                "model",
                HistGradientBoostingRegressor(
                    **params,
                    max_iter=300,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline(steps)


def analysis_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
    required = RAW_MODEL_INPUTS + [
        target,
        "cycle",
        "participant_id",
        "race_ethnicity_code",
        "exam_sample_weight",
    ]
    return data[required].dropna().copy()


def metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    sample_weight: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    residual = predicted - actual_array
    absolute = np.abs(residual)
    return {
        "mae": float(mean_absolute_error(actual_array, predicted)),
        "weighted_mae": float(
            mean_absolute_error(actual_array, predicted, sample_weight=sample_weight)
        ),
        "rmse": float(mean_squared_error(actual_array, predicted) ** 0.5),
        "r2": float(r2_score(actual_array, predicted)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "mean_error": float(residual.mean()),
    }


def run_model_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    configs = candidate_configs()
    for target in TARGETS:
        development = analysis_data(data, target)
        development = development.loc[development["cycle"].isin(DEVELOPMENT_CYCLES)]
        for config in configs:
            for validation_cycle in DEVELOPMENT_CYCLES:
                train = development.loc[development["cycle"].ne(validation_cycle)]
                validation = development.loc[development["cycle"].eq(validation_cycle)]
                estimator = make_estimator(config)
                estimator.fit(train[RAW_MODEL_INPUTS], train[target])
                predicted = estimator.predict(validation[RAW_MODEL_INPUTS])
                fold_rows.append(
                    {
                        "target": target,
                        "config_id": config["config_id"],
                        "family": config["family"],
                        "params_json": json.dumps(config["params"], sort_keys=True),
                        "validation_cycle": validation_cycle,
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(validation)),
                        **metrics(
                            validation[target],
                            predicted,
                            validation["exam_sample_weight"],
                        ),
                    }
                )
    folds = pd.DataFrame(fold_rows)
    aggregate = (
        folds.groupby(
            ["target", "config_id", "family", "params_json"], as_index=False
        )
        .agg(
            mean_cv_mae=("mae", "mean"),
            std_cv_mae=("mae", "std"),
            mean_weighted_cv_mae=("weighted_mae", "mean"),
            mean_cv_rmse=("rmse", "mean"),
            mean_cv_r2=("r2", "mean"),
            mean_cv_p90=("absolute_error_p90", "mean"),
            max_cycle_mae=("mae", "max"),
            mean_error=("mean_error", "mean"),
        )
        .sort_values(["target", "mean_cv_mae", "std_cv_mae"])
        .reset_index(drop=True)
    )
    return folds, aggregate


def select_winners(aggregate: pd.DataFrame) -> pd.DataFrame:
    indexes = aggregate.groupby("target")["mean_cv_mae"].idxmin()
    return aggregate.loc[indexes].sort_values("target").reset_index(drop=True)


def fit_winners(
    data: pd.DataFrame,
    winners: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    interval_rows: list[dict[str, object]] = []
    config_lookup = {config["config_id"]: config for config in candidate_configs()}

    for winner in winners.itertuples(index=False):
        target = winner.target
        config = config_lookup[winner.config_id]
        target_data = analysis_data(data, target)
        development = target_data.loc[target_data["cycle"].isin(DEVELOPMENT_CYCLES)]
        benchmark = target_data.loc[target_data["cycle"].eq(BENCHMARK_CYCLE)]

        oof_predictions = np.empty(len(development), dtype=float)
        for validation_cycle in DEVELOPMENT_CYCLES:
            train_mask = development["cycle"].ne(validation_cycle)
            validation_mask = development["cycle"].eq(validation_cycle)
            fold_model = make_estimator(config)
            fold_model.fit(
                development.loc[train_mask, RAW_MODEL_INPUTS],
                development.loc[train_mask, target],
            )
            oof_predictions[validation_mask.to_numpy()] = fold_model.predict(
                development.loc[validation_mask, RAW_MODEL_INPUTS]
            )

        oof_residuals = np.abs(development[target].to_numpy() - oof_predictions)
        global_interval_half_width = finite_sample_absolute_quantile(
            oof_residuals, coverage=INTERVAL_COVERAGE
        )
        development_bmi = development["weight_kg"] / (
            development["height_cm"] / 100.0
        ) ** 2
        development_bmi_group = pd.cut(
            development_bmi,
            bins=BMI_GROUP_EDGES,
            labels=BMI_GROUP_LABELS,
            right=False,
        )
        group_interval_half_widths: dict[str, float] = {}
        for group_name in BMI_GROUP_LABELS:
            group_mask = development_bmi_group.eq(group_name).to_numpy()
            group_interval_half_widths[group_name] = finite_sample_absolute_quantile(
                oof_residuals[group_mask], coverage=INTERVAL_COVERAGE
            )

        final_model = make_estimator(config)
        final_model.fit(development[RAW_MODEL_INPUTS], development[target])
        predicted = final_model.predict(benchmark[RAW_MODEL_INPUTS])
        actual = benchmark[target].to_numpy()
        benchmark_bmi = benchmark["weight_kg"] / (benchmark["height_cm"] / 100.0) ** 2
        benchmark_bmi_group = pd.cut(
            benchmark_bmi,
            bins=BMI_GROUP_EDGES,
            labels=BMI_GROUP_LABELS,
            right=False,
        )
        interval_half_widths = np.asarray(
            [
                group_interval_half_widths.get(str(group), global_interval_half_width)
                for group in benchmark_bmi_group
            ]
        )
        lower = predicted - interval_half_widths
        upper = predicted + interval_half_widths
        covered = (actual >= lower) & (actual <= upper)

        benchmark_rows.append(
            {
                "target": target,
                "config_id": config["config_id"],
                "family": config["family"],
                "params_json": json.dumps(config["params"], sort_keys=True),
                "train_rows": int(len(development)),
                "benchmark_rows": int(len(benchmark)),
                **metrics(actual, predicted, benchmark["exam_sample_weight"]),
                "interval_nominal_coverage": INTERVAL_COVERAGE,
                "interval_calibration": "bmi_group",
                "interval_global_fallback_half_width": global_interval_half_width,
                "interval_mean_half_width": float(interval_half_widths.mean()),
                "interval_min_half_width": float(interval_half_widths.min()),
                "interval_max_half_width": float(interval_half_widths.max()),
                "interval_benchmark_coverage": float(covered.mean()),
                "interval_mean_width": float((upper - lower).mean()),
            }
        )

        predictions = benchmark[
            [
                "participant_id",
                "sex_code",
                "race_ethnicity_code",
                "age_years",
                "height_cm",
                "weight_kg",
                "waist_circumference_cm",
            ]
        ].copy()
        predictions["bmi"] = benchmark_bmi.to_numpy()
        predictions["bmi_group"] = benchmark_bmi_group.astype(object).to_numpy()
        predictions["target"] = target
        predictions["actual"] = actual
        predictions["predicted"] = predicted
        predictions["error"] = predicted - actual
        predictions["absolute_error"] = np.abs(predicted - actual)
        predictions["interval_lower"] = lower
        predictions["interval_upper"] = upper
        predictions["interval_half_width"] = interval_half_widths
        predictions["interval_covered"] = covered
        prediction_frames.append(predictions)

        interval_rows.append(
            {
                "target": target,
                "bmi_group": "__global_fallback__",
                "calibration_method": "leave-one-cycle-out absolute residual quantile",
                "calibration_rows": int(len(development)),
                "nominal_coverage": INTERVAL_COVERAGE,
                "corrected_absolute_residual_quantile": global_interval_half_width,
            }
        )
        for group_name in BMI_GROUP_LABELS:
            group_mask = development_bmi_group.eq(group_name).to_numpy()
            interval_rows.append(
                {
                    "target": target,
                    "bmi_group": group_name,
                    "calibration_method": "BMI-group leave-one-cycle-out absolute residual quantile",
                    "calibration_rows": int(group_mask.sum()),
                    "nominal_coverage": INTERVAL_COVERAGE,
                    "corrected_absolute_residual_quantile": group_interval_half_widths[
                        group_name
                    ],
                }
            )

        joblib.dump(
            {
                "model": final_model,
                "raw_inputs": RAW_MODEL_INPUTS,
                "target": target,
                "target_unit": TARGET_UNITS[target],
                "training_cycles": DEVELOPMENT_CYCLES,
                "benchmark_cycle": BENCHMARK_CYCLE,
                "model_selection": {
                    "config_id": config["config_id"],
                    "family": config["family"],
                    "params": config["params"],
                    "criterion": "lowest mean leave-one-cycle-out development MAE",
                },
                "prediction_interval": {
                    "method": "BMI-group leave-one-cycle-out absolute residual quantile",
                    "nominal_coverage": INTERVAL_COVERAGE,
                    "bmi_group_edges": BMI_GROUP_EDGES,
                    "bmi_group_labels": BMI_GROUP_LABELS,
                    "group_half_widths": group_interval_half_widths,
                    "global_fallback_half_width": global_interval_half_width,
                },
                "intended_use": "personal software feasibility prototype; not medical diagnosis",
            },
            MODEL_DIR / f"v0_1__{target}.joblib",
            compress=3,
        )

    return (
        pd.DataFrame(benchmark_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(interval_rows),
    )


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    working = predictions.copy()
    working["bmi_group"] = pd.cut(
        working["bmi"],
        bins=BMI_GROUP_EDGES,
        labels=BMI_GROUP_LABELS,
        right=False,
    )
    frames = []
    for group_column in ["sex_code", "race_ethnicity_code", "bmi_group"]:
        result = (
            working.groupby(["target", group_column], observed=True, dropna=False)
            .agg(
                rows=("actual", "size"),
                mae=("absolute_error", "mean"),
                mean_error=("error", "mean"),
                absolute_error_p90=("absolute_error", lambda values: values.quantile(0.90)),
                interval_coverage=("interval_covered", "mean"),
            )
            .reset_index()
            .rename(columns={group_column: "group_value"})
        )
        result.insert(1, "group", group_column)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def plot_family_comparison(aggregate: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    family_best = (
        aggregate.loc[aggregate.groupby(["target", "family"])["mean_cv_mae"].idxmin()]
        .sort_values(["target", "mean_cv_mae"])
    )
    families = sorted(family_best["family"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, target in zip(axes, TARGETS, strict=True):
        subset = family_best.loc[family_best["target"].eq(target)].set_index("family")
        values = [subset.loc[family, "mean_cv_mae"] for family in families]
        errors = [subset.loc[family, "std_cv_mae"] for family in families]
        ax.bar(range(len(families)), values, yerr=errors, capsize=3, color="#2563eb")
        ax.set_xticks(range(len(families)), [name.replace("_", "\n") for name in families])
        ax.set_title(target)
        ax.set_ylabel("Leave-one-cycle-out MAE")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Best configuration within each model family")
    fig.savefig(FIGURE_DIR / "01_model_family_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_manifest() -> None:
    manifest = {
        "version": "v0.1",
        "raw_inputs": RAW_MODEL_INPUTS,
        "derived_input": "bmi = weight_kg / (height_cm / 100)^2",
        "targets": TARGETS,
        "development_cycles": DEVELOPMENT_CYCLES,
        "benchmark_cycle": BENCHMARK_CYCLE,
        "selection_criterion": "lowest mean leave-one-cycle-out development MAE",
        "candidate_count": len(candidate_configs()),
        "candidates": candidate_configs(),
        "interval": {
            "nominal_coverage": INTERVAL_COVERAGE,
            "method": "BMI-group leave-one-cycle-out absolute residual quantile",
            "bmi_groups": ["<25", "25-29.9", "30-34.9", "35-39.9", "40+"],
        },
    }
    (OUTPUT_DIR / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_model_artifact_manifest() -> None:
    models = {}
    for path in sorted(MODEL_DIR.glob("v0_1__*.joblib")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        models[path.name] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "model_version": "v0.1",
        "artifact_format": "trusted local joblib bundles",
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
    (OUTPUT_DIR / "model_manifest_v0_1.json").write_text(content, encoding="utf-8")
    (OUTPUT_DIR / "model_manifest.json").write_text(content, encoding="utf-8")


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA_PATH)

    folds, aggregate = run_model_selection(data)
    winners = select_winners(aggregate)
    benchmark, predictions, intervals = fit_winners(data, winners)
    subgroups = subgroup_metrics(predictions)

    folds.to_csv(OUTPUT_DIR / "cycle_cv_fold_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "cycle_cv_aggregate.csv", index=False)
    winners.to_csv(OUTPUT_DIR / "selected_models.csv", index=False)
    benchmark.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    predictions.to_parquet(OUTPUT_DIR / "benchmark_predictions.parquet", index=False)
    intervals.to_csv(OUTPUT_DIR / "interval_calibration.csv", index=False)
    subgroups.to_csv(OUTPUT_DIR / "benchmark_subgroup_metrics.csv", index=False)
    write_manifest()
    write_model_artifact_manifest()
    plot_family_comparison(aggregate)

    print("Selected using development cycles only:")
    print(
        winners[
            ["target", "config_id", "family", "mean_cv_mae", "std_cv_mae"]
        ].to_string(index=False)
    )
    print("\nHeld-out-cycle benchmark and empirical interval coverage:")
    print(
        benchmark[
            [
                "target",
                "family",
                "mae",
                "rmse",
                "r2",
                "interval_mean_half_width",
                "interval_benchmark_coverage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

