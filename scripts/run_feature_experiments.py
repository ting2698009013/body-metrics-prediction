"""Compare incremental feature sets with leave-one-cycle-out validation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from body_soft_sensor.features import (
    FEATURE_SETS,
    RAW_REQUIRED_COLUMNS,
    add_derived_features,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nhanes_2011_2018_body_composition.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "feature_engineering"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
DEVELOPMENT_CYCLES = ["2011-2012", "2013-2014", "2015-2016"]
BENCHMARK_CYCLE = "2017-2018"
TARGETS = ["body_fat_percent", "total_fat_mass_kg", "total_lean_mass_kg"]
RANDOM_STATE = 42


def make_estimators() -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def score_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    sample_weight: pd.Series,
) -> dict[str, float]:
    absolute_error = np.abs(actual.to_numpy() - predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "weighted_mae": float(
            mean_absolute_error(actual, predicted, sample_weight=sample_weight)
        ),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "absolute_error_p90": float(np.quantile(absolute_error, 0.90)),
        "mean_error": float(np.mean(predicted - actual.to_numpy())),
    }


def build_analysis_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Use identical complete cases for every feature set within one target."""

    required = RAW_REQUIRED_COLUMNS + [
        "cycle",
        "participant_id",
        "race_ethnicity_code",
        "exam_sample_weight",
        target,
    ]
    complete = data[required].dropna().copy()
    return add_derived_features(complete)


def run_cycle_validation(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        analysis = build_analysis_data(data, target)
        development = analysis.loc[analysis["cycle"].isin(DEVELOPMENT_CYCLES)]
        for feature_set, features in FEATURE_SETS.items():
            for validation_cycle in DEVELOPMENT_CYCLES:
                train = development.loc[development["cycle"].ne(validation_cycle)]
                validation = development.loc[development["cycle"].eq(validation_cycle)]
                for model_name, estimator in make_estimators().items():
                    estimator.fit(train[features], train[target])
                    predicted = estimator.predict(validation[features])
                    rows.append(
                        {
                            "target": target,
                            "feature_set": feature_set,
                            "feature_count": len(features),
                            "model": model_name,
                            "validation_cycle": validation_cycle,
                            "train_rows": int(len(train)),
                            "validation_rows": int(len(validation)),
                            **score_predictions(
                                validation[target],
                                predicted,
                                validation["exam_sample_weight"],
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_cycle_results(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby(
            ["target", "feature_set", "feature_count", "model"],
            as_index=False,
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
        .sort_values(["target", "mean_cv_mae", "feature_count"])
        .reset_index(drop=True)
    )


def selected_configurations(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Return statistical winners and simpler product candidates.

    Both rules use development-cycle MAE only. The pragmatic rule chooses the
    smallest feature set whose mean MAE is within 1% of the best result.
    """

    rows = []
    for target, target_rows in aggregate.groupby("target"):
        winner = target_rows.loc[target_rows["mean_cv_mae"].idxmin()].copy()
        winner["selection_rule"] = "minimum_cv_mae"
        rows.append(winner)

        threshold = float(winner["mean_cv_mae"]) * 1.01
        practical = target_rows.loc[target_rows["mean_cv_mae"].le(threshold)].sort_values(
            ["feature_count", "mean_cv_mae"]
        ).iloc[0].copy()
        practical["selection_rule"] = "simplest_within_1_percent"
        rows.append(practical)
    return pd.DataFrame(rows).sort_values(["target", "selection_rule"]).reset_index(drop=True)


def fit_and_score_benchmark(
    data: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for configuration in selected.itertuples(index=False):
        target = configuration.target
        feature_set = configuration.feature_set
        model_name = configuration.model
        selection_rule = configuration.selection_rule
        features = FEATURE_SETS[feature_set]
        analysis = build_analysis_data(data, target)
        train = analysis.loc[analysis["cycle"].isin(DEVELOPMENT_CYCLES)]
        benchmark = analysis.loc[analysis["cycle"].eq(BENCHMARK_CYCLE)]
        estimator = make_estimators()[model_name]
        estimator.fit(train[features], train[target])
        predicted = estimator.predict(benchmark[features])

        metric_rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "feature_count": len(features),
                "model": model_name,
                "selection_rule": selection_rule,
                "train_rows": int(len(train)),
                "benchmark_rows": int(len(benchmark)),
                **score_predictions(
                    benchmark[target],
                    predicted,
                    benchmark["exam_sample_weight"],
                ),
            }
        )

        predictions = benchmark[
            ["participant_id", "sex_code", "race_ethnicity_code", "bmi", "waist_circumference_cm"]
        ].copy()
        predictions["target"] = target
        predictions["feature_set"] = feature_set
        predictions["model"] = model_name
        predictions["selection_rule"] = selection_rule
        predictions["actual"] = benchmark[target].to_numpy()
        predictions["predicted"] = predicted
        predictions["error"] = predictions["predicted"] - predictions["actual"]
        predictions["absolute_error"] = predictions["error"].abs()
        prediction_frames.append(predictions)

        joblib.dump(
            {
                "model": estimator,
                "features": features,
                "feature_set": feature_set,
                "target": target,
                "training_cycles": DEVELOPMENT_CYCLES,
                "benchmark_cycle": BENCHMARK_CYCLE,
                "selection_rule": selection_rule,
            },
            MODEL_DIR / f"{target}__{selection_rule}.joblib",
        )

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    working = predictions.copy()
    working["bmi_group"] = pd.cut(
        working["bmi"],
        bins=[0, 25, 30, 35, 40, np.inf],
        labels=["<25", "25-29.9", "30-34.9", "35-39.9", "40+"],
        right=False,
    )
    working["waist_group"] = pd.qcut(
        working["waist_circumference_cm"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    for group_column in ["sex_code", "race_ethnicity_code", "bmi_group", "waist_group"]:
        result = (
            working.groupby(
                ["target", "selection_rule", group_column], observed=True, dropna=False
            )
            .agg(
                rows=("actual", "size"),
                mae=("absolute_error", "mean"),
                mean_error=("error", "mean"),
                absolute_error_p90=("absolute_error", lambda values: values.quantile(0.90)),
            )
            .reset_index()
            .rename(columns={group_column: "group_value"})
        )
        result.insert(2, "group", group_column)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def plot_cv_results(aggregate: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    order = list(FEATURE_SETS)
    colors = {"ridge": "#2563eb", "extra_trees": "#f97316"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, target in zip(axes, TARGETS, strict=True):
        subset = aggregate.loc[aggregate["target"].eq(target)]
        for model_name in ["ridge", "extra_trees"]:
            model_rows = subset.loc[subset["model"].eq(model_name)].set_index("feature_set").reindex(order)
            ax.errorbar(
                range(len(order)),
                model_rows["mean_cv_mae"],
                yerr=model_rows["std_cv_mae"],
                marker="o",
                capsize=3,
                label=model_name,
                color=colors[model_name],
            )
        ax.set_xticks(range(len(order)), [name.split("_")[0] for name in order])
        ax.set_title(target)
        ax.set_xlabel("Incremental feature set")
        ax.set_ylabel("Leave-one-cycle-out MAE")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Feature ablation on development cycles only")
    fig.savefig(FIGURE_DIR / "01_cycle_cv_feature_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_manifest() -> None:
    manifest = {
        "development_cycles": DEVELOPMENT_CYCLES,
        "benchmark_cycle": BENCHMARK_CYCLE,
        "selection_metric": "mean leave-one-cycle-out unweighted MAE",
        "selection_rules": {
            "minimum_cv_mae": "lowest development-cycle mean MAE",
            "simplest_within_1_percent": (
                "fewest model features within 1% of the lowest development-cycle mean MAE"
            ),
        },
        "comparison_cohort": "complete cases for every F0-F6 source variable within each target",
        "recommended_v0_1_raw_inputs": [
            "sex_code",
            "age_years",
            "height_cm",
            "weight_kg",
            "waist_circumference_cm",
        ],
        "recommended_v0_1_feature_set": "F2_add_waist",
        "feature_sets": FEATURE_SETS,
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA_PATH)

    folds = run_cycle_validation(data)
    aggregate = aggregate_cycle_results(folds)
    selected = selected_configurations(aggregate)
    benchmark, predictions = fit_and_score_benchmark(data, selected)
    subgroups = subgroup_metrics(predictions)

    folds.to_csv(OUTPUT_DIR / "cycle_cv_fold_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "cycle_cv_aggregate.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected_configurations.csv", index=False)
    benchmark.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    predictions.to_parquet(OUTPUT_DIR / "benchmark_predictions.parquet", index=False)
    subgroups.to_csv(OUTPUT_DIR / "benchmark_subgroup_metrics.csv", index=False)
    write_manifest()
    plot_cv_results(aggregate)

    print("Selected using development cycles only:")
    print(
        selected[
            [
                "target",
                "selection_rule",
                "feature_set",
                "model",
                "mean_cv_mae",
                "std_cv_mae",
            ]
        ].to_string(index=False)
    )
    print("\nHeld-out-cycle benchmark:")
    print(
        benchmark[
            ["target", "selection_rule", "feature_set", "model", "mae", "rmse", "r2"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

