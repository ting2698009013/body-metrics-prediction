"""Train transparent baseline models and compare two input sets."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nhanes_2011_2018_body_composition.parquet"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
RANDOM_STATE = 42

FEATURE_SETS = {
    "minimal": ["sex_code", "age_years", "height_cm", "weight_kg"],
    "tape_assisted": [
        "sex_code",
        "age_years",
        "height_cm",
        "weight_kg",
        "arm_circumference_cm",
        "waist_circumference_cm",
    ],
}

TARGETS = [
    "body_fat_percent",
    "total_fat_mass_kg",
    "total_lean_mass_kg",
]


def make_estimators() -> dict[str, object]:
    """Return a transparent linear baseline and a nonlinear comparison."""

    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def metric_row(
    *,
    feature_set: str,
    target: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    sample_weight: pd.Series,
) -> dict[str, object]:
    absolute_error = np.abs(y_true.to_numpy() - y_pred)
    return {
        "feature_set": feature_set,
        "target": target,
        "model": model_name,
        "test_rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "weighted_mae": float(
            mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        ),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "absolute_error_p90": float(np.quantile(absolute_error, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute_error, 0.95)),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}. Run prepare_nhanes.py first."
        )

    data = pd.read_parquet(DATA_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for feature_set_name, features in FEATURE_SETS.items():
        for target in TARGETS:
            columns = features + [
                target,
                "cycle",
                "participant_id",
                "race_ethnicity_code",
                "exam_sample_weight",
            ]
            experiment = data[columns].dropna(subset=[target]).copy()
            train = experiment.loc[experiment["cycle"] != "2017-2018"].copy()
            test = experiment.loc[experiment["cycle"] == "2017-2018"].copy()

            fitted_models = {}
            for model_name, estimator in make_estimators().items():
                estimator.fit(train[features], train[target])
                predictions = estimator.predict(test[features])
                rows.append(
                    metric_row(
                        feature_set=feature_set_name,
                        target=target,
                        model_name=model_name,
                        y_true=test[target],
                        y_pred=predictions,
                        sample_weight=test["exam_sample_weight"],
                    )
                )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "participant_id": test["participant_id"].to_numpy(),
                            "sex_code": test["sex_code"].to_numpy(),
                            "race_ethnicity_code": test[
                                "race_ethnicity_code"
                            ].to_numpy(),
                            "feature_set": feature_set_name,
                            "target": target,
                            "model": model_name,
                            "actual": test[target].to_numpy(),
                            "predicted": predictions,
                        }
                    )
                )
                fitted_models[model_name] = estimator

            target_rows = [
                row
                for row in rows
                if row["feature_set"] == feature_set_name and row["target"] == target
            ]
            best_name = min(target_rows, key=lambda row: row["mae"])["model"]
            model_path = MODEL_DIR / f"{feature_set_name}__{target}.joblib"
            joblib.dump(
                {
                    "model": fitted_models[best_name],
                    "features": features,
                    "target": target,
                    "training_cycles": ["2011-2012", "2013-2014", "2015-2016"],
                    "evaluation_cycle": "2017-2018",
                    "evaluation_note": "Temporal external validation on a held-out NHANES cycle.",
                },
                model_path,
            )

    metrics = pd.DataFrame(rows).sort_values(
        ["target", "feature_set", "mae"], ignore_index=True
    )
    metrics.to_csv(ARTIFACT_DIR / "baseline_metrics.csv", index=False)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    best_keys = metrics.loc[
        metrics.groupby(["feature_set", "target"])["mae"].idxmin(),
        ["feature_set", "target", "model"],
    ]
    best_predictions = predictions.merge(
        best_keys,
        on=["feature_set", "target", "model"],
        how="inner",
        validate="many_to_one",
    )
    best_predictions["error"] = best_predictions["predicted"] - best_predictions["actual"]
    best_predictions["absolute_error"] = best_predictions["error"].abs()

    subgroup_frames = []
    for group_column in ["sex_code", "race_ethnicity_code"]:
        grouped = (
            best_predictions.groupby(
                ["feature_set", "target", "model", group_column],
                dropna=False,
            )
            .agg(
                rows=("actual", "size"),
                mean_error=("error", "mean"),
                mae=("absolute_error", "mean"),
                absolute_error_p90=("absolute_error", lambda values: values.quantile(0.90)),
            )
            .reset_index()
            .rename(columns={group_column: "group_value"})
        )
        grouped.insert(3, "group", group_column)
        subgroup_frames.append(grouped)

    subgroup_metrics = pd.concat(subgroup_frames, ignore_index=True)
    subgroup_metrics.to_csv(ARTIFACT_DIR / "subgroup_metrics.csv", index=False)
    best_predictions.to_parquet(ARTIFACT_DIR / "test_predictions.parquet", index=False)

    best = (
        metrics.loc[metrics.groupby(["feature_set", "target"])["mae"].idxmin()]
        .sort_values(["target", "feature_set"])
        .to_dict(orient="records")
    )
    (ARTIFACT_DIR / "baseline_summary.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_columns = ["feature_set", "target", "model", "mae", "rmse", "r2"]
    print(metrics[display_columns].to_string(index=False))
    print(f"Metrics: {ARTIFACT_DIR / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
