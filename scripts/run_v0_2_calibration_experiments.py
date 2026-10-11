"""Compare transparent prediction-interval calibration schemes for v0.2.

The point models remain locked to the v0.1 winners. Calibration choices are
made from leave-one-cycle-out residuals on 2011--2016; 2017--2018 is used only
for the final benchmark report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_soft_sensor.modeling import RAW_MODEL_INPUTS, finite_sample_absolute_quantile

from train_v0_1_models import (
    BENCHMARK_CYCLE,
    BMI_GROUP_EDGES,
    BMI_GROUP_LABELS,
    DATA_PATH,
    DEVELOPMENT_CYCLES,
    INTERVAL_COVERAGE,
    OUTPUT_DIR as V01_OUTPUT_DIR,
    analysis_data,
    candidate_configs,
    make_estimator,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "v0_2_calibration"
CALIBRATION_METHODS = ["global", "bmi_group", "sex", "sex_bmi_group"]
SEX_LABELS = {1: "male", 2: "female"}


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["sex"] = result["sex_code"].map(SEX_LABELS)
    bmi = result["weight_kg"] / (result["height_cm"] / 100.0) ** 2
    result["bmi"] = bmi
    result["bmi_group"] = pd.cut(
        bmi,
        bins=BMI_GROUP_EDGES,
        labels=BMI_GROUP_LABELS,
        right=False,
    ).astype(object)
    result["sex_bmi_group"] = result["sex"] + "|" + result["bmi_group"].astype(str)
    return result


def make_oof_and_benchmark_predictions(
    data: pd.DataFrame, winners: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = {item["config_id"]: item for item in candidate_configs()}
    oof_frames: list[pd.DataFrame] = []
    benchmark_frames: list[pd.DataFrame] = []
    for winner in winners.itertuples(index=False):
        target_data = analysis_data(data, winner.target)
        development = target_data.loc[
            target_data["cycle"].isin(DEVELOPMENT_CYCLES)
        ].copy()
        benchmark = target_data.loc[target_data["cycle"].eq(BENCHMARK_CYCLE)].copy()
        config = configs[winner.config_id]

        oof_prediction = np.empty(len(development), dtype=float)
        for validation_cycle in DEVELOPMENT_CYCLES:
            train_mask = development["cycle"].ne(validation_cycle)
            validation_mask = development["cycle"].eq(validation_cycle)
            model = make_estimator(config)
            model.fit(
                development.loc[train_mask, RAW_MODEL_INPUTS],
                development.loc[train_mask, winner.target],
            )
            oof_prediction[validation_mask.to_numpy()] = model.predict(
                development.loc[validation_mask, RAW_MODEL_INPUTS]
            )

        development = add_groups(development)
        development["target"] = winner.target
        development["actual"] = development[winner.target]
        development["predicted"] = oof_prediction
        development["absolute_error"] = np.abs(
            development["actual"] - development["predicted"]
        )
        oof_frames.append(development)

        final_model = make_estimator(config)
        final_model.fit(development[RAW_MODEL_INPUTS], development[winner.target])
        benchmark = add_groups(benchmark)
        benchmark["target"] = winner.target
        benchmark["actual"] = benchmark[winner.target]
        benchmark["predicted"] = final_model.predict(benchmark[RAW_MODEL_INPUTS])
        benchmark["absolute_error"] = np.abs(
            benchmark["actual"] - benchmark["predicted"]
        )
        benchmark_frames.append(benchmark)

    keep = [
        "participant_id",
        "cycle",
        "sex_code",
        "sex",
        "bmi",
        "bmi_group",
        "sex_bmi_group",
        "target",
        "actual",
        "predicted",
        "absolute_error",
    ]
    return (
        pd.concat(oof_frames, ignore_index=True)[keep],
        pd.concat(benchmark_frames, ignore_index=True)[keep],
    )


def fit_widths(oof: pd.DataFrame, method: str) -> dict[str, dict[str, float] | float]:
    result: dict[str, dict[str, float] | float] = {}
    for target, target_frame in oof.groupby("target", observed=True):
        fallback = finite_sample_absolute_quantile(
            target_frame["absolute_error"], INTERVAL_COVERAGE
        )
        if method == "global":
            widths: dict[str, float] = {"__global__": fallback}
        else:
            widths = {
                str(group): finite_sample_absolute_quantile(
                    group_frame["absolute_error"], INTERVAL_COVERAGE
                )
                for group, group_frame in target_frame.groupby(method, observed=True)
            }
        result[target] = {"fallback": fallback, "groups": widths}
    return result


def apply_widths(
    frame: pd.DataFrame,
    method: str,
    widths: dict[str, dict[str, float] | float],
) -> pd.DataFrame:
    result = frame.copy()
    half_widths: list[float] = []
    for row in result.itertuples(index=False):
        target_widths = widths[row.target]
        fallback = float(target_widths["fallback"])
        groups = target_widths["groups"]
        key = "__global__" if method == "global" else str(getattr(row, method))
        half_widths.append(float(groups.get(key, fallback)))
    result["method"] = method
    result["half_width"] = half_widths
    result["covered"] = result["absolute_error"] <= result["half_width"]
    return result


def summarize(evaluated: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        evaluated.groupby(["method", "target"], observed=True)
        .agg(
            rows=("actual", "size"),
            coverage=("covered", "mean"),
            mean_half_width=("half_width", "mean"),
        )
        .reset_index()
    )
    overall.insert(0, "split", split)

    subgroup = (
        evaluated.groupby(
            ["method", "target", "sex_bmi_group"], observed=True
        )
        .agg(
            rows=("actual", "size"),
            coverage=("covered", "mean"),
            mean_half_width=("half_width", "mean"),
        )
        .reset_index()
    )
    subgroup.insert(0, "split", split)
    return overall, subgroup


def selection_table(
    development_overall: pd.DataFrame, development_subgroups: pd.DataFrame
) -> pd.DataFrame:
    subgroup_scores = (
        development_subgroups.assign(
            absolute_coverage_error=lambda x: np.abs(
                x["coverage"] - INTERVAL_COVERAGE
            )
        )
        .groupby(["method", "target"], observed=True)
        .agg(
            worst_subgroup_coverage=("coverage", "min"),
            max_subgroup_coverage=("coverage", "max"),
            max_subgroup_calibration_error=("absolute_coverage_error", "max"),
        )
        .reset_index()
    )
    result = development_overall.merge(
        subgroup_scores, on=["method", "target"], how="left"
    )
    # The product serves all three targets, so the method is selected jointly.
    joint = (
        result.groupby("method", observed=True)
        .agg(
            max_subgroup_calibration_error=("max_subgroup_calibration_error", "max"),
            worst_subgroup_coverage=("worst_subgroup_coverage", "min"),
            mean_half_width=("mean_half_width", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
        .sort_values(
            ["max_subgroup_calibration_error", "mean_half_width"],
            ascending=[True, True],
        )
    )
    joint["selected"] = False
    joint.loc[joint.index[0], "selected"] = True
    return joint.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA_PATH)
    winners = pd.read_csv(V01_OUTPUT_DIR / "selected_models.csv")
    oof, benchmark = make_oof_and_benchmark_predictions(data, winners)

    evaluated_oof = []
    evaluated_calibration_cv = []
    evaluated_benchmark = []
    all_widths = {}
    for method in CALIBRATION_METHODS:
        widths = fit_widths(oof, method)
        all_widths[method] = widths
        evaluated_oof.append(apply_widths(oof, method, widths))
        evaluated_benchmark.append(apply_widths(benchmark, method, widths))
        # Select the partitioning scheme without evaluating a residual against
        # a quantile that was fitted using that same survey cycle.
        for validation_cycle in DEVELOPMENT_CYCLES:
            calibration = oof.loc[oof["cycle"].ne(validation_cycle)]
            validation = oof.loc[oof["cycle"].eq(validation_cycle)]
            fold_widths = fit_widths(calibration, method)
            evaluated_calibration_cv.append(
                apply_widths(validation, method, fold_widths)
            )

    development_eval = pd.concat(evaluated_oof, ignore_index=True)
    calibration_cv_eval = pd.concat(evaluated_calibration_cv, ignore_index=True)
    benchmark_eval = pd.concat(evaluated_benchmark, ignore_index=True)
    fitted_overall, fitted_subgroups = summarize(
        development_eval, "development_fitted_calibration"
    )
    dev_overall, dev_subgroups = summarize(
        calibration_cv_eval, "development_calibration_cv"
    )
    test_overall, test_subgroups = summarize(benchmark_eval, "benchmark_2017_2018")
    selection = selection_table(dev_overall, dev_subgroups)

    pd.concat([fitted_overall, dev_overall, test_overall], ignore_index=True).to_csv(
        OUTPUT_DIR / "overall_metrics.csv", index=False
    )
    pd.concat([fitted_subgroups, dev_subgroups, test_subgroups], ignore_index=True).to_csv(
        OUTPUT_DIR / "sex_bmi_subgroup_metrics.csv", index=False
    )
    selection.to_csv(OUTPUT_DIR / "method_selection.csv", index=False)
    benchmark_eval.to_parquet(OUTPUT_DIR / "benchmark_predictions.parquet", index=False)
    (OUTPUT_DIR / "calibration_widths.json").write_text(
        json.dumps(all_widths, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected = selection.loc[selection["selected"], "method"].iat[0]
    print("Development-only selection:")
    print(selection.to_string(index=False))
    print(f"\nSelected: {selected}")
    print("\nHeld-out 2017-2018 benchmark:")
    print(
        test_overall.loc[test_overall["method"].eq(selected)].to_string(index=False)
    )


if __name__ == "__main__":
    main()
