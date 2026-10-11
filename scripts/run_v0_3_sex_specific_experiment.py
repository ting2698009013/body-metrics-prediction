"""Test whether separate male/female point models justify a v0.3 upgrade."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from body_soft_sensor.modeling import RAW_MODEL_INPUTS

from train_v0_1_models import (
    DATA_PATH,
    DEVELOPMENT_CYCLES,
    OUTPUT_DIR as MODEL_SELECTION_DIR,
    analysis_data,
    candidate_configs,
    make_estimator,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "v0_3_sex_specific"
MIN_RELATIVE_IMPROVEMENT = 0.01
MAX_CYCLE_TOLERANCE = 0.02
MAX_SEX_TOLERANCE = 0.01


def predictions_for_strategy(
    development: pd.DataFrame,
    target: str,
    config: dict[str, object],
    validation_cycle: str,
    strategy: str,
) -> pd.DataFrame:
    train = development.loc[development["cycle"].ne(validation_cycle)]
    validation = development.loc[development["cycle"].eq(validation_cycle)].copy()
    prediction = np.empty(len(validation), dtype=float)
    if strategy == "pooled":
        model = make_estimator(config)
        model.fit(train[RAW_MODEL_INPUTS], train[target])
        prediction[:] = model.predict(validation[RAW_MODEL_INPUTS])
    elif strategy == "sex_specific":
        for sex_code in [1, 2]:
            train_mask = train["sex_code"].eq(sex_code)
            validation_mask = validation["sex_code"].eq(sex_code)
            model = make_estimator(config)
            model.fit(train.loc[train_mask, RAW_MODEL_INPUTS], train.loc[train_mask, target])
            prediction[validation_mask.to_numpy()] = model.predict(
                validation.loc[validation_mask, RAW_MODEL_INPUTS]
            )
    else:
        raise ValueError(strategy)
    return pd.DataFrame(
        {
            "target": target,
            "strategy": strategy,
            "validation_cycle": validation_cycle,
            "sex_code": validation["sex_code"].to_numpy(),
            "actual": validation[target].to_numpy(),
            "predicted": prediction,
            "absolute_error": np.abs(validation[target].to_numpy() - prediction),
        }
    )


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = (
        predictions.groupby(["target", "strategy", "validation_cycle"], observed=True)
        .agg(rows=("actual", "size"), mae=("absolute_error", "mean"))
        .reset_index()
    )
    sexes = (
        predictions.groupby(["target", "strategy", "sex_code"], observed=True)
        .agg(rows=("actual", "size"), mae=("absolute_error", "mean"))
        .reset_index()
    )
    return cycles, sexes


def decide(cycles: pd.DataFrame, sexes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in cycles["target"].unique():
        target_cycles = cycles.loc[cycles["target"].eq(target)]
        target_sexes = sexes.loc[sexes["target"].eq(target)]
        pooled = target_cycles.loc[target_cycles["strategy"].eq("pooled")]
        separate = target_cycles.loc[target_cycles["strategy"].eq("sex_specific")]
        pooled_mean = float(pooled["mae"].mean())
        separate_mean = float(separate["mae"].mean())
        relative_improvement = (pooled_mean - separate_mean) / pooled_mean
        max_cycle_ratio = float(separate["mae"].max() / pooled["mae"].max())

        sex_pivot = target_sexes.pivot(index="sex_code", columns="strategy", values="mae")
        worst_sex_ratio = float(
            (sex_pivot["sex_specific"] / sex_pivot["pooled"]).max()
        )
        accepted = (
            relative_improvement >= MIN_RELATIVE_IMPROVEMENT
            and max_cycle_ratio <= 1 + MAX_CYCLE_TOLERANCE
            and worst_sex_ratio <= 1 + MAX_SEX_TOLERANCE
        )
        rows.append(
            {
                "target": target,
                "pooled_mean_cycle_mae": pooled_mean,
                "sex_specific_mean_cycle_mae": separate_mean,
                "relative_improvement": relative_improvement,
                "max_cycle_mae_ratio": max_cycle_ratio,
                "worst_sex_mae_ratio": worst_sex_ratio,
                "accepted": accepted,
                "decision": "upgrade" if accepted else "keep_v0.2_pooled_model",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(DATA_PATH)
    winners = pd.read_csv(MODEL_SELECTION_DIR / "selected_models.csv")
    config_lookup = {item["config_id"]: item for item in candidate_configs()}
    predictions = []
    for winner in winners.itertuples(index=False):
        target = winner.target
        development = analysis_data(data, target)
        development = development.loc[development["cycle"].isin(DEVELOPMENT_CYCLES)]
        for cycle in DEVELOPMENT_CYCLES:
            for strategy in ["pooled", "sex_specific"]:
                predictions.append(
                    predictions_for_strategy(
                        development,
                        target,
                        config_lookup[winner.config_id],
                        cycle,
                        strategy,
                    )
                )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    cycles, sexes = summarize(prediction_frame)
    decision = decide(cycles, sexes)
    cycles.to_csv(OUTPUT_DIR / "cycle_metrics.csv", index=False)
    sexes.to_csv(OUTPUT_DIR / "sex_metrics.csv", index=False)
    decision.to_csv(OUTPUT_DIR / "decision.csv", index=False)
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
