"""Exercise data boundaries, not just expected numeric implementation details."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v0_3_nested_calibration as experiment  # noqa: E402


TARGET = "body_fat_percent"
CONFIGS = [
    {"config_id": "mean", "family": "fixture", "params": {"offset": 0.0}},
    {"config_id": "shifted", "family": "fixture", "params": {"offset": 0.5}},
]


@pytest.fixture
def development() -> pd.DataFrame:
    rows = []
    for cycle_index, cycle in enumerate(experiment.DEVELOPMENT_CYCLES):
        for participant in range(40):
            rows.append({
                "participant_id": 1000 * cycle_index + participant,
                "cycle": cycle,
                "sex_code": 1 + participant % 2,
                # Decade identifies a cycle so the spy can audit raw fit inputs.
                "age_years": 20 + 10 * cycle_index + participant / 100,
                "height_cm": 170.0,
                "weight_kg": [20, 27, 32, 37, 43][participant % 5] * 1.7**2,
                "waist_circumference_cm": 70.0 + participant,
                TARGET: 21.0 + cycle_index + (participant % 11) / 3,
                "race_ethnicity_code": 1,
                "exam_sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


def spy_factory(events: list):
    class Estimator:
        def __init__(self, config):
            self.offset = config["params"]["offset"]

        def fit(self, features, target):
            events.append(set((features["age_years"] // 10).astype(int)))
            self.mean = float(np.mean(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.mean + self.offset)

    return Estimator


def run_fixture(data, events=None):
    return experiment.run_nested(
        data, CONFIGS, targets=[TARGET],
        estimator_factory=spy_factory([] if events is None else events),
        workers=1,
    )


def test_every_fit_excludes_the_outer_cycle(development):
    events = []
    result = run_fixture(development, events)
    fits_per_outer = len(CONFIGS) * 2 + 1
    assert len(events) == 3 * fits_per_outer
    for outer_index, outer_cycle in enumerate(experiment.DEVELOPMENT_CYCLES):
        fits = events[outer_index * fits_per_outer:(outer_index + 1) * fits_per_outer]
        assert all(outer_index + 2 not in training_cycles for training_cycles in fits)
        assert all(len(training_cycles) == 1 for training_cycles in fits[:-1])
        assert len(fits[-1]) == 2
        fold = result["folds"][outer_index]
        for name in ["point_training_cycles", "config_selection_cycles", "method_selection_cycles", "calibration_fit_cycles"]:
            assert outer_cycle not in fold[name]
    # Each participant is evaluated once, by the complete selected procedure.
    assert result["predictions"]["participant_id"].is_unique
    assert len(result["predictions"]) == len(development)


def test_outer_labels_and_covariates_cannot_change_its_selection_or_widths(development):
    before = run_fixture(development)
    changed = development.copy()
    cycle = experiment.DEVELOPMENT_CYCLES[0]
    mask = changed["cycle"].eq(cycle)
    changed.loc[mask, TARGET] += 1_000_000
    changed.loc[mask, "weight_kg"] += 100
    changed.loc[mask, "waist_circumference_cm"] += 30
    after = run_fixture(changed)
    assert before["folds"][0] == after["folds"][0]
    for table_name in ["config_scores", "method_scores"]:
        pd.testing.assert_frame_equal(
            before[table_name].loc[lambda frame: frame["outer_cycle"].eq(cycle)].reset_index(drop=True),
            after[table_name].loc[lambda frame: frame["outer_cycle"].eq(cycle)].reset_index(drop=True),
        )
    original = before["predictions"].loc[lambda frame: frame["outer_cycle"].eq(cycle)]
    perturbed = after["predictions"].loc[lambda frame: frame["outer_cycle"].eq(cycle)]
    assert not np.array_equal(original["actual"], perturbed["actual"])


def test_historical_benchmark_is_ignored_even_if_invalid(development):
    baseline = run_fixture(development)
    benchmark = development.iloc[:3].copy()
    benchmark["cycle"] = "2017-2018"
    benchmark["sex_code"] = 99
    benchmark[TARGET] = np.nan
    benchmark["height_cm"] = -1
    benchmark["participant_id"] = development.iloc[0]["participant_id"]
    with_benchmark = run_fixture(pd.concat([development, benchmark], ignore_index=True))
    assert baseline["folds"] == with_benchmark["folds"]
    pd.testing.assert_frame_equal(baseline["predictions"], with_benchmark["predictions"])


def test_calibration_selection_split_is_disjoint_and_error_independent(development):
    frame = development.loc[development["cycle"].eq(experiment.DEVELOPMENT_CYCLES[0])].copy()
    frame["absolute_error"] = np.arange(len(frame))
    calibration, validation = experiment.split_calibration_selection(frame)
    assert set(calibration["participant_id"]).isdisjoint(validation["participant_id"])
    assert len(calibration) + len(validation) == len(frame)
    frame["absolute_error"] = -frame["absolute_error"] + 999
    repeated, _ = experiment.split_calibration_selection(frame.sample(frac=1, random_state=2))
    assert set(calibration["participant_id"]) == set(repeated["participant_id"])


def test_duplicate_participants_across_cycles_are_rejected(development):
    development.loc[40, "participant_id"] = development.loc[0, "participant_id"]
    with pytest.raises(ValueError, match="participant"):
        run_fixture(development)
