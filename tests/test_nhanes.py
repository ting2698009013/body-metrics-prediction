from pathlib import Path

import pandas as pd
import pytest

import body_soft_sensor.nhanes as nhanes
from body_soft_sensor.nhanes import build_quality_report, prepare_body_composition_dataset


@pytest.fixture
def synthetic_raw(monkeypatch):
    """Fictional records exercise the real join/filter/conversion pipeline."""
    demo = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5], "RIAGENDR": [1, 2, 1, 2, 1],
        "RIDAGEYR": [17, 25, 45, 62, 30], "RIDRETH3": [1] * 5,
        "WTMEC2YR": [1.0] * 5,
    })
    body = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5], "BMXWT": [70] * 5, "BMXHT": [170] * 5,
        "BMXBMI": [24] * 5, "BMXARML": [35] * 5, "BMXARMC": [30] * 5,
        "BMXWAIST": [85] * 5, "BMXHIP": [95] * 5,
    })
    dxa = pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5], "DXAEXSTS": [1, 1, 1, 1, 2],
        "DXDTOFAT": [10000, 20000, 35000, 20000, 20000],
        "DXDTOLE": [45000, 45000, 55000, 45000, 45000],
        "DXDTOPF": [20.0] * 5,
    })
    monkeypatch.setattr(nhanes, "load_raw_tables", lambda _path, _suffix: (demo, body, dxa))
    return demo, body, dxa


def test_prepared_dataset_has_one_row_per_adult(synthetic_raw) -> None:
    data = prepare_body_composition_dataset(Path("unused-synthetic-fixture"))

    assert data["participant_id"].tolist() == [2, 3]
    assert data["participant_id"].is_unique
    assert data["age_years"].between(18, 59).all()
    assert data["body_fat_percent"].notna().all()


def test_mass_units_are_kilograms(synthetic_raw) -> None:
    data = prepare_body_composition_dataset(Path("unused-synthetic-fixture"))

    assert data["total_fat_mass_kg"].tolist() == [20.0, 35.0]
    assert data["total_lean_mass_kg"].tolist() == [45.0, 55.0]


def test_duplicate_join_keys_fail_instead_of_multiplying_people(synthetic_raw):
    demo, _, _ = synthetic_raw
    demo.loc[1, "SEQN"] = 1
    with pytest.raises(pd.errors.MergeError):
        prepare_body_composition_dataset(Path("unused-synthetic-fixture"))


def test_quality_report_is_serializable_in_shape() -> None:
    data = pd.DataFrame(
        {
            "participant_id": [1, 2],
            "cycle": ["2017-2018", "2017-2018"],
            "sex_code": [1, 2],
            "age_years": [20, 30],
            "race_ethnicity_code": [3, 4],
            "exam_sample_weight": [1.0, 1.0],
            "weight_kg": [70.0, 60.0],
            "height_cm": [175.0, 165.0],
            "bmi": [22.9, 22.0],
            "upper_arm_length_cm": [38.0, 36.0],
            "arm_circumference_cm": [30.0, 27.0],
            "waist_circumference_cm": [80.0, 72.0],
            "hip_circumference_cm": [95.0, 92.0],
            "body_fat_percent": [18.0, 25.0],
            "total_fat_mass_kg": [12.0, 15.0],
            "total_lean_mass_kg": [55.0, 43.0],
        }
    )

    report = build_quality_report(data)
    assert report["rows"] == 2
    assert report["unique_participants"] == 2
    assert report["age_range"] == [20, 30]
