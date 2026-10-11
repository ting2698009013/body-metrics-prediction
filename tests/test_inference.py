import pytest

from body_soft_sensor.inference import (
    BodyMeasurements,
    bmi_group,
    bmi_label,
    interval_calibration_group,
)


def test_body_measurements_map_public_contract_to_model_inputs() -> None:
    measurements = BodyMeasurements(
        sex="female",
        age_years=30,
        height_cm=160.0,
        weight_kg=64.0,
        waist_circumference_cm=75.0,
    )

    assert measurements.sex_code == 2
    assert measurements.bmi == pytest.approx(25.0)
    assert measurements.as_model_frame().iloc[0].to_dict() == {
        "sex_code": 2.0,
        "age_years": 30.0,
        "height_cm": 160.0,
        "weight_kg": 64.0,
        "waist_circumference_cm": 75.0,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(24.999, "<25"), (25.0, "25-29.9"), (30.0, "30-34.9"), (40.0, "40+")],
)
def test_bmi_group_uses_unrounded_value(value: float, expected: str) -> None:
    assert bmi_group(value) == expected


def test_bmi_label_is_descriptive_not_diagnostic() -> None:
    assert bmi_label(23.0) == "BMI 18.5–24.9"


def test_sex_bmi_interval_group_uses_both_public_inputs() -> None:
    measurements = BodyMeasurements("female", 30, 160, 64.1, 75)
    assert interval_calibration_group(measurements, "sex_bmi_group") == "female|25-29.9"
    assert interval_calibration_group(measurements, "bmi_group") == "25-29.9"


def test_domain_layer_rejects_unsupported_combinations() -> None:
    measurements = BodyMeasurements(
        sex="male",
        age_years=30,
        height_cm=135.3,
        weight_kg=187.5,
        waist_circumference_cm=85.0,
    )
    with pytest.raises(ValueError, match="BMI"):
        measurements.validate_model_scope()

