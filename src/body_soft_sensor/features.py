"""Feature definitions for the v0.1 body-composition experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = [
    "sex_code",
    "age_years",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_circumference_cm",
    "arm_circumference_cm",
    "upper_arm_length_cm",
]


FEATURE_SETS = {
    "F0_basic": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
    ],
    "F1_add_bmi": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
    ],
    "F2_add_waist": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_circumference_cm",
    ],
    "F3_add_arm_circumference": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_circumference_cm",
        "arm_circumference_cm",
    ],
    "F4_add_arm_length": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_circumference_cm",
        "arm_circumference_cm",
        "upper_arm_length_cm",
    ],
    "F5_add_ratios": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_circumference_cm",
        "arm_circumference_cm",
        "upper_arm_length_cm",
        "waist_to_height_ratio",
        "arm_to_height_ratio",
    ],
    "F6_add_interactions": [
        "female_indicator",
        "age_years",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_circumference_cm",
        "arm_circumference_cm",
        "upper_arm_length_cm",
        "waist_to_height_ratio",
        "arm_to_height_ratio",
        "age_squared",
        "bmi_female_interaction",
        "waist_ratio_female_interaction",
        "arm_ratio_female_interaction",
    ],
}


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create documented, product-obtainable features without touching labels."""

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise KeyError(f"Missing source columns for feature engineering: {missing}")

    result = data.copy()
    result["female_indicator"] = result["sex_code"].eq(2).astype(float)
    result["waist_to_height_ratio"] = (
        result["waist_circumference_cm"] / result["height_cm"]
    )
    result["arm_to_height_ratio"] = (
        result["arm_circumference_cm"] / result["height_cm"]
    )
    result["age_squared"] = result["age_years"] ** 2
    result["bmi_female_interaction"] = result["bmi"] * result["female_indicator"]
    result["waist_ratio_female_interaction"] = (
        result["waist_to_height_ratio"] * result["female_indicator"]
    )
    result["arm_ratio_female_interaction"] = (
        result["arm_to_height_ratio"] * result["female_indicator"]
    )

    engineered = sorted(
        set().union(*FEATURE_SETS.values()) - set(RAW_REQUIRED_COLUMNS)
    )
    if not np.isfinite(result[engineered].to_numpy(dtype=float)).all():
        raise ValueError("Engineered features contain non-finite values")
    return result

