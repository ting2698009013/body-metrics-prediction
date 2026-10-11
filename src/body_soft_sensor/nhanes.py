"""Load and prepare NHANES 2017-2018 body-composition data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEMO_COLUMNS = [
    "SEQN",
    "RIAGENDR",
    "RIDAGEYR",
    "RIDRETH3",
    "WTMEC2YR",
]

BODY_COLUMNS = [
    "SEQN",
    "BMXWT",
    "BMXHT",
    "BMXBMI",
    "BMXARML",
    "BMXARMC",
    "BMXWAIST",
]

OPTIONAL_BODY_COLUMNS = ["BMXHIP"]

DXA_COLUMNS = [
    "SEQN",
    "DXAEXSTS",
    "DXDTOFAT",
    "DXDTOLE",
    "DXDTOPF",
]

RENAME_MAP = {
    "SEQN": "participant_id",
    "RIAGENDR": "sex_code",
    "RIDAGEYR": "age_years",
    "RIDRETH3": "race_ethnicity_code",
    "WTMEC2YR": "exam_sample_weight",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "BMXARML": "upper_arm_length_cm",
    "BMXARMC": "arm_circumference_cm",
    "BMXWAIST": "waist_circumference_cm",
    "BMXHIP": "hip_circumference_cm",
    "DXAEXSTS": "dxa_status",
    "DXDTOFAT": "total_fat_mass_g",
    "DXDTOLE": "total_lean_mass_g",
    "DXDTOPF": "body_fat_percent",
}

MODEL_COLUMNS = [
    "cycle",
    "participant_id",
    "sex_code",
    "age_years",
    "race_ethnicity_code",
    "exam_sample_weight",
    "weight_kg",
    "height_cm",
    "bmi",
    "upper_arm_length_cm",
    "arm_circumference_cm",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "body_fat_percent",
    "total_fat_mass_kg",
    "total_lean_mass_kg",
]


def validate_raw_files(raw_dir: Path, suffix: str) -> None:
    """Fail early if one of the expected public-use files is absent."""

    names = tuple(f"{component}_{suffix}.xpt" for component in ("DEMO", "BMX", "DXX"))
    missing = [name for name in names if not (raw_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing NHANES source file(s): {joined}")


def load_raw_tables(
    raw_dir: Path, suffix: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three SAS transport files and retain documented variables."""

    validate_raw_files(raw_dir, suffix)
    demo = pd.read_sas(raw_dir / f"DEMO_{suffix}.xpt", format="xport")[DEMO_COLUMNS]
    body_raw = pd.read_sas(raw_dir / f"BMX_{suffix}.xpt", format="xport")
    body_columns = BODY_COLUMNS + [
        column for column in OPTIONAL_BODY_COLUMNS if column in body_raw.columns
    ]
    body = body_raw[body_columns].copy()
    for column in OPTIONAL_BODY_COLUMNS:
        if column not in body.columns:
            body[column] = pd.NA
    dxa = pd.read_sas(raw_dir / f"DXX_{suffix}.xpt", format="xport")[DXA_COLUMNS]
    return demo, body, dxa


def prepare_body_composition_cycle(raw_dir: Path, suffix: str, cycle: str) -> pd.DataFrame:
    """Create one adult participant row with easy inputs and DXA targets."""

    demo, body, dxa = load_raw_tables(raw_dir, suffix)
    merged = demo.merge(body, on="SEQN", how="inner", validate="one_to_one")
    merged = merged.merge(dxa, on="SEQN", how="inner", validate="one_to_one")
    merged = merged.rename(columns=RENAME_MAP)

    # DXA was only offered to ages 8-59. The product prototype targets adults.
    merged = merged.loc[
        merged["age_years"].between(18, 59) & merged["dxa_status"].eq(1)
    ].copy()

    merged["total_fat_mass_kg"] = merged["total_fat_mass_g"] / 1000.0
    merged["total_lean_mass_kg"] = merged["total_lean_mass_g"] / 1000.0
    merged["cycle"] = cycle

    merged = merged[MODEL_COLUMNS].sort_values("participant_id").reset_index(drop=True)
    return merged


def prepare_body_composition_dataset(raw_dir: Path) -> pd.DataFrame:
    """Backward-compatible helper for the initially downloaded 2017-2018 cycle."""

    return prepare_body_composition_cycle(raw_dir, suffix="J", cycle="2017-2018")


def prepare_all_cycles(
    raw_root: Path,
    cycles: dict[str, str],
) -> pd.DataFrame:
    """Append harmonized cycle tables while preserving the source cycle."""

    frames = [
        prepare_body_composition_cycle(raw_root / cycle, suffix, cycle)
        for cycle, suffix in cycles.items()
    ]
    return pd.concat(frames, ignore_index=True)


def build_quality_report(data: pd.DataFrame) -> dict[str, object]:
    """Return a JSON-serializable snapshot of data suitability."""

    target_columns = [
        "body_fat_percent",
        "total_fat_mass_kg",
        "total_lean_mass_kg",
    ]
    feature_columns = [
        "sex_code",
        "age_years",
        "weight_kg",
        "height_cm",
        "arm_circumference_cm",
        "waist_circumference_cm",
        "hip_circumference_cm",
    ]

    target_summary = {}
    for column in target_columns:
        series = data[column]
        target_summary[column] = {
            "non_missing": int(series.notna().sum()),
            "mean": float(series.mean()),
            "standard_deviation": float(series.std()),
            "minimum": float(series.min()),
            "median": float(series.median()),
            "maximum": float(series.max()),
        }

    return {
        "rows": int(len(data)),
        "unique_participants": int(data["participant_id"].nunique()),
        "cycle_counts": {
            str(key): int(value)
            for key, value in data["cycle"].value_counts().sort_index().items()
        },
        "age_range": [int(data["age_years"].min()), int(data["age_years"].max())],
        "sex_counts": {
            str(int(key)): int(value)
            for key, value in data["sex_code"].value_counts().sort_index().items()
        },
        "missing_fraction": {
            column: float(data[column].isna().mean())
            for column in feature_columns + target_columns
        },
        "targets": target_summary,
    }


def write_quality_report(report: dict[str, object], output_path: Path) -> None:
    """Write the quality report in a reviewable text format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
