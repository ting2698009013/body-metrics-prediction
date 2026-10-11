"""Evaluate an independent DXA cohort without fitting or replacing any model."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from body_soft_sensor import __version__
from body_soft_sensor.inference import BodyMeasurements, EXPECTED_TARGETS, ModelRegistry, bmi_group
from body_soft_sensor.modeling import RAW_MODEL_INPUTS

INPUTS = ["sex", "age_years", "height_cm", "weight_kg", "waist_circumference_cm"]
UNITS = {"age_years": "years", "height_cm": "cm", "weight_kg": "kg",
         "waist_circumference_cm": "cm", "body_fat_percent": "percent",
         "total_fat_mass_kg": "kg", "total_lean_mass_kg": "kg"}


def validate_metadata(metadata: dict[str, Any]) -> None:
    for field in ("source_id", "population", "reference", "measurement_protocol"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Missing non-empty metadata: {field}")
        if metadata[field].startswith("REPLACE_"):
            raise ValueError(f"Replace the example metadata before evaluation: {field}")
    if metadata.get("measurement_method") != "DXA":
        raise ValueError("Independent labels must be DXA; BIA and formula-derived labels are not interchangeable")
    if metadata.get("body_region") != "total_body_including_head":
        raise ValueError("Labels must include the head; subtotal/trunk/appendicular DXA is not interchangeable")
    if metadata.get("lean_mass_definition") != "excludes_bone_mineral_content":
        raise ValueError("total_lean_mass_kg must exclude bone mineral content")
    if metadata.get("units") != UNITS:
        raise ValueError("Metadata units do not match the required external-data contract")


def wilson_interval(covered: int, count: int) -> list[float]:
    """Descriptive 95% binomial interval (not complex-survey inference)."""
    z = 1.959963984540054
    rate = covered / count
    denominator = 1 + z * z / count
    middle = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, middle - half), min(1.0, middle + half)]


def summarize(actual: np.ndarray, predicted: np.ndarray, widths: np.ndarray, target: str) -> dict[str, Any]:
    error = predicted - actual
    lower = np.maximum(0.0, predicted - widths)
    upper = predicted + widths
    if target == "body_fat_percent":
        upper = np.minimum(100.0, upper)
    hits = int(((actual >= lower) & (actual <= upper)).sum())
    count = len(actual)
    return {"rows": count, "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()),
            "coverage": hits / count, "coverage_95_percent_interval": wilson_interval(hits, count),
            "mean_half_width": float(widths.mean()), "small_sample": count < 100}


def evaluate_external(data: pd.DataFrame, metadata: dict[str, Any], registry: ModelRegistry) -> dict[str, Any]:
    """Return aggregate metrics only; never save participant inputs or identifiers."""
    validate_metadata(metadata)
    required = ["participant_id", *INPUTS, *EXPECTED_TARGETS]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("External dataset is empty")
    if data["participant_id"].isna().any() or data["participant_id"].astype(str).str.strip().eq("").any():
        raise ValueError("participant_id must be a non-empty pseudonymous identifier")
    normalized_ids = data["participant_id"].astype(str).str.strip()
    if normalized_ids.duplicated().any():
        raise ValueError("Expected one independent record per participant; repeated/imputed records need a separate design")
    work = data[required].copy()
    work["participant_id"] = normalized_ids
    for name in INPUTS[1:] + EXPECTED_TARGETS:
        converted = pd.to_numeric(work[name], errors="coerce")
        if (work[name].notna() & converted.isna()).any():
            raise ValueError(f"Non-numeric values in {name}")
        work[name] = converted
    valid = []
    rejected = {"missing_inputs": 0, "invalid_or_outside_model_scope": 0}
    for row in work.to_dict("records"):
        if any(pd.isna(row[name]) for name in INPUTS):
            rejected["missing_inputs"] += 1
            continue
        try:
            measurement = BodyMeasurements(**{name: row[name] for name in INPUTS})
            measurement.validate_model_scope()
        except (ValueError, TypeError, OverflowError):
            rejected["invalid_or_outside_model_scope"] += 1
            continue
        row["sex_code"] = measurement.sex_code
        row["bmi_group"] = bmi_group(measurement.bmi)
        valid.append(row)
    accepted = pd.DataFrame(valid)
    report: dict[str, Any] = {
        "software_version": __version__, "model_version": registry.version,
        "source": metadata, "input_rows": len(work), "accepted_input_rows": len(valid),
        "rejected_rows": rejected, "targets": {}, "subgroups": [],
        "limitations": ["Descriptive unweighted metrics; no complex survey correction.",
                        "No automatic pass/fail, recalibration, training or model activation.",
                        "Check measurement protocols and independence from development data before interpreting metrics."],
    }
    for target in EXPECTED_TARGETS:
        numeric = work[target].dropna().to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (target == "body_fat_percent" and (numeric > 100).any()):
            raise ValueError(f"Invalid DXA target values in {target}")
        subset = accepted.dropna(subset=[target]) if len(accepted) else accepted
        if subset.empty:
            report["targets"][target] = {"rows": 0, "status": "no_evaluable_labels"}
            continue
        bundle = registry.bundles[target]
        predicted = np.asarray(bundle["model"].predict(subset[RAW_MODEL_INPUTS]), dtype=float)
        from body_soft_sensor.inference import interval_calibration_group
        interval = bundle["prediction_interval"]
        widths = []
        for row in subset.to_dict("records"):
            measurements = BodyMeasurements(**{name: row[name] for name in INPUTS})
            group = interval_calibration_group(measurements, interval.get("calibration_key", "bmi_group"))
            widths.append(interval["group_half_widths"].get(group, interval["global_fallback_half_width"]))
        widths = np.asarray(widths, dtype=float)
        actual = subset[target].to_numpy(dtype=float)
        report["targets"][target] = summarize(actual, predicted, widths, target)
        report["targets"][target]["missing_labels_in_accepted_rows"] = len(accepted) - len(subset)
        for grouping in ("sex", "bmi_group"):
            for group in sorted(subset[grouping].unique()):
                mask = subset[grouping].eq(group).to_numpy()
                report["subgroups"].append({"target": target, "grouping": grouping, "group": group,
                    **summarize(actual[mask], predicted[mask], widths[mask], target)})
    return report
