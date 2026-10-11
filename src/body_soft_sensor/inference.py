"""Reusable inference service for versioned body-composition model bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import joblib
import pandas as pd
import sklearn

from body_soft_sensor import __version__
from body_soft_sensor.modeling import RAW_MODEL_INPUTS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "model_selection" / "models"


def resolve_model_dir() -> Path:
    """Support a source checkout and installed wheels without guessing cwd."""
    configured = os.environ.get("BODY_SOFT_SENSOR_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    if (ROOT / "pyproject.toml").is_file() and DEFAULT_MODEL_DIR.is_dir():
        return DEFAULT_MODEL_DIR
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
    return base / "body-soft-sensor" / "models"
EXPECTED_TARGETS = ["body_fat_percent", "total_fat_mass_kg", "total_lean_mass_kg"]

TARGET_PRESENTATION = {
    "body_fat_percent": {
        "label": "体脂率",
        "unit": "%",
        "digits": 1,
        "description": "体重中脂肪组织所占的估算比例",
    },
    "total_fat_mass_kg": {
        "label": "脂肪量",
        "unit": "kg",
        "digits": 1,
        "description": "全身脂肪组织的估算质量",
    },
    "total_lean_mass_kg": {
        "label": "瘦组织量",
        "unit": "kg",
        "digits": 1,
        "description": "不含骨矿物质的瘦组织质量，不等同于肌肉量",
    },
}

# Hard limits stay inside the observed development-data support. The inner
# reference band is approximately the 1st–99th percentile and triggers a note.
INPUT_LIMITS = {
    "age_years": {"minimum": 18.0, "maximum": 59.0, "reference": [18.0, 59.0]},
    "height_cm": {"minimum": 135.3, "maximum": 193.3, "reference": [146.6, 188.2]},
    "weight_kg": {"minimum": 32.8, "maximum": 187.5, "reference": [45.6, 144.9]},
    "waist_circumference_cm": {
        "minimum": 55.5,
        "maximum": 176.0,
        "reference": [67.2, 144.4],
    },
}

DERIVED_LIMITS = {
    "bmi": {"minimum": 14.14, "maximum": 69.04, "reference": [17.7, 49.8]},
    "waist_to_height_ratio": {
        "minimum": 0.356,
        "maximum": 1.028,
        "reference": [0.401, 0.851],
    },
}

INPUT_LABELS = {
    "age_years": "年龄",
    "height_cm": "身高",
    "weight_kg": "体重",
    "waist_circumference_cm": "腰围",
}


@dataclass(frozen=True)
class BodyMeasurements:
    sex: str
    age_years: float
    height_cm: float
    weight_kg: float
    waist_circumference_cm: float

    @property
    def sex_code(self) -> int:
        if self.sex == "male":
            return 1
        if self.sex == "female":
            return 2
        raise ValueError("sex must be 'male' or 'female'")

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100.0) ** 2

    def as_model_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sex_code": self.sex_code,
                    "age_years": self.age_years,
                    "height_cm": self.height_cm,
                    "weight_kg": self.weight_kg,
                    "waist_circumference_cm": self.waist_circumference_cm,
                }
            ],
            columns=RAW_MODEL_INPUTS,
        )

    def validate_model_scope(self) -> None:
        self.sex_code
        values = {
            "age_years": self.age_years,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "waist_circumference_cm": self.waist_circumference_cm,
        }
        for name, value in values.items():
            limits = INPUT_LIMITS[name]
            if isinstance(value, bool) or not math.isfinite(value) or not limits["minimum"] <= value <= limits["maximum"]:
                raise ValueError(f"{name} is outside the model training support")
        if self.age_years != int(self.age_years):
            raise ValueError("age_years must be a whole number of years")
        bmi_limits = DERIVED_LIMITS["bmi"]
        if not bmi_limits["minimum"] <= self.bmi <= bmi_limits["maximum"]:
            raise ValueError("BMI is outside the model training support")
        waist_to_height = self.waist_circumference_cm / self.height_cm
        ratio_limits = DERIVED_LIMITS["waist_to_height_ratio"]
        if not ratio_limits["minimum"] <= waist_to_height <= ratio_limits["maximum"]:
            raise ValueError("waist-to-height ratio is outside the model training support")


def bmi_group(bmi: float) -> str:
    if bmi < 25:
        return "<25"
    if bmi < 30:
        return "25-29.9"
    if bmi < 35:
        return "30-34.9"
    if bmi < 40:
        return "35-39.9"
    return "40+"


def bmi_label(bmi: float) -> str:
    if bmi < 18.5:
        return "BMI <18.5"
    if bmi < 25:
        return "BMI 18.5–24.9"
    if bmi < 30:
        return "BMI 25–29.9"
    if bmi < 35:
        return "BMI 30–34.9"
    if bmi < 40:
        return "BMI 35–39.9"
    return "BMI ≥40"


def interval_calibration_group(
    measurements: BodyMeasurements, calibration_key: str
) -> str:
    """Resolve the bundle's calibration partition from public inputs."""

    current_bmi_group = bmi_group(measurements.bmi)
    if calibration_key == "sex_bmi_group":
        return f"{measurements.sex}|{current_bmi_group}"
    if calibration_key == "sex":
        return measurements.sex
    if calibration_key == "global":
        return "__global__"
    if calibration_key == "bmi_group":
        return current_bmi_group
    raise ValueError(f"Unknown calibration method: {calibration_key}")


class ModelRegistry:
    """Load model bundles once and expose a stable product-facing response."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_model_dir()
        self.bundles: dict[str, dict[str, Any]] = {}
        self.version = "v0.1"
        self.integrity_verified = False
        self.artifact_runtime: dict[str, str] = {}
        self.dataset_metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        manifest_path = self.model_dir.parent / "model_manifest.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.artifact_runtime = manifest.get("runtime", {})
            self.dataset_metadata = manifest.get("dataset", {})
            self.version = str(manifest.get("model_version", self.version))
            trained_sklearn = self.artifact_runtime.get("scikit_learn")
            if trained_sklearn and trained_sklearn != sklearn.__version__:
                raise ValueError(f"Model requires scikit-learn {trained_sklearn}; installed {sklearn.__version__}")
        if not re.fullmatch(r"v\d+\.\d+(?:\.\d+)?", self.version):
            raise ValueError("Invalid model version in manifest")
        file_version = self.version.replace(".", "_")
        for target in EXPECTED_TARGETS:
            path = self.model_dir / f"{file_version}__{target}.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Missing trained model: {path}")
            if manifest is not None:
                expected = manifest.get("models", {}).get(path.name)
                if expected is None:
                    raise ValueError(f"Model is missing from artifact manifest: {path.name}")
                if path.stat().st_size != expected.get("size_bytes"):
                    raise ValueError(f"Model size mismatch: {path.name}")
                actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_digest != expected.get("sha256"):
                    raise ValueError(f"Model checksum mismatch: {path.name}")
            bundle = joblib.load(path)
            if bundle.get("target") != target:
                raise ValueError(f"Model target mismatch in {path}")
            bundle_version = bundle.get("model_version", "v0.1")
            if bundle_version != self.version:
                raise ValueError(f"Model version mismatch in {path}")
            if bundle.get("raw_inputs") != RAW_MODEL_INPUTS:
                raise ValueError(f"Model input contract mismatch in {path}")
            self.bundles[target] = bundle
        self.integrity_verified = manifest is not None

    @property
    def targets(self) -> list[str]:
        return list(self.bundles)

    def model_info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_version": self.version,
            "software_version": __version__,
            "raw_inputs": RAW_MODEL_INPUTS,
            "input_limits": INPUT_LIMITS,
            "derived_limits": DERIVED_LIMITS,
            "targets": [
                {
                    "target": target,
                    **TARGET_PRESENTATION[target],
                    "model_family": self.bundles[target]["model_selection"]["family"],
                    "training_cycles": self.bundles[target]["training_cycles"],
                    "interval_coverage": self.bundles[target]["prediction_interval"][
                        "nominal_coverage"
                    ],
                    # Never attach another model's historical score to a new bundle.
                    "benchmark": self.bundles[target].get("benchmark", {}),
                }
                for target in EXPECTED_TARGETS
            ],
            "scope": self.dataset_metadata.get("scope", "Not declared in model manifest"),
            "benchmark_cycle": self.dataset_metadata.get("benchmark_cycle"),
            "synthetic_data": bool(self.dataset_metadata.get("synthetic", False)),
            "artifact_integrity_verified": self.integrity_verified,
            "artifact_runtime": self.artifact_runtime,
            "privacy": "local inference; request data is not persisted",
        }

    def predict(self, measurements: BodyMeasurements) -> dict[str, Any]:
        measurements.validate_model_scope()
        frame = measurements.as_model_frame()
        current_bmi = measurements.bmi
        current_bmi_group = bmi_group(current_bmi)
        results = []

        for target in EXPECTED_TARGETS:
            bundle = self.bundles[target]
            estimate = float(bundle["model"].predict(frame)[0])
            interval = bundle["prediction_interval"]
            calibration_key = interval.get("calibration_key", "bmi_group")
            calibration_group = interval_calibration_group(
                measurements, calibration_key
            )
            half_width = float(
                interval["group_half_widths"].get(
                    calibration_group, interval["global_fallback_half_width"]
                )
            )
            lower = estimate - half_width
            upper = estimate + half_width
            if target == "body_fat_percent":
                lower = max(0.0, lower)
                upper = min(100.0, upper)
            else:
                lower = max(0.0, lower)
            presentation = TARGET_PRESENTATION[target]
            results.append(
                {
                    "target": target,
                    "label": presentation["label"],
                    "unit": presentation["unit"],
                    "digits": presentation["digits"],
                    "description": presentation["description"],
                    "estimate": estimate,
                    "interval": {
                        "nominal_coverage": interval["nominal_coverage"],
                        "lower": lower,
                        "upper": upper,
                        "half_width": half_width,
                        "calibration_group": calibration_group,
                        "calibration_method": calibration_key,
                    },
                    "model_family": bundle["model_selection"]["family"],
                }
            )

        return {
            "version": self.version,
            "model_version": self.version,
            "software_version": __version__,
            "inputs": asdict(measurements),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "derived": {
                "bmi": current_bmi,
                "bmi_label": bmi_label(current_bmi),
                "interval_group": current_bmi_group,
            },
            "results": results,
            "warnings": distribution_warnings(measurements),
            "disclaimer": (
                "纯合成数据演示，输出没有人体测量或医学意义。"
                if self.dataset_metadata.get("synthetic")
                else "结果是群体数据估算，不是医学诊断或 DXA 实测。"
            ),
        }


def distribution_warnings(measurements: BodyMeasurements) -> list[str]:
    values = {
        "age_years": measurements.age_years,
        "height_cm": measurements.height_cm,
        "weight_kg": measurements.weight_kg,
        "waist_circumference_cm": measurements.waist_circumference_cm,
    }
    warnings = []
    for name, value in values.items():
        reference_min, reference_max = INPUT_LIMITS[name]["reference"]
        if value < reference_min or value > reference_max:
            warnings.append(
                f"{INPUT_LABELS[name]}处于训练样本较少的范围，误差可能高于平均水平。"
            )
    bmi = measurements.bmi
    bmi_reference = DERIVED_LIMITS["bmi"]["reference"]
    if bmi < bmi_reference[0] or bmi > bmi_reference[1]:
        warnings.append("当前身高体重组合在训练样本中较少见，误差可能高于平均水平。")
    ratio = measurements.waist_circumference_cm / measurements.height_cm
    ratio_reference = DERIVED_LIMITS["waist_to_height_ratio"]["reference"]
    if not ratio_reference[0] <= ratio <= ratio_reference[1]:
        warnings.append("当前腰围身高组合在训练样本中较少见，误差可能高于平均水平。")
    return warnings

