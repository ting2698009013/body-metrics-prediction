import hashlib
import json
from pathlib import Path

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from body_soft_sensor import __version__
from body_soft_sensor.inference import (
    BodyMeasurements, EXPECTED_TARGETS, ModelRegistry, resolve_model_dir,
)
from body_soft_sensor.modeling import RAW_MODEL_INPUTS
from body_soft_sensor.webapp import create_app


def write_bundle(tmp_path, version="v0.2"):
    directory = tmp_path / "models"
    directory.mkdir()
    files = {}
    for target in EXPECTED_TARGETS:
        bundle = {
            "model": DummyRegressor(strategy="constant", constant=25).fit([[0]], [25]),
            "raw_inputs": RAW_MODEL_INPUTS, "target": target,
            "training_cycles": ["fixture"], "model_selection": {"family": "fixture"},
            "prediction_interval": {
                "nominal_coverage": 0.9, "calibration_key": "sex_bmi_group",
                "group_half_widths": {"female|25-29.9": 4},
                "global_fallback_half_width": 5,
            },
        }
        if version != "v0.1":
            bundle["model_version"] = version
        path = directory / f"{version.replace('.', '_')}__{target}.joblib"
        joblib.dump(bundle, path)
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size}
    manifest = {"model_version": version, "runtime": {"scikit_learn": sklearn.__version__},
                "models": files}
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.mark.parametrize("version", ["v0.1", "v0.2"])
def test_disk_bundle_roundtrip_and_api_snapshot(tmp_path, version):
    registry = ModelRegistry(write_bundle(tmp_path, version))
    payload = dict(sex="female", age_years=30, height_cm=160.0,
                   weight_kg=64.1, waist_circumference_cm=75.0)
    with TestClient(create_app(registry)) as client:
        data = client.post("/api/v1/predict", json=payload).json()
        assert data["inputs"] == payload
        assert data["software_version"] == __version__
        assert data["version"] == data["model_version"] == version
        assert data["generated_at"].endswith("+00:00")
        assert data["results"][0]["interval"]["half_width"] == 4
        assert data["results"][2]["label"] == "瘦组织量"
        assert "不含骨矿物质" in data["results"][2]["description"]
        assert client.get("/assets/app.js").status_code == 200
    assert registry.integrity_verified


def test_corrupted_model_is_rejected_before_deserialization(tmp_path, monkeypatch):
    directory = write_bundle(tmp_path)
    path = next(directory.glob("*.joblib"))
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    calls = []
    original_load = joblib.load
    def checked_load(path):
        calls.append(Path(path))
        return original_load(path)
    monkeypatch.setattr(joblib, "load", checked_load)
    with pytest.raises(ValueError, match="checksum"):
        ModelRegistry(directory)
    assert path not in calls


def test_runtime_mismatch_is_rejected_before_load(tmp_path, monkeypatch):
    directory = write_bundle(tmp_path)
    monkeypatch.setattr(sklearn, "__version__", "incompatible")
    with pytest.raises(ValueError, match="requires scikit-learn"):
        ModelRegistry(directory)


def test_installed_location_can_be_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("BODY_SOFT_SENSOR_MODEL_DIR", str(tmp_path))
    assert resolve_model_dir() == tmp_path


def test_unreported_model_metrics_do_not_inherit_historical_scores(tmp_path):
    info = ModelRegistry(write_bundle(tmp_path)).model_info()
    assert all(target["benchmark"] == {} for target in info["targets"])
    assert info["benchmark_cycle"] is None
    assert info["scope"] == "Not declared in model manifest"


def test_synthetic_dataset_is_identified_in_model_info_and_prediction(tmp_path):
    directory = write_bundle(tmp_path)
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = {"synthetic": True, "scope": "toy fixture", "benchmark_cycle": "toy test"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry = ModelRegistry(directory)
    assert registry.model_info()["synthetic_data"] is True
    assert registry.model_info()["scope"] == "toy fixture"
    assert "纯合成" in registry.predict(BodyMeasurements("male", 30, 175, 75, 85))["disclaimer"]
