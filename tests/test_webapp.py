from fastapi.testclient import TestClient

import body_soft_sensor.webapp as webapp
from body_soft_sensor import __version__


class FakeRegistry:
    targets = ["body_fat_percent", "total_fat_mass_kg", "total_lean_mass_kg"]

    def model_info(self) -> dict[str, object]:
        return {"version": "v0.1", "targets": self.targets}

    def predict(self, measurements: object) -> dict[str, object]:
        return {
            "version": "v0.1",
            "derived": {"bmi": measurements.bmi, "bmi_label": "BMI 18.5–24.9"},
            "results": [],
            "warnings": [],
            "disclaimer": "test",
        }


def valid_payload() -> dict[str, object]:
    return {
        "sex": "male",
        "age_years": 30,
        "height_cm": 175.0,
        "weight_kg": 75.0,
        "waist_circumference_cm": 85.0,
    }


def test_web_app_serves_ui_health_and_prediction() -> None:
    with TestClient(webapp.create_app(registry=FakeRegistry())) as client:
        assert client.get("/").status_code == 200
        css = client.get("/assets/styles.css")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert client.get("/api/health").json()["models_loaded"] == 3
        response = client.post("/api/v1/predict", json=valid_payload())
        assert response.status_code == 200
        assert response.json()["derived"]["bmi"] == 24.489795918367346


def test_prediction_rejects_out_of_scope_combination() -> None:
    payload = valid_payload()
    payload.update({"height_cm": 135.3, "weight_kg": 187.5})
    with TestClient(webapp.create_app(registry=FakeRegistry())) as client:
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert "超出模型训练范围" in response.text


def test_prediction_contract_is_strict() -> None:
    payload = valid_payload()
    payload["age_years"] = "30"
    with TestClient(webapp.create_app(registry=FakeRegistry())) as client:
        response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_missing_models_return_safe_degraded_state(monkeypatch) -> None:
    def fail_to_load() -> None:
        raise FileNotFoundError("C:/private/path/model.joblib")

    monkeypatch.setattr(webapp, "ModelRegistry", fail_to_load)
    with TestClient(webapp.create_app()) as client:
        health = client.get("/api/health")
        prediction = client.post("/api/v1/predict", json=valid_payload())
    assert health.json() == {
        "status": "degraded",
        "models_loaded": 0,
        "detail": "模型文件缺失或不兼容。",
        "software_version": __version__,
        "model_version": None,
    }
    assert "private" not in health.text
    assert prediction.status_code == 503

