"""Build and smoke-test a wheel, then package the active trusted model bundle.

The release archive is local; this script does not publish or change the active model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from body_soft_sensor import __version__
from body_soft_sensor.inference import ModelRegistry, resolve_model_dir

ROOT = Path(__file__).resolve().parents[1]


def smoke_test_wheel(wheel: Path, model_dir: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="body-wheel-") as temporary:
        isolated = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(isolated)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(isolated)
        environment["BODY_SOFT_SENSOR_MODEL_DIR"] = str(model_dir or isolated / "missing-models")
        environment["BODY_SMOKE_EXPECT_MODELS"] = "1" if model_dir is not None else "0"
        environment["BODY_SMOKE_MODEL_VERSION"] = ModelRegistry(model_dir).version if model_dir is not None else ""
        code = '''
import json
import os
from pathlib import Path
import body_soft_sensor
from fastapi.testclient import TestClient
from body_soft_sensor.webapp import create_app
assert Path(body_soft_sensor.__file__).resolve().is_relative_to(Path.cwd())
with TestClient(create_app()) as client:
    assert client.get('/').status_code == 200
    assert client.get('/assets/app.js').status_code == 200
    assert client.get('/assets/styles.css').status_code == 200
    health = client.get('/api/health').json()
    payload = dict(sex='male', age_years=30, height_cm=175.0, weight_kg=75.0, waist_circumference_cm=85.0)
    result = client.post('/api/v1/predict', json=payload)
    if os.environ['BODY_SMOKE_EXPECT_MODELS'] == '1':
        assert health['status'] == 'ok', health
        assert health['models_loaded'] == 3
        assert health['model_version'] == os.environ['BODY_SMOKE_MODEL_VERSION']
        assert result.status_code == 200
        assert len(result.json()['results']) == 3
        assert result.json()['inputs'] == payload
    else:
        assert health['status'] == 'degraded', health
        assert result.status_code == 503
    print(json.dumps(health))
'''
        subprocess.run([sys.executable, "-c", code], cwd=isolated, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--without-models", action="store_true", help="Build a wheel and test its safe no-model state (CI)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "releases")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = (args.model_dir or resolve_model_dir()).resolve()
    registry = None if args.without_models else ModelRegistry(model_dir)
    if registry and not registry.integrity_verified:
        raise ValueError("Release requires a model manifest with verified hashes")
    # Build in a clean source copy so stale local build/ files cannot enter the wheel.
    with tempfile.TemporaryDirectory(prefix="body-release-") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copyfile(ROOT / name, source / name)
        shutil.copytree(ROOT / "src" / "body_soft_sensor", source / "src" / "body_soft_sensor",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "--wheel-dir", str(output)], check=True)
    wheel = output / f"body_soft_sensor-{__version__}-py3-none-any.whl"
    smoke_test_wheel(wheel)
    if registry is None:
        return
    smoke_test_wheel(wheel, model_dir)
    manifest_path = model_dir.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = output / f"body-soft-sensor-{__version__}-model-{registry.version}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(wheel, wheel.name)
        archive.write(manifest_path, "model_bundle/model_manifest.json")
        file_prefix = registry.version.replace(".", "_") + "__"
        for target in registry.targets:
            name = f"{file_prefix}{target}.joblib"
            archive.write(model_dir / name, f"model_bundle/models/{name}")
        archive.writestr("START-HERE.txt", f'''Body composition workbench {__version__}
Model version: {registry.version}
Python 3.11+ required; install from this extracted directory:
  python -m venv .venv
  .venv\\Scripts\\python -m pip install {wheel.name}
  .venv\\Scripts\\body-soft-sensor --model-dir model_bundle/models --check-models
  .venv\\Scripts\\body-soft-sensor --model-dir model_bundle/models
Open http://127.0.0.1:8000
Only load trusted local model bundles. SHA-256 verifies consistency, not publisher identity.
Predictions are population estimates, not medical diagnosis or future change forecasts.
The lean-mass output excludes bone mineral content.
''')
    report = {"software_version": __version__, "model_version": registry.version,
              "wheel_smoke_tests": ["without_models", "active_real_models"],
              "model_manifest": manifest,
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, archive_path)}}
    (output / f"release-{__version__}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified local release: {archive_path}")


if __name__ == "__main__":
    main()
