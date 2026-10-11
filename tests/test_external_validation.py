import json

import pandas as pd
import pytest

from body_soft_sensor.external_validation import UNITS, evaluate_external
from body_soft_sensor.inference import ModelRegistry
from test_model_bundles import write_bundle


def metadata():
    return {"source_id": "synthetic-test", "population": "synthetic adults",
            "reference": "fixture only", "measurement_protocol": "synthetic",
            "measurement_method": "DXA", "lean_mass_definition": "excludes_bone_mineral_content",
            "body_region": "total_body_including_head",
            "units": UNITS.copy()}


def data():
    return pd.DataFrame([dict(participant_id="anonymous-a", sex="female", age_years=30,
        height_cm=160, weight_kg=64.1, waist_circumference_cm=75,
        body_fat_percent=24, total_fat_mass_kg=24, total_lean_mass_kg=24)])


def test_external_report_is_aggregate_and_reports_scope_rejections(tmp_path):
    good = data()
    bad = data().assign(participant_id="anonymous-b", age_years=70)
    report = evaluate_external(pd.concat([good, bad]), metadata(), ModelRegistry(write_bundle(tmp_path)))
    assert report["accepted_input_rows"] == 1
    assert report["rejected_rows"]["invalid_or_outside_model_scope"] == 1
    assert report["targets"]["body_fat_percent"]["mae"] == 1
    assert report["targets"]["body_fat_percent"]["coverage"] == 1
    assert report["targets"]["body_fat_percent"]["small_sample"]
    assert "anonymous-a" not in json.dumps(report)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", ["duplicate", "non_numeric", "invalid_label", "wrong_method", "wrong_definition", "wrong_region"])
def test_external_contract_rejects_incompatible_data(tmp_path, mutation):
    frame, meta = data(), metadata()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame])
    elif mutation == "non_numeric":
        frame["height_cm"] = "unknown"
    elif mutation == "invalid_label":
        frame["body_fat_percent"] = 101
    elif mutation == "wrong_method":
        meta["measurement_method"] = "BIA"
    elif mutation == "wrong_region":
        meta["body_region"] = "subtotal_without_head"
    else:
        meta["lean_mass_definition"] = "includes_bone_mineral_content"
    with pytest.raises(ValueError):
        evaluate_external(frame, meta, ModelRegistry(write_bundle(tmp_path)))


def test_no_supported_rows_is_explicit_not_successful_metrics(tmp_path):
    frame = data().assign(age_years=70)
    report = evaluate_external(frame, metadata(), ModelRegistry(write_bundle(tmp_path)))
    assert report["accepted_input_rows"] == 0
    assert report["targets"]["body_fat_percent"]["status"] == "no_evaluable_labels"


def test_fractional_age_does_not_bypass_product_input_contract(tmp_path):
    report = evaluate_external(data().assign(age_years=30.5), metadata(), ModelRegistry(write_bundle(tmp_path)))
    assert report["accepted_input_rows"] == 0


def test_whitespace_in_identifiers_cannot_create_extra_participants(tmp_path):
    frame = pd.concat([data(), data().assign(participant_id=" anonymous-a ")])
    with pytest.raises(ValueError, match="one independent record"):
        evaluate_external(frame, metadata(), ModelRegistry(write_bundle(tmp_path)))
