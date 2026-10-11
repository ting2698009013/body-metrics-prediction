"""Validate a harmonized independent DXA dataset; do not fit or activate models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from body_soft_sensor.external_validation import evaluate_external
from body_soft_sensor.inference import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Harmonized CSV or Parquet")
    parser.add_argument("--metadata", required=True, type=Path, help="Source and measurement contract JSON")
    parser.add_argument("--output", required=True, type=Path, help="Aggregate JSON report, without participant rows")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.data.resolve(), args.metadata.resolve()):
        parser.error("Output must not overwrite an input")
    data = pd.read_parquet(args.data) if args.data.suffix.lower() == ".parquet" else pd.read_csv(args.data, dtype={"participant_id": str})
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    report = evaluate_external(data, metadata, ModelRegistry(args.model_dir))
    report["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Accepted inputs: {report['accepted_input_rows']}/{report['input_rows']}; report: {args.output}")


if __name__ == "__main__":
    main()
