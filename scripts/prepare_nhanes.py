"""Build the first analysis-ready NHANES table."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from body_soft_sensor.nhanes import (
    build_quality_report,
    prepare_all_cycles,
    write_quality_report,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "nhanes"
OUTPUT_PATH = ROOT / "data" / "processed" / "nhanes_2011_2018_body_composition.parquet"
QUALITY_PATH = ROOT / "artifacts" / "data_quality.json"
CYCLES = {
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}


def main() -> None:
    data = prepare_all_cycles(RAW_ROOT, CYCLES)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)
    write_quality_report(build_quality_report(data), QUALITY_PATH)
    sources = []
    for cycle, suffix in CYCLES.items():
        for component in ("DEMO", "BMX", "DXX"):
            source = RAW_ROOT / cycle / f"{component}_{suffix}.xpt"
            sources.append({"cycle": cycle, "file": source.name,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "size_bytes": source.stat().st_size})
    manifest = {"prepared_at": datetime.now(timezone.utc).isoformat(),
                "dataset": OUTPUT_PATH.name,
                "dataset_sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
                "sources": sources, "rows": len(data),
                "targets": {"body_fat_percent": "DXDTOPF (whole body, percent)",
                            "total_fat_mass_kg": "DXDTOFAT / 1000 (whole body, kg)",
                            "total_lean_mass_kg": "DXDTOLE / 1000 (whole body excluding BMC, kg)"}}
    (QUALITY_PATH.parent / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared {len(data):,} adult participant rows across {len(CYCLES)} cycles")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Quality report: {QUALITY_PATH}")


if __name__ == "__main__":
    main()
