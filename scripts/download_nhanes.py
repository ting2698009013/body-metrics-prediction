"""Download the public NHANES files used by this project.

Existing files are never overwritten. The URLs are the official CDC public-use
SAS transport files.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

CYCLES = {
    "2011-2012": ("2011", "G"),
    "2013-2014": ("2013", "H"),
    "2015-2016": ("2015", "I"),
    "2017-2018": ("2017", "J"),
}


def validate_xpt(path: Path) -> None:
    with path.open("rb") as source:
        if not source.read(80).startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"):
            raise ValueError(f"Not a SAS XPORT file (possibly an HTML error page): {path}")


def record_source(url: str, output_path: Path, downloaded: bool) -> None:
    validate_xpt(output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sidecar = output_path.with_suffix(".source.json")
    if sidecar.is_file():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous.get("sha256") != digest or previous.get("source_url") != url:
            raise ValueError(f"Source differs from the recorded provenance; inspect before reuse: {output_path}")
    metadata = {"source_url": url, "file": output_path.name,
                "sha256": digest,
                "size_bytes": output_path.stat().st_size,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "downloaded_this_run": downloaded}
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def download(url: str, output_path: Path) -> None:
    if output_path.is_file():
        record_source(url, output_path, downloaded=False)
        print(f"Skip existing: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Download: {url}")
    with urlopen(url, timeout=120) as response, temporary_path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    validate_xpt(temporary_path)
    temporary_path.replace(output_path)
    record_source(url, output_path, downloaded=True)
    print(f"Saved: {output_path}")


def main() -> None:
    for cycle, (url_year, suffix) in CYCLES.items():
        cycle_dir = ROOT / "data" / "raw" / "nhanes" / cycle
        for component in ("DEMO", "BMX", "DXX"):
            file_name = f"{component}_{suffix}.xpt"
            url = f"{BASE_URL}/{url_year}/DataFiles/{file_name}"
            download(url, cycle_dir / file_name)


if __name__ == "__main__":
    main()

