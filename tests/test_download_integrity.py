import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/download_nhanes.py"
spec = importlib.util.spec_from_file_location("download_nhanes", path)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_html_error_page_is_never_accepted_as_xpt(tmp_path):
    path = tmp_path / "DXX_J.xpt"
    path.write_bytes(b"<html>File not found</html>")
    with pytest.raises(ValueError, match="Not a SAS XPORT"):
        download.record_source("https://example.test/file", path, False)
    assert not path.with_suffix(".source.json").exists()


def test_cached_source_changes_are_detected(tmp_path):
    path = tmp_path / "DXX_J.xpt"
    path.write_bytes(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + b" " * 40)
    download.record_source("https://example.test/file", path, False)
    original = json.loads(path.with_suffix(".source.json").read_text())
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        download.record_source("https://example.test/file", path, False)
    assert json.loads(path.with_suffix(".source.json").read_text()) == original
