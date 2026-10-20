"""Reproducibility checks for the checksum-based public-input fetcher."""
import hashlib

import pytest

from paper.experiments import fetch_external_inputs as inputs


def test_fetch_verifies_download_and_rejects_changed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "CACHE", tmp_path / "cache")
    source = tmp_path / "source"
    source.mkdir()
    original = b"faculty input\n"
    (source / "sample.bin").write_bytes(original)
    entry = {"group": "sample", "cache_path": "sample/sample.bin",
             "source_path": "sample.bin", "bytes": len(original),
             "sha256": hashlib.sha256(original).hexdigest()}
    bases = {"sample": source.as_uri() + "/"}

    assert inputs.fetch(entry, bases, verify_only=False)[0] == "downloaded"
    assert inputs.fetch(entry, bases, verify_only=True)[0] == "verified"
    (inputs.CACHE / entry["cache_path"]).write_bytes(b"tampered\n")
    assert inputs.fetch(entry, bases, verify_only=True)[0] == "missing or mismatched"
    with pytest.raises(ValueError, match="different checksum"):
        inputs.fetch(entry, bases, verify_only=False)


def test_fetcher_rejects_path_outside_cache():
    with pytest.raises(ValueError, match="Unsafe cache path"):
        inputs.target_for({"cache_path": "../elsewhere"})
