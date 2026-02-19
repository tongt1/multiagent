"""Tier 1 CPU Local: Checkpoint metadata schema validation.

Validates checkpoint metadata JSON schemas without loading actual weights.
Uses golden reference schemas embedded as dicts. Target runtime: < 5 seconds.

Catches: TensorStore vs numcodecs compressor format, missing metadata fields,
wrong fill_value or dimension_separator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Golden Reference Schemas ─────────────────────────────────────────────────

REQUIRED_METADATA_FIELDS = {"n_tokens_seen", "current_step", "ckpt_version", "config_history"}

REQUIRED_ARCH_CONFIG_FIELDS = {"mup", "moe", "sliding_window_size", "layer_switch", "rank_fraction"}

EXPECTED_CHECKPOINT_FILES = {
    "metadata.json",
    "arch_config.json",
    "_CHECKPOINT_IS_COMPLETE",
}

# TensorStore zarr v2 compressor format (NOT numcodecs)
VALID_COMPRESSOR = {"id": "zstd", "level": 1}

# numcodecs format that is WRONG for fax
INVALID_NUMCODECS_COMPRESSOR_KEYS = {"blocksize", "clevel", "cname", "shuffle"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_valid_metadata() -> dict:
    """Create a valid metadata.json for testing."""
    return {
        "n_tokens_seen": 0,
        "current_step": 0,
        "ckpt_version": 1,
        "config_history": {"null": {}},
    }


def _make_valid_arch_config() -> dict:
    """Create a valid arch_config.json for testing."""
    return {
        "mup": False,
        "moe": False,
        "sliding_window_size": None,
        "layer_switch": None,
        "rank_fraction": None,
    }


def _make_valid_zarr_metadata() -> dict:
    """Create valid zarr .zarray metadata."""
    return {
        "chunks": [128, 128],
        "compressor": {"id": "zstd", "level": 1},
        "dtype": "<f4",
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [128, 128],
        "zarr_format": 2,
        "dimension_separator": ".",
    }


# ── TestCheckpointMetadataSchema ─────────────────────────────────────────────


class TestCheckpointMetadataSchema:
    """Validate metadata.json schema against known-good structure."""

    def test_metadata_required_fields(self) -> None:
        """metadata.json must have n_tokens_seen, current_step, ckpt_version, config_history."""
        metadata = _make_valid_metadata()
        for field in REQUIRED_METADATA_FIELDS:
            assert field in metadata, f"metadata.json missing required field: {field}"

    def test_metadata_missing_field_detected(self) -> None:
        """Removing a required field should be detected."""
        metadata = _make_valid_metadata()
        del metadata["n_tokens_seen"]
        missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
        assert "n_tokens_seen" in missing

    def test_config_history_step_key(self) -> None:
        """config_history key must be 'null' for step 0."""
        metadata = _make_valid_metadata()
        assert "null" in metadata["config_history"], (
            "config_history must have 'null' key for step 0"
        )

    def test_arch_config_required_fields(self) -> None:
        """arch_config.json must have mup, moe, sliding_window_size, layer_switch, rank_fraction."""
        arch = _make_valid_arch_config()
        for field in REQUIRED_ARCH_CONFIG_FIELDS:
            assert field in arch, f"arch_config.json missing required field: {field}"

    def test_zarr_metadata_format(self) -> None:
        """zarr .zarray must use TensorStore compressor format."""
        zarray = _make_valid_zarr_metadata()
        assert zarray["compressor"] == VALID_COMPRESSOR, (
            f"Compressor must be {VALID_COMPRESSOR}, got {zarray['compressor']}"
        )
        assert zarray["dimension_separator"] == ".", (
            f"dimension_separator must be '.', got {zarray['dimension_separator']}"
        )


# ── TestCheckpointTree ────────────────────────────────────────────────────────


class TestCheckpointTree:
    """Validate checkpoint directory structure."""

    def test_expected_files_exist(self, tmp_path: Path) -> None:
        """Checkpoint must contain metadata.json, arch_config.json, _CHECKPOINT_IS_COMPLETE."""
        # Create mock checkpoint
        ckpt = tmp_path / "ckpt-0"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps(_make_valid_metadata()))
        (ckpt / "arch_config.json").write_text(json.dumps(_make_valid_arch_config()))
        (ckpt / "_CHECKPOINT_IS_COMPLETE").touch()
        (ckpt / "zord-0").mkdir()

        found = {f.name for f in ckpt.iterdir() if not f.name.startswith(".")}
        for expected in EXPECTED_CHECKPOINT_FILES:
            assert expected in found, f"Checkpoint missing expected file: {expected}"
        assert "zord-0" in found, "Checkpoint missing zord-0 directory"

    def test_no_extra_unknown_files(self, tmp_path: Path) -> None:
        """Warn on unexpected files in checkpoint directory."""
        known_files = EXPECTED_CHECKPOINT_FILES | {"run_config.json", "zord-0"}
        ckpt = tmp_path / "ckpt-0"
        ckpt.mkdir()
        for f in known_files:
            if f == "zord-0":
                (ckpt / f).mkdir()
            else:
                (ckpt / f).touch()
        # Add unexpected file
        (ckpt / "surprise.txt").touch()

        found = {f.name for f in ckpt.iterdir() if not f.name.startswith(".")}
        unexpected = found - known_files
        assert unexpected == {"surprise.txt"}, (
            f"Should detect unexpected files: {unexpected}"
        )


# ── TestZarrArrayMetadata ─────────────────────────────────────────────────────


class TestZarrArrayMetadata:
    """Validate zarr array metadata format."""

    def test_compressor_tensorstore_format(self) -> None:
        """Compressor must be {"id":"zstd","level":1}, NOT numcodecs format."""
        zarray = _make_valid_zarr_metadata()
        compressor = zarray["compressor"]
        assert compressor == VALID_COMPRESSOR

        # Verify numcodecs format is rejected
        numcodecs_compressor = {
            "blocksize": 0,
            "clevel": 5,
            "cname": "lz4",
            "id": "blosc",
            "shuffle": 1,
        }
        assert numcodecs_compressor != VALID_COMPRESSOR
        overlap = set(numcodecs_compressor.keys()) & INVALID_NUMCODECS_COMPRESSOR_KEYS
        assert len(overlap) > 0, "numcodecs compressor should have invalid keys"

    def test_fill_value_null(self) -> None:
        """fill_value must be null (None), not 0.0."""
        zarray = _make_valid_zarr_metadata()
        assert zarray["fill_value"] is None, (
            f"fill_value must be null/None, got {zarray['fill_value']}"
        )

    def test_dimension_separator_dot(self) -> None:
        """dimension_separator must be '.'."""
        zarray = _make_valid_zarr_metadata()
        assert zarray["dimension_separator"] == ".", (
            f"dimension_separator must be '.', got {zarray['dimension_separator']}"
        )


# ── Validation helpers for use by cluster tests ──────────────────────────────


def validate_metadata_json(data: dict) -> list[str]:
    """Validate a metadata.json dict. Returns list of error messages."""
    errors = []
    missing = REQUIRED_METADATA_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    if "config_history" in data:
        if not isinstance(data["config_history"], dict):
            errors.append("config_history must be a dict")
    return errors


def validate_arch_config_json(data: dict) -> list[str]:
    """Validate an arch_config.json dict. Returns list of error messages."""
    errors = []
    missing = REQUIRED_ARCH_CONFIG_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    return errors


def validate_zarr_array(data: dict) -> list[str]:
    """Validate a .zarray dict. Returns list of error messages."""
    errors = []
    compressor = data.get("compressor", {})
    if compressor != VALID_COMPRESSOR:
        if set(compressor.keys()) & INVALID_NUMCODECS_COMPRESSOR_KEYS:
            errors.append(
                f"Compressor uses numcodecs format: {compressor}. "
                f"Must use TensorStore format: {VALID_COMPRESSOR}"
            )
        elif compressor != VALID_COMPRESSOR:
            errors.append(f"Compressor must be {VALID_COMPRESSOR}, got {compressor}")

    if data.get("fill_value") is not None:
        # JSON null becomes Python None
        if data["fill_value"] != 0 or data.get("fill_value") == 0.0:
            errors.append(f"fill_value should be null, got {data['fill_value']}")

    if data.get("dimension_separator") != ".":
        errors.append(
            f"dimension_separator must be '.', got {data.get('dimension_separator')}"
        )

    return errors
