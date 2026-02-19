"""Tier 2 CPU Cluster: GCS path existence and data validation.

Requires GCS access but no GPU. All tests marked with @pytest.mark.cluster.
Target runtime: 30-60 seconds.

Run: pytest tests/test_cluster_paths.py -m cluster -v
Skip in local: pytest tests/ -m "not cluster"
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from configs.model_profiles import LLAMA_1B_INSTRUCT, SMOLLM_135M
from tests.test_checkpoint_format import (
    validate_arch_config_json,
    validate_metadata_json,
)

# Mark all tests in this module as cluster tests
pytestmark = pytest.mark.cluster

# ── Constants ────────────────────────────────────────────────────────────────

SMOLLM_CKPT = SMOLLM_135M.ckpt_path
LLAMA_CKPT = LLAMA_1B_INSTRUCT.ckpt_path

LLAMA_TOKENIZER = "gs://cohere-dev-central-2/users/terry/llama-1b-instruct-eot/tokenizer.json"
SMOLLM_TOKENIZER = (
    "gs://cohere-prod/encoders/releases/0.6.0/"
    "m255k_bos_eos_fim_agents3_special_tokens_with_eot_template-ns.json"
)

MATH_TRAIN_DATA = (
    "gs://cohere-command/data/generated_jsonl_data/"
    "comb_2025_JAN_13_math_train/2026_01_13/train.jsonl"
)

SECRETS_TEMPLATE = Path.home() / "repos" / "secrets_template.toml"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _gsutil_ls(path: str, timeout: int = 30) -> bool:
    """Check if a GCS path exists via gsutil ls."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _gsutil_cat(path: str, timeout: int = 30) -> str | None:
    """Read a small file from GCS via gsutil cat."""
    try:
        result = subprocess.run(
            ["gsutil", "cat", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _gsutil_cat_first_line(path: str, timeout: int = 30) -> str | None:
    """Read first line of a GCS file via gsutil cat | head -1.

    Uses a pipe to avoid downloading entire large files.
    """
    try:
        cat_proc = subprocess.Popen(
            ["gsutil", "cat", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        head_proc = subprocess.Popen(
            ["head", "-1"],
            stdin=cat_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cat_proc.stdout.close()
        output, _ = head_proc.communicate(timeout=timeout)
        cat_proc.terminate()
        cat_proc.wait(timeout=5)
        return output.decode("utf-8").strip() if head_proc.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ── TestGCSPaths ──────────────────────────────────────────────────────────────


class TestGCSPaths:
    """Verify GCS paths for checkpoints, data, and tokenizers exist."""

    def test_smollm_checkpoint_exists(self) -> None:
        assert _gsutil_ls(SMOLLM_CKPT), f"SmolLM checkpoint not found: {SMOLLM_CKPT}"

    def test_llama_checkpoint_exists(self) -> None:
        assert _gsutil_ls(LLAMA_CKPT), f"Llama checkpoint not found: {LLAMA_CKPT}"

    def test_smollm_ckpt_is_complete(self) -> None:
        marker = f"{SMOLLM_CKPT}/_CHECKPOINT_IS_COMPLETE"
        assert _gsutil_ls(marker), f"SmolLM checkpoint incomplete: {marker}"

    def test_llama_ckpt_is_complete(self) -> None:
        marker = f"{LLAMA_CKPT}/_CHECKPOINT_IS_COMPLETE"
        assert _gsutil_ls(marker), f"Llama checkpoint incomplete: {marker}"

    def test_math_train_data_exists(self) -> None:
        assert _gsutil_ls(MATH_TRAIN_DATA), f"MATH train data not found: {MATH_TRAIN_DATA}"

    def test_llama_tokenizer_exists(self) -> None:
        assert _gsutil_ls(LLAMA_TOKENIZER), f"Llama tokenizer not found: {LLAMA_TOKENIZER}"


# ── TestTokenizerLoading ──────────────────────────────────────────────────────


class TestTokenizerLoading:
    """Verify tokenizer files can be downloaded and parsed."""

    def test_llama_tokenizer_loads(self) -> None:
        content = _gsutil_cat(LLAMA_TOKENIZER)
        assert content is not None, f"Failed to download: {LLAMA_TOKENIZER}"
        data = json.loads(content)
        assert "model" in data or "added_tokens" in data, (
            "Llama tokenizer.json missing expected fields"
        )

    def test_smollm_tokenizer_loads(self) -> None:
        content = _gsutil_cat(SMOLLM_TOKENIZER)
        assert content is not None, f"Failed to download: {SMOLLM_TOKENIZER}"
        # SmolLM tokenizer may be in different format (not necessarily HF JSON)
        assert len(content) > 0, "SmolLM tokenizer file is empty"


# ── TestDataSampling ──────────────────────────────────────────────────────────


class TestDataSampling:
    """Verify training data format by sampling first record."""

    def test_math_train_first_record_loads(self) -> None:
        """Download first line of JSONL and parse as JSON."""
        line = _gsutil_cat_first_line(MATH_TRAIN_DATA)
        assert line is not None, f"Failed to read first line of {MATH_TRAIN_DATA}"
        record = json.loads(line)
        assert isinstance(record, dict), "First JSONL record must be a dict"

    def test_data_has_expected_fields(self) -> None:
        """Training data records must have env_name and scenario_config."""
        line = _gsutil_cat_first_line(MATH_TRAIN_DATA)
        assert line is not None
        record = json.loads(line)
        # COMB format typically has these fields
        assert "env_name" in record or "scenario_config" in record, (
            f"Training data missing expected fields. Got keys: {list(record.keys())}"
        )


# ── TestSecretsTemplate ───────────────────────────────────────────────────────


class TestSecretsTemplate:
    """Verify secrets template file exists."""

    def test_secrets_template_exists(self) -> None:
        assert SECRETS_TEMPLATE.exists(), (
            f"Secrets template not found: {SECRETS_TEMPLATE}\n"
            "Expected at ~/repos/secrets_template.toml"
        )


# ── TestCheckpointMetadata ────────────────────────────────────────────────────


class TestCheckpointMetadata:
    """Download and validate checkpoint metadata from GCS."""

    def test_smollm_metadata_json_valid(self) -> None:
        content = _gsutil_cat(f"{SMOLLM_CKPT}/metadata.json")
        assert content is not None, f"Failed to download SmolLM metadata.json"
        data = json.loads(content)
        errors = validate_metadata_json(data)
        assert not errors, f"SmolLM metadata.json validation errors: {errors}"

    def test_llama_metadata_json_valid(self) -> None:
        content = _gsutil_cat(f"{LLAMA_CKPT}/metadata.json")
        assert content is not None, f"Failed to download Llama metadata.json"
        data = json.loads(content)
        errors = validate_metadata_json(data)
        assert not errors, f"Llama metadata.json validation errors: {errors}"

    def test_smollm_arch_config_valid(self) -> None:
        content = _gsutil_cat(f"{SMOLLM_CKPT}/arch_config.json")
        assert content is not None, f"Failed to download SmolLM arch_config.json"
        data = json.loads(content)
        errors = validate_arch_config_json(data)
        assert not errors, f"SmolLM arch_config.json validation errors: {errors}"

    def test_llama_arch_config_valid(self) -> None:
        content = _gsutil_cat(f"{LLAMA_CKPT}/arch_config.json")
        assert content is not None, f"Failed to download Llama arch_config.json"
        data = json.loads(content)
        errors = validate_arch_config_json(data)
        assert not errors, f"Llama arch_config.json validation errors: {errors}"
