"""Tier 1 CPU Local: .run config file parsing and validation.

Parses .run files (GIN-style key=value format) and validates fields.
Catches tokenizer/data path issues. Target runtime: < 5 seconds.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from configs.model_profiles import LLAMA_1B_INSTRUCT, SMOLLM_135M

# ── Paths ────────────────────────────────────────────────────────────────────

_RUN_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "run_configs"

SMOLLM_RUN = _RUN_CONFIGS_DIR / "smollm_135m_rloo_math.run"
LLAMA_RUN = _RUN_CONFIGS_DIR / "llama_1b_rloo_math.run"


# ── Simple .run file parser ──────────────────────────────────────────────────


def parse_run_file(path: Path) -> dict[str, str]:
    """Parse a .run file into a flat dict of key=value pairs.

    Handles:
    - Simple key = value lines
    - Multi-line values with parentheses (e.g., tokenizer_path = (\n"..."\n))
    - Dict/list literals spanning multiple lines
    - Comments starting with #
    """
    content = path.read_text()
    result: dict[str, str] = {}

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            i += 1
            continue

        # Check for key = value pattern
        match = re.match(r"^([\w.]+)\s*=\s*(.*)", line)
        if not match:
            i += 1
            continue

        key = match.group(1)
        value = match.group(2).strip()

        # Handle multi-line values (parenthesized or dict/list)
        if value and (value.startswith("(") or value.startswith("{") or value.startswith("[")):
            open_char = value[0]
            close_char = {"(": ")", "{": "}", "[": "]"}[open_char]
            depth = value.count(open_char) - value.count(close_char)
            while depth > 0 and i + 1 < len(lines):
                i += 1
                next_line = lines[i].strip()
                value += " " + next_line
                depth += next_line.count(open_char) - next_line.count(close_char)

        result[key] = value
        i += 1

    return result


def get_field(config: dict[str, str], key: str) -> str | None:
    """Get a field value, stripping quotes."""
    val = config.get(key)
    if val is None:
        return None
    # Strip surrounding quotes
    val = val.strip()
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    elif val.startswith("'") and val.endswith("'"):
        val = val[1:-1]
    return val


# ── TestRunConfigFields ───────────────────────────────────────────────────────


class TestRunConfigFields:
    """Validate individual .run file fields."""

    def test_smollm_run_config_parses(self) -> None:
        """smollm_135m_rloo_math.run parses without error."""
        assert SMOLLM_RUN.exists(), f"Missing: {SMOLLM_RUN}"
        config = parse_run_file(SMOLLM_RUN)
        assert len(config) > 0, "Empty config after parsing"

    def test_llama_run_config_parses(self) -> None:
        """llama_1b_rloo_math.run parses without error."""
        assert LLAMA_RUN.exists(), f"Missing: {LLAMA_RUN}"
        config = parse_run_file(LLAMA_RUN)
        assert len(config) > 0, "Empty config after parsing"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_tokenizer_path_present(self, run_file: Path) -> None:
        """tokenizer_path is set."""
        config = parse_run_file(run_file)
        # tokenizer_path may use multi-line parens
        raw = run_file.read_text()
        assert "tokenizer_path" in raw, f"{run_file.name}: missing tokenizer_path"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_tokenizer_path_has_eot_or_eos(self, run_file: Path) -> None:
        """Tokenizer path must contain 'eot' or 'eos' for TokenizerInfo heuristic."""
        raw = run_file.read_text()
        # Extract tokenizer_path value from raw text
        match = re.search(r'tokenizer_path\s*=\s*(?:\(\s*)?["\']([^"\']+)["\']', raw)
        assert match, f"{run_file.name}: could not extract tokenizer_path"
        tok_path = match.group(1)
        assert "eot" in tok_path or "eos" in tok_path, (
            f"{run_file.name}: tokenizer_path must contain 'eot' or 'eos' "
            f"for TokenizerInfo heuristic. Got: {tok_path}"
        )

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_data_dir_dict_has_gcs_paths(self, run_file: Path) -> None:
        """All data paths must start with gs://."""
        raw = run_file.read_text()
        # Find all GCS paths in data_dir_dict
        data_matches = re.findall(r'"(gs://[^"]+)"', raw)
        assert len(data_matches) > 0, f"{run_file.name}: no GCS paths found in data_dir_dict"
        for path in data_matches:
            assert path.startswith("gs://"), f"Data path must be GCS: {path}"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_wandb_project_set(self, run_file: Path) -> None:
        """wandb.project_name is set."""
        config = parse_run_file(run_file)
        assert "wandb.project_name" in config, f"{run_file.name}: missing wandb.project_name"
        val = get_field(config, "wandb.project_name")
        assert val, f"{run_file.name}: wandb.project_name is empty"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_max_sequence_length_set(self, run_file: Path) -> None:
        """max_sequence_length > 0."""
        config = parse_run_file(run_file)
        assert "max_sequence_length" in config, f"{run_file.name}: missing max_sequence_length"
        val = int(config["max_sequence_length"])
        assert val > 0, f"{run_file.name}: max_sequence_length must be > 0, got {val}"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_reward_model_set(self, run_file: Path) -> None:
        """reward_model.name is set."""
        config = parse_run_file(run_file)
        assert "reward_model.name" in config, f"{run_file.name}: missing reward_model.name"
        val = get_field(config, "reward_model.name")
        assert val, f"{run_file.name}: reward_model.name is empty"

    @pytest.mark.parametrize("run_file", [SMOLLM_RUN, LLAMA_RUN], ids=["smollm", "llama"])
    def test_seed_set(self, run_file: Path) -> None:
        """seed is set."""
        config = parse_run_file(run_file)
        assert "seed" in config, f"{run_file.name}: missing seed"
        val = int(config["seed"])
        assert val >= 0, f"{run_file.name}: seed must be >= 0, got {val}"


# ── TestRunConfigConsistency ──────────────────────────────────────────────────


class TestRunConfigConsistency:
    """Cross-validate .run files against model profiles."""

    def test_smollm_seq_len_matches_profile(self) -> None:
        """SmolLM .run max_sequence_length == model profile (2048)."""
        config = parse_run_file(SMOLLM_RUN)
        val = int(config["max_sequence_length"])
        assert val == SMOLLM_135M.max_sequence_length, (
            f"SmolLM .run has max_sequence_length={val}, "
            f"but model profile says {SMOLLM_135M.max_sequence_length}"
        )

    def test_llama_seq_len_matches_profile(self) -> None:
        """Llama .run max_sequence_length == model profile (4096)."""
        config = parse_run_file(LLAMA_RUN)
        val = int(config["max_sequence_length"])
        assert val == LLAMA_1B_INSTRUCT.max_sequence_length, (
            f"Llama .run has max_sequence_length={val}, "
            f"but model profile says {LLAMA_1B_INSTRUCT.max_sequence_length}"
        )
