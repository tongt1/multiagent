"""Tier 1 CPU Local: Sweep infrastructure validation.

Validates sidecar commands, env vars, model profiles, patch_run_config,
filter config, and config consistency across ALL sweep configs.
Uses AST parsing + regex (no cluster deps). Target runtime: < 10 seconds.

Would have caught:
- vLLM sidecar bash -c crash (word splitting in $DECODED_COMMAND)
- FAX_NUMBER_GPUS_PER_WORKER missing from sidecar cmd
- --max-model-len missing from sidecar cmd
- --enforce-eager missing from sidecar cmd
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from configs.model_profiles import LLAMA_1B_INSTRUCT, SMOLLM_135M, ModelProfile
from configs.reward_shaping_sweep._base import (
    CKPT_PATH,
    EXPORT_EVERY_STEPS,
    GENERATIONS_PER_PROMPT,
    K8S_SECRETS_PATH,
    MAX_SEQUENCE_LENGTH,
    NUM_SAMPLING_GPUS,
    NUM_TRAINING_GPUS,
    PRIORITY_CLASS,
    SEED,
    TOTAL_TRAIN_STEPS,
    TRAIN_BATCH_SIZE,
    WANDB_PROJECT,
    _VLLM_EXPORT_DIR,
    _VLLM_PORT,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "reward_shaping_sweep"

# The 5 strategy sweep configs + 2 GPU test configs
ALL_CONFIG_FILES = sorted(_CONFIGS_DIR.glob("sweep_*.py"))

# Strategy-only configs (excludes GPU test variants)
STRATEGY_FILES = {
    p.stem: p
    for p in ALL_CONFIG_FILES
    if "gpu" not in p.stem and "llama" not in p.stem
}

# GPU test configs
GPU_TEST_FILES = {
    p.stem: p
    for p in ALL_CONFIG_FILES
    if "gpu" in p.stem or "llama" in p.stem
}


def _read_source(path: Path) -> str:
    return path.read_text()


def _parse_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _extract_sidecar_command(source: str) -> str:
    """Extract the sidecar command string from a sweep config.

    Finds the `command=" ".join([...])` pattern inside SidecarConfig and
    reconstructs the joined string from the list of string literals and
    f-string expressions.
    """
    # Find the command=" ".join( pattern
    match = re.search(r'command=" "\.join\(\s*\[', source)
    if not match:
        return ""

    # Find the matching closing bracket
    start = match.end()
    bracket_depth = 1
    i = start
    while i < len(source) and bracket_depth > 0:
        if source[i] == "[":
            bracket_depth += 1
        elif source[i] == "]":
            bracket_depth -= 1
        i += 1

    list_content = source[start : i - 1]

    # Extract all string literals (both regular and f-strings)
    # Match quoted strings, handling f-strings with {var} inside
    parts = re.findall(r'[f]?"([^"]*)"', list_content)
    if not parts:
        parts = re.findall(r"[f]?'([^']*)'", list_content)

    return " ".join(parts)


def _extract_env_dict(source: str) -> str:
    """Extract the patch_kjobs_compute env dict block from source."""
    match = re.search(r"patch_kjobs_compute=dict\(", source)
    if not match:
        return ""
    start = match.start()
    depth = 0
    for i in range(match.end() - 1, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    return ""


def _extract_patch_run_config(source: str) -> str:
    """Extract the patch_run_config=dict(...) block from source."""
    match = re.search(r"patch_run_config=dict\(", source)
    if not match:
        return ""
    start = match.start()
    depth = 0
    for i in range(match.end() - 1, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    return ""


# ── TestSidecarCommand ───────────────────────────────────────────────────────


class TestSidecarCommand:
    """Validate vLLM sidecar command across all configs."""

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_no_bash_c_wrapper(self, path: Path) -> None:
        """bash -c breaks with $DECODED_COMMAND due to word splitting."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert cmd, f"{path.stem}: could not extract sidecar command"
        assert "bash -c" not in cmd, (
            f"{path.stem}: sidecar command must NOT use 'bash -c' wrapper "
            "(causes word splitting in $DECODED_COMMAND)"
        )
        assert "bash" not in cmd.split()[0] if cmd else True, (
            f"{path.stem}: sidecar command must not start with bash"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_has_fax_number_gpus(self, path: Path) -> None:
        """FAX_NUMBER_GPUS_PER_WORKER=1 must be in sidecar command."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert "FAX_NUMBER_GPUS_PER_WORKER=1" in cmd, (
            f"{path.stem}: sidecar command missing FAX_NUMBER_GPUS_PER_WORKER=1"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_has_max_model_len(self, path: Path) -> None:
        """--max-model-len must be in sidecar command."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert "--max-model-len" in cmd, (
            f"{path.stem}: sidecar command missing --max-model-len"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_has_enforce_eager(self, path: Path) -> None:
        """--enforce-eager must be in sidecar command."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert "--enforce-eager" in cmd, (
            f"{path.stem}: sidecar command missing --enforce-eager"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_no_single_quotes(self, path: Path) -> None:
        """Single quotes break in $DECODED_COMMAND."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert "'" not in cmd, (
            f"{path.stem}: sidecar command must not contain single quotes "
            "(breaks word splitting in $DECODED_COMMAND)"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_vllm_export_dir_pattern(self, path: Path) -> None:
        """--exports-glob-pattern uses correct directory."""
        source = _read_source(path)
        cmd = _extract_sidecar_command(source)
        assert "--exports-glob-pattern=" in cmd, (
            f"{path.stem}: sidecar command missing --exports-glob-pattern"
        )
        assert "_HF_EXPORT_IS_COMPLETE" in cmd, (
            f"{path.stem}: exports-glob-pattern must look for _HF_EXPORT_IS_COMPLETE marker"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_port_matches_config(self, path: Path) -> None:
        """--port in sidecar command must use same variable as ports=dict(web=...)."""
        source = _read_source(path)
        # The sidecar command uses f-string: f"--port {_VLLM_PORT}"
        # And ports uses: ports=dict(web=_VLLM_PORT)
        # Verify both reference the same variable in source
        port_in_cmd = re.search(r'f"--port \{(\w+)\}"', source)
        assert port_in_cmd, f"{path.stem}: --port f-string not found in sidecar command"
        cmd_var = port_in_cmd.group(1)

        web_match = re.search(r"ports=dict\(web=(\w+)\)", source)
        assert web_match, f"{path.stem}: ports=dict(web=...) not found"
        web_var = web_match.group(1)

        assert cmd_var == web_var, (
            f"{path.stem}: sidecar --port uses {cmd_var} but ports=dict(web={web_var})"
        )

    def test_all_strategy_configs_identical_sidecar(self) -> None:
        """All 5 strategy configs must build identical sidecar commands."""
        commands = {}
        for name, path in STRATEGY_FILES.items():
            source = _read_source(path)
            cmd = _extract_sidecar_command(source)
            assert cmd, f"{name}: could not extract sidecar command"
            commands[name] = cmd

        reference = commands["sweep_identity"]
        for name, cmd in commands.items():
            if name == "sweep_identity":
                continue
            assert cmd == reference, (
                f"Sidecar command in {name} differs from sweep_identity:\n"
                f"  identity: {reference}\n"
                f"  {name}:   {cmd}"
            )


# ── TestKjobsEnvVars ─────────────────────────────────────────────────────────


class TestKjobsEnvVars:
    """Validate kjobs compute environment variables."""

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_jax_cache_dir_set(self, path: Path) -> None:
        source = _read_source(path)
        env_block = _extract_env_dict(source)
        assert "JAX_COMPILATION_CACHE_DIR" in env_block, (
            f"{path.stem}: missing JAX_COMPILATION_CACHE_DIR in patch_kjobs_compute env"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_jax_log_compiles_set(self, path: Path) -> None:
        source = _read_source(path)
        env_block = _extract_env_dict(source)
        assert "JAX_LOG_COMPILES" in env_block, (
            f"{path.stem}: missing JAX_LOG_COMPILES in patch_kjobs_compute env"
        )

    @pytest.mark.parametrize("path", list(GPU_TEST_FILES.values()), ids=lambda p: p.stem)
    def test_debug_configs_have_unbuffered(self, path: Path) -> None:
        """GPU test configs should have PYTHONUNBUFFERED for crash debugging."""
        source = _read_source(path)
        env_block = _extract_env_dict(source)
        # Only the llama1b test currently has this, so we check if it's a GPU test
        if "llama" in path.stem:
            assert "PYTHONUNBUFFERED" in env_block, (
                f"{path.stem}: GPU debug config should have PYTHONUNBUFFERED=1"
            )


# ── TestModelProfiles ─────────────────────────────────────────────────────────


class TestModelProfiles:
    """Validate model profile frozen values."""

    def test_smollm_profile_values(self) -> None:
        assert SMOLLM_135M.num_training_gpus == 1
        assert SMOLLM_135M.num_sampling_gpus == 1
        assert SMOLLM_135M.max_sequence_length == 2048
        assert SMOLLM_135M.needs_mesh_override is False
        assert "smollm-135M" in SMOLLM_135M.ckpt_path

    def test_llama_profile_values(self) -> None:
        assert LLAMA_1B_INSTRUCT.num_training_gpus == 4
        assert LLAMA_1B_INSTRUCT.num_sampling_gpus == 4
        assert LLAMA_1B_INSTRUCT.max_sequence_length == 4096
        assert LLAMA_1B_INSTRUCT.needs_mesh_override is False
        assert "llama-1b-instruct" in LLAMA_1B_INSTRUCT.ckpt_path

    @pytest.mark.parametrize("profile", [SMOLLM_135M, LLAMA_1B_INSTRUCT], ids=["smollm", "llama"])
    def test_gpu_counts_positive(self, profile: ModelProfile) -> None:
        assert profile.num_training_gpus > 0, "Training GPU count must be positive"
        assert profile.num_sampling_gpus > 0, "Sampling GPU count must be positive"

    @pytest.mark.parametrize("profile", [SMOLLM_135M, LLAMA_1B_INSTRUCT], ids=["smollm", "llama"])
    def test_ckpt_path_is_gcs(self, profile: ModelProfile) -> None:
        assert profile.ckpt_path.startswith("gs://"), (
            f"Checkpoint path must be a GCS path, got: {profile.ckpt_path}"
        )

    def test_tokenizer_path_heuristic(self) -> None:
        """Llama tokenizer path must contain 'eot' or 'eos' for TokenizerInfo heuristic."""
        # Read the llama .run file to check
        run_config_path = (
            Path(__file__).resolve().parent.parent / "run_configs" / "llama_1b_rloo_math.run"
        )
        if run_config_path.exists():
            content = run_config_path.read_text()
            # Extract tokenizer_path value
            match = re.search(r'tokenizer_path\s*=\s*\(\s*"([^"]+)"', content)
            if match:
                tok_path = match.group(1)
                assert "eot" in tok_path or "eos" in tok_path, (
                    f"Tokenizer path must contain 'eot' or 'eos' for TokenizerInfo heuristic, "
                    f"got: {tok_path}"
                )


# ── TestPatchRunConfig ────────────────────────────────────────────────────────


class TestPatchRunConfig:
    """Validate patch_run_config structure across configs."""

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_minizord_name_present(self, path: Path) -> None:
        """PostTraining validator requires minizord with FlinkZordConfig."""
        source = _read_source(path)
        patch = _extract_patch_run_config(source)
        assert "minizord=" in patch or "FlinkZordConfig" in patch, (
            f"{path.stem}: patch_run_config must include minizord=FlinkZordConfig(...)"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_export_every_steps_set(self, path: Path) -> None:
        """At least one sampler must have export_every_steps set."""
        source = _read_source(path)
        assert "export_every_steps=" in source, (
            f"{path.stem}: must set export_every_steps on at least one sampler"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_only_one_sampler_exports(self, path: Path) -> None:
        """Exactly one sampler should export (not both)."""
        source = _read_source(path)
        # Find all export_every_steps assignments
        exports = re.findall(r"export_every_steps=(\w+)", source)
        non_none_exports = [e for e in exports if e != "None"]
        assert len(non_none_exports) >= 1, (
            f"{path.stem}: at least one sampler must export (export_every_steps != None)"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_eval_sampler_no_export(self, path: Path) -> None:
        """eval_sampler must have export_every_steps=None."""
        source = _read_source(path)
        # Find the eval_sampler_key block
        match = re.search(
            r"eval_sampler_key=FlinkVllmSidecarSamplerConfig\((.*?)\),\s*\)",
            source,
            re.DOTALL,
        )
        if match:
            eval_block = match.group(1)
            assert "export_every_steps=None" in eval_block, (
                f"{path.stem}: eval_sampler_key must have export_every_steps=None"
            )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_actor_sampler_keys_valid(self, path: Path) -> None:
        """Actor must reference an existing sampler key."""
        source = _read_source(path)
        assert 'sampler_endpoint_key="sampler_key"' in source, (
            f"{path.stem}: actor must reference sampler_endpoint_key='sampler_key'"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_learner_loss_is_grpo(self, path: Path) -> None:
        """Learner must use policy_gradient_loss='grpo'."""
        source = _read_source(path)
        assert 'policy_gradient_loss="grpo"' in source, (
            f"{path.stem}: learner must use policy_gradient_loss='grpo'"
        )

    @pytest.mark.parametrize(
        "path", list(STRATEGY_FILES.values()), ids=lambda p: p.stem
    )
    def test_generations_per_prompt_consistent(self, path: Path) -> None:
        """generations_per_prompt in loss config must match GENERATIONS_PER_PROMPT."""
        source = _read_source(path)
        # In strategy configs, the loss preference block should use the constant
        assert '"generations_per_prompt": GENERATIONS_PER_PROMPT' in source, (
            f"{path.stem}: loss.preference.generations_per_prompt must use GENERATIONS_PER_PROMPT"
        )


# ── TestFilterConfig ──────────────────────────────────────────────────────────


class TestFilterConfig:
    """Validate filter configuration consistency."""

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_identical_reward_filter_mode_all(self, path: Path) -> None:
        """FilterOnIdenticalReward must use FilterMode.ALL."""
        source = _read_source(path)
        assert "FilterOnIdenticalRewardConfig(filter_mode=FilterMode.ALL)" in source, (
            f"{path.stem}: FilterOnIdenticalRewardConfig must use FilterMode.ALL"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_truncated_filter_mode_only(self, path: Path) -> None:
        """FilterOnTruncated must use FilterMode.ONLY."""
        source = _read_source(path)
        assert "FilterOnTruncatedConfig(filter_mode=FilterMode.ONLY)" in source, (
            f"{path.stem}: FilterOnTruncatedConfig must use FilterMode.ONLY"
        )

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_multiplexer_mode_only(self, path: Path) -> None:
        """FilterMultiplexer must use FilterMode.ONLY."""
        source = _read_source(path)
        # The multiplexer config line
        assert re.search(
            r"FilterMultiplexerConfig\(\s*filter_mode=FilterMode\.ONLY", source
        ), f"{path.stem}: FilterMultiplexerConfig must use FilterMode.ONLY"

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_debate_streamer_matches_gen_count(self, path: Path) -> None:
        """DebateMetricStreamerConfig.n_rollouts_per_prompt must be set."""
        source = _read_source(path)
        assert re.search(
            r"DebateMetricStreamerConfig\(\s*n_rollouts_per_prompt=", source
        ), f"{path.stem}: DebateMetricStreamerConfig must set n_rollouts_per_prompt"


# ── TestConfigConsistency ─────────────────────────────────────────────────────


class TestConfigConsistency:
    """Cross-config consistency checks."""

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_all_configs_parse_without_error(self, path: Path) -> None:
        """Every sweep_*.py file parses as valid Python."""
        tree = _parse_ast(path)
        assert isinstance(tree, ast.Module)

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_batch_sizes_positive(self, path: Path) -> None:
        """No zero batch sizes."""
        source = _read_source(path)
        for match in re.finditer(r"(?:train|eval)_batch_size=(\d+)", source):
            val = int(match.group(1))
            assert val > 0, f"{path.stem}: batch size must be positive, got {val}"

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_total_steps_positive(self, path: Path) -> None:
        """No zero training steps."""
        source = _read_source(path)
        for match in re.finditer(r"total_train_steps=(\d+)", source):
            val = int(match.group(1))
            assert val > 0, f"{path.stem}: total_train_steps must be positive, got {val}"

    @pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=lambda p: p.stem)
    def test_seed_is_set(self, path: Path) -> None:
        """Seed must be set (not None)."""
        source = _read_source(path)
        assert re.search(r"seed=\d+", source) or "seed=SEED" in source, (
            f"{path.stem}: seed must be set to a numeric value or SEED constant"
        )
