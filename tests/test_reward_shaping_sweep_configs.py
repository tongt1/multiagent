"""Validation tests for reward shaping sweep configs (ECFG-01 through ECFG-08).

These tests verify that all 5 SWEEP configs are structurally correct and
identical except for reward_shaping_strategy and reward_shaping_params.

Since the `sweep` module is an internal cluster-only dependency, tests use
AST parsing and text comparison rather than runtime import + instantiation.
This approach validates the source-level invariants that guarantee an
apples-to-apples comparison experiment.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from configs.reward_shaping_sweep._base import (
    CKPT_PATH,
    GENERATIONS_PER_PROMPT,
    MAX_SEQUENCE_LENGTH,
    NUM_TRAINING_GPUS,
    NUM_SAMPLING_GPUS,
    PRIORITY_CLASS,
    WANDB_PROJECT,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "reward_shaping_sweep"

STRATEGY_FILES = {
    "identity": _CONFIGS_DIR / "sweep_identity.py",
    "difference_rewards": _CONFIGS_DIR / "sweep_difference_rewards.py",
    "potential_based": _CONFIGS_DIR / "sweep_potential_based.py",
    "coma_advantage": _CONFIGS_DIR / "sweep_coma_advantage.py",
    "reward_mixing": _CONFIGS_DIR / "sweep_reward_mixing.py",
}


def _read_source(path: Path) -> str:
    """Read file contents."""
    return path.read_text()


def _parse_ast(path: Path) -> ast.Module:
    """Parse file as AST."""
    source = _read_source(path)
    return ast.parse(source, filename=str(path))


def _strip_docstring_and_comments(source: str) -> str:
    """Remove module docstring and single-line comments to isolate code."""
    # Remove single-line comments
    lines = source.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines)


def _extract_debate_metric_streamer_block(source: str) -> str:
    """Extract the DebateMetricStreamerConfig(...) call from source.

    Returns the full constructor call including arguments.
    """
    # Find DebateMetricStreamerConfig( and match balanced parens
    pattern = r"DebateMetricStreamerConfig\("
    match = re.search(pattern, source)
    if not match:
        return ""

    start = match.start()
    # Count balanced parentheses
    depth = 0
    i = match.end() - 1  # Start at the opening paren
    for i in range(match.end() - 1, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                break

    return source[start : i + 1]


def _get_non_streamer_code(source: str) -> str:
    """Get all code EXCEPT the DebateMetricStreamerConfig block and docstring.

    This is the code that must be identical across all 5 configs.
    """
    # Remove the module docstring (triple-quoted string at top)
    # Find first triple-quote
    source_no_doc = re.sub(
        r'^""".*?"""', "", source, count=1, flags=re.DOTALL
    )

    # Remove the DebateMetricStreamerConfig(...) block
    streamer_block = _extract_debate_metric_streamer_block(source_no_doc)
    if streamer_block:
        source_no_doc = source_no_doc.replace(streamer_block, "DEBATE_METRIC_STREAMER_PLACEHOLDER")

    # Remove single-line comments (they may differ for strategy-specific notes)
    lines = source_no_doc.split("\n")
    code_lines = [line for line in lines if not line.lstrip().startswith("#")]
    return "\n".join(code_lines)


# ── Test 1: All five configs importable as valid Python (ECFG-01 to ECFG-05) ─

class TestAllFiveConfigsImportable:
    """ECFG-01 through ECFG-05: Each config can be parsed as valid Python."""

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_is_valid_python(self, strategy: str, path: Path) -> None:
        """Config file parses as valid Python AST."""
        assert path.exists(), f"Config file missing: {path}"
        tree = _parse_ast(path)
        assert isinstance(tree, ast.Module)

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_has_reward_shaping_sweep_class(self, strategy: str, path: Path) -> None:
        """Config file contains exactly one RewardShapingSweep class."""
        tree = _parse_ast(path)
        classes = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "RewardShapingSweep"
        ]
        assert len(classes) == 1, f"{strategy}: expected 1 RewardShapingSweep class, found {len(classes)}"

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_has_main_guard(self, strategy: str, path: Path) -> None:
        """Config file has if __name__ == '__main__' guard."""
        source = _read_source(path)
        assert 'if __name__ == "__main__"' in source, f"{strategy}: missing __main__ guard"

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_has_get_search_space(self, strategy: str, path: Path) -> None:
        """Config file defines get_search_space method."""
        tree = _parse_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_search_space":
                return
        pytest.fail(f"{strategy}: missing get_search_space method")


# ── Test 2: Configs differ only in reward shaping (ECFG-06) ─────────────────

class TestConfigsDifferOnlyInRewardShaping:
    """ECFG-06: All non-reward-shaping code is identical across all 5 configs."""

    def test_non_streamer_code_identical(self) -> None:
        """All 5 configs produce identical code when DebateMetricStreamerConfig and docstring are removed."""
        normalized_sources = {}
        for strategy, path in STRATEGY_FILES.items():
            source = _read_source(path)
            normalized = _get_non_streamer_code(source)
            # Also normalize whitespace for comparison
            normalized = re.sub(r"\s+", " ", normalized).strip()
            normalized_sources[strategy] = normalized

        reference = normalized_sources["identity"]
        for strategy, code in normalized_sources.items():
            if strategy == "identity":
                continue
            assert code == reference, (
                f"Config {strategy} differs from identity in non-reward-shaping code.\n"
                f"To debug: diff sweep_identity.py sweep_{strategy}.py"
            )

    def test_diff_only_in_debate_metric_streamer(self) -> None:
        """Diffing code after docstring shows changes ONLY in DebateMetricStreamerConfig."""
        import difflib

        def strip_module_docstring(source: str) -> str:
            """Remove the module docstring, keeping only code."""
            # Find end of module docstring (closing triple-quote after opening)
            in_docstring = False
            lines = source.split("\n")
            code_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i == 0 and stripped.startswith('"""'):
                    if stripped.endswith('"""') and len(stripped) > 3:
                        code_start = i + 1
                        break
                    in_docstring = True
                    continue
                if in_docstring and stripped.endswith('"""'):
                    code_start = i + 1
                    break
            return "\n".join(lines[code_start:])

        sources = {
            strategy: strip_module_docstring(_read_source(path))
            for strategy, path in STRATEGY_FILES.items()
        }

        identity_lines = sources["identity"].split("\n")

        for strategy, source in sources.items():
            if strategy == "identity":
                continue

            other_lines = source.split("\n")
            diff = list(difflib.unified_diff(
                identity_lines, other_lines,
                fromfile="identity", tofile=strategy,
                lineterm="",
            ))

            # Filter out diff headers
            changed_lines = [
                line for line in diff
                if (line.startswith("+") or line.startswith("-"))
                and not line.startswith("---")
                and not line.startswith("+++")
            ]

            for line in changed_lines:
                content = line[1:]  # Remove +/- prefix
                stripped = content.strip()
                if not stripped:
                    continue

                is_reward_shaping = (
                    "reward_shaping" in content
                    or "DebateMetricStreamer" in content
                )
                assert is_reward_shaping, (
                    f"Unexpected non-reward-shaping diff between identity and {strategy}: {line!r}"
                )


# ── Test 3: All configs use dev-low priority (ECFG-07) ──────────────────────

class TestAllConfigsUseDevLowPriority:
    """ECFG-07: All configs use post_training_cohere_labs_queue with dev-low priority."""

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_priority_class_dev_low(self, strategy: str, path: Path) -> None:
        """Config uses priority_class=PRIORITY_CLASS which resolves to 'dev-low'."""
        source = _read_source(path)
        assert "priority_class=PRIORITY_CLASS" in source, (
            f"{strategy}: must use PRIORITY_CLASS constant (not hardcoded string)"
        )
        # Verify the constant resolves correctly
        assert PRIORITY_CLASS == "dev-low"

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_queue_is_post_training(self, strategy: str, path: Path) -> None:
        """Config uses sweep.Queue.post_training_cohere_labs_queue."""
        source = _read_source(path)
        assert "sweep.Queue.post_training_cohere_labs_queue" in source, (
            f"{strategy}: must use post_training_cohere_labs_queue queue"
        )


# ── Test 4: All configs use SmolLM-135M (ECFG-07/08) ────────────────────────

class TestAllConfigsUseSmolLM135M:
    """ECFG-07/08: All configs target SmolLM-135M (1 GPU, 2048 seq, GCS ckpt)."""

    def test_base_constants_smollm(self) -> None:
        """Base constants reflect SmolLM-135M profile."""
        assert NUM_TRAINING_GPUS == 1, f"Expected 1 training GPU, got {NUM_TRAINING_GPUS}"
        assert NUM_SAMPLING_GPUS == 1, f"Expected 1 sampling GPU, got {NUM_SAMPLING_GPUS}"
        assert MAX_SEQUENCE_LENGTH == 2048, f"Expected 2048 seq len, got {MAX_SEQUENCE_LENGTH}"
        assert "smollm-135M" in CKPT_PATH, f"Expected smollm-135M in ckpt path, got {CKPT_PATH}"

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_partition_gpu_1(self, strategy: str, path: Path) -> None:
        """Config uses partition=gpu_1 (SmolLM fits on 1 GPU)."""
        source = _read_source(path)
        # Training partition
        assert 'partition=f"gpu_{NUM_TRAINING_GPUS}"' in source, (
            f"{strategy}: training partition must reference NUM_TRAINING_GPUS"
        )
        # Sidecar partition
        assert 'partition=f"gpu_{NUM_SAMPLING_GPUS}"' in source, (
            f"{strategy}: sidecar partition must reference NUM_SAMPLING_GPUS"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_imports_base_constants(self, strategy: str, path: Path) -> None:
        """Config imports from configs.reward_shaping_sweep._base."""
        source = _read_source(path)
        assert "from configs.reward_shaping_sweep._base import" in source, (
            f"{strategy}: must import from _base module"
        )


# ── Test 5: Strategy-specific params (ECFG-03, ECFG-04, ECFG-05) ───────────

class TestStrategySpecificParams:
    """ECFG-03/04/05: Each config has correct strategy-specific DebateMetricStreamerConfig."""

    def test_identity_no_reward_shaping(self) -> None:
        """Identity config has no reward_shaping_strategy or reward_shaping_params."""
        source = _read_source(STRATEGY_FILES["identity"])
        block = _extract_debate_metric_streamer_block(source)
        assert block, "identity: DebateMetricStreamerConfig not found"
        assert "reward_shaping_strategy" not in block, (
            "identity: should NOT have reward_shaping_strategy"
        )
        assert "reward_shaping_params" not in block, (
            "identity: should NOT have reward_shaping_params"
        )

    def test_difference_rewards_strategy(self) -> None:
        """Difference rewards config has correct strategy and empty params."""
        source = _read_source(STRATEGY_FILES["difference_rewards"])
        block = _extract_debate_metric_streamer_block(source)
        assert block, "difference_rewards: DebateMetricStreamerConfig not found"
        assert 'reward_shaping_strategy="difference_rewards"' in block
        assert "reward_shaping_params={}" in block

    def test_potential_based_strategy(self) -> None:
        """Potential-based config has gamma=0.99 and potential_type='debate_length'."""
        source = _read_source(STRATEGY_FILES["potential_based"])
        block = _extract_debate_metric_streamer_block(source)
        assert block, "potential_based: DebateMetricStreamerConfig not found"
        assert 'reward_shaping_strategy="potential_based"' in block
        assert '"gamma": 0.99' in block
        assert '"potential_type": "debate_length"' in block

    def test_coma_advantage_strategy(self) -> None:
        """COMA advantage config has n_rollouts_per_prompt in reward_shaping_params."""
        source = _read_source(STRATEGY_FILES["coma_advantage"])
        block = _extract_debate_metric_streamer_block(source)
        assert block, "coma_advantage: DebateMetricStreamerConfig not found"
        assert 'reward_shaping_strategy="coma_advantage"' in block
        assert '"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT' in block

    def test_reward_mixing_strategy(self) -> None:
        """Reward mixing config has alpha=0.5."""
        source = _read_source(STRATEGY_FILES["reward_mixing"])
        block = _extract_debate_metric_streamer_block(source)
        assert block, "reward_mixing: DebateMetricStreamerConfig not found"
        assert 'reward_shaping_strategy="reward_mixing"' in block
        assert '"alpha": 0.5' in block


# ── Test 6: All configs use math_debate env (ECFG-08) ───────────────────────

class TestAllConfigsUseMathDebateEnv:
    """ECFG-08: All configs remap math -> math_debate for debate environment."""

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_env_name_remap(self, strategy: str, path: Path) -> None:
        """Config has env_name_remap={'math': 'math_debate'}."""
        source = _read_source(path)
        assert 'env_name_remap={"math": "math_debate"}' in source, (
            f"{strategy}: must include math -> math_debate env remap"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_react_config_present(self, strategy: str, path: Path) -> None:
        """Config has react_config with reasoning_effort and grounding_style."""
        source = _read_source(path)
        assert "react_config" in source, f"{strategy}: missing react_config"
        assert "reasoning_effort" in source, (
            f"{strategy}: missing reasoning_effort in react_config"
        )
        assert "grounding_style" in source, (
            f"{strategy}: missing grounding_style in react_config"
        )


# ── Test 7: n_rollouts_per_prompt consistent (Pitfall 3/4) ──────────────────

class TestNRolloutsPerPromptConsistent:
    """Pitfall 3/4: n_rollouts_per_prompt matches GENERATIONS_PER_PROMPT everywhere."""

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_streamer_n_rollouts_per_prompt(self, strategy: str, path: Path) -> None:
        """DebateMetricStreamerConfig.n_rollouts_per_prompt uses GENERATIONS_PER_PROMPT constant."""
        source = _read_source(path)
        block = _extract_debate_metric_streamer_block(source)
        assert "n_rollouts_per_prompt=GENERATIONS_PER_PROMPT" in block, (
            f"{strategy}: DebateMetricStreamerConfig must use GENERATIONS_PER_PROMPT for n_rollouts_per_prompt"
        )

    def test_generations_per_prompt_value(self) -> None:
        """GENERATIONS_PER_PROMPT is 4 (must match loss config and COMA params)."""
        assert GENERATIONS_PER_PROMPT == 4

    def test_coma_dual_n_rollouts(self) -> None:
        """COMA config has n_rollouts_per_prompt in BOTH DebateMetricStreamerConfig and reward_shaping_params."""
        source = _read_source(STRATEGY_FILES["coma_advantage"])
        block = _extract_debate_metric_streamer_block(source)
        # Must appear as both a direct field and in reward_shaping_params
        assert "n_rollouts_per_prompt=GENERATIONS_PER_PROMPT" in block, (
            "COMA: n_rollouts_per_prompt must be in DebateMetricStreamerConfig"
        )
        assert '"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT' in block, (
            "COMA: n_rollouts_per_prompt must also be in reward_shaping_params"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_num_actors_per_batch_item(self, strategy: str, path: Path) -> None:
        """num_actors_per_batch_item must equal GENERATIONS_PER_PROMPT."""
        source = _read_source(path)
        assert "num_actors_per_batch_item=GENERATIONS_PER_PROMPT" in source, (
            f"{strategy}: num_actors_per_batch_item must use GENERATIONS_PER_PROMPT"
        )


# ── Test 8: No excluded patterns (SmolLM-specific) ──────────────────────────

class TestNoExcludedPatterns:
    """SmolLM-135M configs must NOT include 7B-specific patterns."""

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_no_advanced_logging(self, strategy: str, path: Path) -> None:
        """SmolLM configs should not include advanced_logging block."""
        source = _read_source(path)
        assert "advanced_logging" not in source, (
            f"{strategy}: SmolLM configs should not include advanced_logging"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_no_model_to_query(self, strategy: str, path: Path) -> None:
        """SmolLM configs should not include MODEL_TO_QUERY."""
        source = _read_source(path)
        assert "MODEL_TO_QUERY" not in source, (
            f"{strategy}: SmolLM configs should not include MODEL_TO_QUERY"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_no_mesh_override(self, strategy: str, path: Path) -> None:
        """SmolLM configs should not include override_mesh_for_local_gathering."""
        source = _read_source(path)
        assert "override_mesh_for_local_gathering" not in source, (
            f"{strategy}: SmolLM (tp=1) should not include mesh override"
        )


# -- Test 9: All configs share WANDB_PROJECT constant (OBSV-02) ---------------

class TestAllConfigsShareWandbProject:
    """OBSV-02: All configs import and use the same WANDB_PROJECT from _base.py."""

    def test_wandb_project_value(self) -> None:
        """WANDB_PROJECT from _base.py equals 'multiagent-debate-rl'."""
        assert WANDB_PROJECT == "multiagent-debate-rl", (
            f"Expected WANDB_PROJECT='multiagent-debate-rl', got '{WANDB_PROJECT}'"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_imports_from_base(self, strategy: str, path: Path) -> None:
        """Each config imports from configs.reward_shaping_sweep._base."""
        source = _read_source(path)
        assert "from configs.reward_shaping_sweep._base import" in source, (
            f"{strategy}: must import from _base module"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_config_references_wandb_project(self, strategy: str, path: Path) -> None:
        """Each config source contains WANDB_PROJECT reference."""
        source = _read_source(path)
        assert "WANDB_PROJECT" in source, (
            f"{strategy}: must reference WANDB_PROJECT constant from _base"
        )

    def test_base_wandb_project_via_ast(self) -> None:
        """_base.py defines WANDB_PROJECT as a string literal via AST inspection."""
        base_path = _CONFIGS_DIR / "_base.py"
        tree = _parse_ast(base_path)

        wandb_project_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "WANDB_PROJECT":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            wandb_project_value = node.value.value

        assert wandb_project_value is not None, (
            "_base.py: WANDB_PROJECT not found as a string literal assignment"
        )
        assert wandb_project_value == "multiagent-debate-rl", (
            f"_base.py: WANDB_PROJECT={wandb_project_value!r}, expected 'multiagent-debate-rl'"
        )

    @pytest.mark.parametrize("strategy,path", STRATEGY_FILES.items())
    def test_all_configs_share_wandb_project(self, strategy: str, path: Path) -> None:
        """Each config uses the shared WANDB_PROJECT constant (not a hardcoded string)."""
        source = _read_source(path)
        # Must import WANDB_PROJECT from _base
        assert "WANDB_PROJECT" in source, (
            f"{strategy}: must use WANDB_PROJECT constant"
        )
        # Must use it in wandb_project= assignment (not a hardcoded string)
        assert "wandb_project=WANDB_PROJECT" in source, (
            f"{strategy}: must use wandb_project=WANDB_PROJECT (not hardcoded string)"
        )
