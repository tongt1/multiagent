"""SWEEP config for multi-model math debate GRPO training.

Trains on debate trajectories (solver-verifier-judge) using SEPARATE models
for solver and verifier/judge roles. Extends the single-model MathDebateGRPO
pattern with dual vLLM sidecars, asymmetric GPU partitioning, and freeze support.

Key differences from single-model sweep_math_debate_grpo.py:
- Two vLLM sidecars: vllm_solver and vllm_verifier (separate export dirs)
- Two checkpoint paths: SOLVER_CKPT_PATH and VERIFIER_CKPT_PATH
- Port separation: solver on 8000, verifier on 8001
- Export directory separation: .../solver/ and .../verifier/ (prevents weight collision)
- freeze_roles config: can freeze one model's weights while training the other
- Asymmetric GPU partitioning: e.g., 7B solver gets more GPUs than 3B verifier

Usage:
    # Preview (no submission)
    uv run python configs/sweep_math_debate_multimodel_grpo.py start

    # Submit to cluster
    uv run python configs/sweep_math_debate_multimodel_grpo.py --submit start

OWNERS: Multiagent Debate RL Experiment
"""

from __future__ import annotations

from collections.abc import Iterable

import configs._image_reuse  # noqa: F401 -- skip Docker builds when REUSE_IMAGE_TAG is set
import sweep

from post_training.canonicals import sweep_base
from post_training.flink import flink_zord
from post_training.flink.components import flink_reward_model
from post_training.flink.components.flink_comb_actor import FlinkCombActorConfig
from post_training.flink.components.flink_eval import FlinkEvalConfig
from post_training.flink.components.flink_input_data_preprocessors import CombItemsPreprocessorConfig
from post_training.flink.components.flink_learner_rloo import FlinkRlooLearnerConfig
from post_training.flink.components.flink_learning_filter import FilterMode, FilterMultiplexerConfig
from post_training.flink.components.flink_learning_filter.filter_on_identical_reward import (
    FilterOnIdenticalRewardConfig,
)
from post_training.flink.components.flink_learning_filter.filter_on_truncated import FilterOnTruncatedConfig
from post_training.flink.components.flink_learning_filter.filtering_streamer import FilteringStreamerConfig
from post_training.flink.components.flink_sampler_vllm_sidecar import FlinkVllmSidecarSamplerConfig
from post_training.flink.utils.endpoint_resolver import EndpointResolverConfig

# Debate-specific enrichment (Phase 5)
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig

# Multi-model integration (Phase 10)
from src.training.multi_model.config import MultiModelConfig
from src.training.multi_model.sweep_integration import (
    build_patch_run_config,
    build_sidecar_configs,
    compute_gpu_split,
)

# =============================================================================
# Multi-model checkpoint paths
# =============================================================================
# Solver model: generates solutions in the debate
SOLVER_CKPT_PATH = "c3_7B_12-2024_command_release"

# Verifier/judge model: evaluates and judges solver outputs
# Same default checkpoint for testing; set different checkpoint for asymmetric training
VERIFIER_CKPT_PATH = "c3_7B_12-2024_command_release"

# =============================================================================
# Multi-model GPU allocation
# =============================================================================
# Training GPUs (shared for now -- both models train on the same GPU partition)
NUM_TRAINING_GPUS = 8

# Sampling GPUs (must sum to total sampling capacity)
# For same-size models (both 7B): split evenly
# For asymmetric models (e.g., 7B + 3B): allocate proportionally to model size
#   - 7B needs more memory per worker -> more GPUs
#   - 3B fits smaller workers -> fewer GPUs
SOLVER_SAMPLING_GPUS = 8
VERIFIER_SAMPLING_GPUS = 8

# =============================================================================
# Freeze configuration
# =============================================================================
# Controls which models receive gradient updates during training.
# This is key for curriculum-style training where you freeze one model
# and train the other, then swap.
#
# Set to ["verifier", "judge"] to freeze verifier model, train only solver
# Set to ["solver"] to freeze solver model, train only verifier
# Set to [] to train both models simultaneously
FREEZE_ROLES: list[str] = []

# =============================================================================
# Per-model learning rates
# =============================================================================
# Can differ for asymmetric training. Smaller models may benefit from
# lower learning rates to prevent catastrophic forgetting.
SOLVER_LEARNING_RATE = 3e-6
VERIFIER_LEARNING_RATE = 3e-6

# =============================================================================
# Shared training hyperparameters
# =============================================================================
# These MUST be identical between debate and baseline for fair comparison
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TOTAL_TRAIN_STEPS = 300
MAX_SEQUENCE_LENGTH = 4096
KL_BETA = 0.03
GENERATIONS_PER_PROMPT = 4
EXPORT_EVERY_STEPS = 20  # Must match HARD_UPDATE_REF_EVERY_STEPS per model
HARD_UPDATE_REF_EVERY_STEPS = 20
SEED = 42

# =============================================================================
# Model and infrastructure settings
# =============================================================================
MODEL_TO_QUERY = "command-a-03-2025"
IS_STAGING = True
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"

# =============================================================================
# Dual vLLM sidecar configuration
# =============================================================================
# Why two sidecars: Each sidecar serves a different model checkpoint.
# The solver sidecar exports solver model weights, and the verifier sidecar
# exports verifier model weights. They MUST have separate export directories
# to prevent weight file collisions during async model updates.

# Solver sidecar: serves the solver model for generating debate solutions
_VLLM_SOLVER_SIDECAR = "vllm_solver"
_VLLM_SOLVER_PORT = 8000
_VLLM_SOLVER_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}/solver"

# Verifier sidecar: serves the verifier/judge model for evaluating solutions
# IMPORTANT: Must use a DIFFERENT port than solver to avoid binding conflicts
_VLLM_VERIFIER_SIDECAR = "vllm_verifier"
_VLLM_VERIFIER_PORT = 8001
_VLLM_VERIFIER_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}/verifier"

# Build multi-model config for integration helpers
_MULTI_MODEL_CONFIG = MultiModelConfig(
    solver_ckpt=SOLVER_CKPT_PATH,
    verifier_ckpt=VERIFIER_CKPT_PATH,
    solver_model_size="7B",  # Used for GPU split calculation
    verifier_model_size="7B",  # Change to "3B" etc. for asymmetric models
    freeze_roles=FREEZE_ROLES,
)


class MathDebateMultiModelGRPO(sweep_base.Sweep):
    """SWEEP config for dual-model debate GRPO training.

    Extends the standard MathDebateGRPO pattern with:
    - Two vLLM sidecars (solver + verifier) with independent export directories
    - Separate checkpoint paths for each model
    - freeze_roles support for curriculum-style training
    - Asymmetric GPU partitioning based on model sizes

    How freeze_roles interacts with gradient computation:
    - When a role is in freeze_roles, its model key is marked as frozen
    - The Flink training loop skips gradient computation for frozen model keys
    - This allows training one model while keeping the other fixed as an anchor
    - Example: freeze ["verifier", "judge"] -> only solver model gets gradients

    How to switch between same-size and asymmetric-size models:
    - Same size: Set both ckpt paths to same-size models, GPU split is even
    - Asymmetric: Set different ckpt paths (e.g., 7B solver + 3B verifier),
      update solver_model_size/verifier_model_size in _MULTI_MODEL_CONFIG,
      and GPU split will adjust proportionally
    """

    settings: sweep.SweepSettings = sweep.SweepSettings(
        sweep_output_path="${HOME}/sweep_jobs/multiagent_debate_rl/",
        cluster=sweep.Cluster.cw_us_east_04_prod,
    )
    fax: sweep.FaxConfig = sweep_base.PostTraining(
        partition=f"gpu_{NUM_TRAINING_GPUS}",
        queue=sweep.Queue.post_training_cohere_labs_queue,
        jobs_max_fanout=1,
        wandb_project="multiagent-debate-rl",
        priority_class="dev-high",
        run_config="${HOME}/repos/post_training/post_training/experimental/comb_flink/configs/rloo_7B_math.run",
        k8s_env_secrets_toml=K8S_SECRETS_PATH,
        # Use solver checkpoint as primary (Flink's main model init)
        ckpt_path=SOLVER_CKPT_PATH,
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/debate-multimodel/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=build_patch_run_config(
            _MULTI_MODEL_CONFIG,
            dict(
                # v1.1 Phase 5: Gradient norm logging
                advanced_logging=dict(
                    enabled=True,
                    norm_granularity=["global"],
                    norm_target_trees=["grad", "update"],
                    log_histograms=False,
                ),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                total_train_steps=TOTAL_TRAIN_STEPS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                validation_every_steps=100,
                n_gradient_accumulation_steps=4,
                lr_schedule={
                    "kwargs": {
                        "peak_lr": SOLVER_LEARNING_RATE,
                        "end_lr": SOLVER_LEARNING_RATE,
                    },
                },
                seed=SEED,
                objective={
                    "loss": {
                        "kwargs": {
                            "rl_training_steps": 1,
                            "hard_update_ref_every_steps": HARD_UPDATE_REF_EVERY_STEPS,
                            "preference": {
                                "loss_variation": "gspo_use_ref_policy_as_pi_old",
                                "beta": KL_BETA,
                                "avg_loglikelihood": False,
                                "generations_per_prompt": GENERATIONS_PER_PROMPT,
                            },
                        },
                    },
                },
                n_last_ckpts_to_keep=2,
                checkpoint_every_steps=100,
                # Dual-sidecar FlinkZordConfig:
                # - solver_sampler_key connects to vllm_solver sidecar
                # - verifier_sampler_key connects to vllm_verifier sidecar
                # Each sampler has its own export directory to prevent weight collision
                minizord=flink_zord.FlinkZordConfig(
                    samplers=dict(
                        # Solver model sampler -- generates debate solutions
                        solver_sampler_key=FlinkVllmSidecarSamplerConfig(
                            vllm_endpoint=EndpointResolverConfig(
                                job_name=_VLLM_SOLVER_SIDECAR,
                                job_port=_VLLM_SOLVER_PORT,
                            ),
                            export_every_steps=EXPORT_EVERY_STEPS,
                            export_dir=_VLLM_SOLVER_EXPORT_DIR,
                            temperature=1,
                        ),
                        # Verifier model sampler -- evaluates and judges solutions
                        verifier_sampler_key=FlinkVllmSidecarSamplerConfig(
                            vllm_endpoint=EndpointResolverConfig(
                                job_name=_VLLM_VERIFIER_SIDECAR,
                                job_port=_VLLM_VERIFIER_PORT,
                            ),
                            export_every_steps=EXPORT_EVERY_STEPS,
                            export_dir=_VLLM_VERIFIER_EXPORT_DIR,
                            temperature=1,
                        ),
                        # Evaluation sampler -- uses solver model for eval consistency
                        eval_sampler_key=FlinkVllmSidecarSamplerConfig(
                            vllm_endpoint=EndpointResolverConfig(
                                job_name=_VLLM_SOLVER_SIDECAR,
                                job_port=_VLLM_SOLVER_PORT,
                            ),
                            export_every_steps=None,
                            export_dir=_VLLM_SOLVER_EXPORT_DIR,
                            temperature=0.6,
                            p=0.95,
                            max_sequence_length_at_evaluation=48000,
                        ),
                    ),
                    reward_models=dict(
                        reward_model_key=flink_reward_model.FlinkWrapRewardModelEndpointConfig(),
                    ),
                    input_data_preprocessor=CombItemsPreprocessorConfig(
                        env_name_remap={"math": "math_debate"},
                        patch_data_item={
                            "math_debate": dict(
                                agent_trajectory=dict(
                                    preamble=dict(
                                        react_config=dict(
                                            reasoning_effort="ON",
                                            grounding_style="SPAN",
                                        )
                                    ),
                                ),
                            )
                        },
                        patch_scenario_config={},
                    ),
                    actor=FlinkCombActorConfig(
                        sampler_endpoint_key="solver_sampler_key",
                    ),
                    num_actors_per_batch_item=GENERATIONS_PER_PROMPT,
                    actors_queue_batches=32,
                    eval_actors_queue_batches=32,
                    learner=FlinkRlooLearnerConfig(policy_gradient_loss="grpo"),
                    actor_outputs_streamers=[
                        FilteringStreamerConfig(
                            filter=FilterMultiplexerConfig(
                                filter_mode=FilterMode.ONLY,
                                filter_configs=[
                                    FilterOnIdenticalRewardConfig(filter_mode=FilterMode.ALL),
                                    FilterOnTruncatedConfig(filter_mode=FilterMode.ONLY),
                                ],
                            )
                        ),
                        # Phase 5: Debate metric enrichment
                        # Phase 8: Reward shaping (applied after rollout strategy)
                        # Phase 9: Rollout strategy (applied before reward shaping)
                        DebateMetricStreamerConfig(
                            n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
                            # rollout_strategy="best_of_n",
                            # rollout_strategy_params={},
                            # reward_shaping_strategy="reward_mixing",
                            # reward_shaping_params={"alpha": 0.5},
                        ),
                    ],
                    eval=FlinkEvalConfig(
                        n_generation_steps=-1,
                        actor=FlinkCombActorConfig(
                            sampler_endpoint_key="eval_sampler_key",
                            patch_number_of_generation_per_prompt=1,
                        ),
                    ),
                    log_train_generations_every_steps=10,
                    save_debug_data_every_steps=1,
                ).model_dump(),
                likelihood_evals=None,
                finetuning=dict(
                    lora=dict(
                        enabled=False,
                        rank=8,
                        alpha=8.0,
                    ),
                ),
                ckpt=dict(force_at_init=False),
                validation=dict(force_at_init=False),
                hf_export=dict(
                    override_mesh_for_local_gathering=True,
                ),
                model_export=dict(enabled=False),
                profile_every_steps=5,
                first_step_to_profile=1,
                read_extra_state=False,
            ),
        ),
        retries=0,
        kjobs_compute="",
        patch_kjobs_compute=dict(
            env={
                "JAX_COMPILATION_CACHE_DIR": "/data/1d/jax-cache/${USER}",
                "JAX_LOG_COMPILES": "1",
            },
        ),
        # Dual vLLM sidecars: each serves a different model checkpoint
        sidecars=[
            # Solver sidecar: serves the solver model
            sweep.SidecarConfig(
                name=_VLLM_SOLVER_SIDECAR,
                repo=sweep.RepoConfig(
                    directory=sweep_base.get_current_repo_directory() + "/vinfer",
                    kind=sweep.RepoKinds.vllm,
                ),
                partition=f"gpu_{SOLVER_SAMPLING_GPUS}",
                command=" ".join(
                    [
                        "python -m vinfer.main",
                        f"--exports-glob-pattern={_VLLM_SOLVER_EXPORT_DIR}/*/_HF_EXPORT_IS_COMPLETE",
                        f"--port {_VLLM_SOLVER_PORT}",
                        "--gpus-per-vllm-worker=1",  # 7B model fits on 1 GPU
                    ]
                ),
                ports=dict(web=_VLLM_SOLVER_PORT),
            ),
            # Verifier sidecar: serves the verifier/judge model
            sweep.SidecarConfig(
                name=_VLLM_VERIFIER_SIDECAR,
                repo=sweep.RepoConfig(
                    directory=sweep_base.get_current_repo_directory() + "/vinfer",
                    kind=sweep.RepoKinds.vllm,
                ),
                partition=f"gpu_{VERIFIER_SAMPLING_GPUS}",
                command=" ".join(
                    [
                        "python -m vinfer.main",
                        f"--exports-glob-pattern={_VLLM_VERIFIER_EXPORT_DIR}/*/_HF_EXPORT_IS_COMPLETE",
                        f"--port {_VLLM_VERIFIER_PORT}",
                        "--gpus-per-vllm-worker=1",  # 7B model fits on 1 GPU (adjust for larger models)
                    ]
                ),
                ports=dict(web=_VLLM_VERIFIER_PORT),
            ),
        ],
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        """Default: single run with both models training.

        To sweep over freeze configurations:
        # Train solver only vs train verifier only vs train both
        # return [
        #     {"freeze_roles": []},  # Train both
        #     {"freeze_roles": ["verifier", "judge"]},  # Train solver only
        #     {"freeze_roles": ["solver"]},  # Train verifier only
        # ]

        To sweep over model size combinations:
        # return [
        #     {"solver_ckpt": "c3_7B_...", "verifier_ckpt": "c3_7B_..."},  # Both 7B
        #     {"solver_ckpt": "c3_7B_...", "verifier_ckpt": "c3_3B_..."},  # 7B+3B asymmetric
        # ]
        """
        return [{}]


if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(MathDebateMultiModelGRPO(), debugging_artefacts=[__file__])
