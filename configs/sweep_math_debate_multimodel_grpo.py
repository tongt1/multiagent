"""SWEEP config for multi-model math debate GRPO training.

Single-sidecar approach: uses the same single-model debate config pattern but
with multi_model metadata logged for provenance and future dual-model support.

The Comb math_debate environment runs the full solver-verifier-judge flow in a
single multi-turn rollout and does not support splitting turns across different
samplers. Therefore, we use a single model serving all roles, with metadata
indicating multi-model intent. True dual-model training (separate sidecars per
role) requires Comb environment changes that are deferred.

For freeze_roles experiments:
- freeze_roles metadata is logged to W&B for provenance
- With a single model, freeze has no gradient effect (same weights for all roles)
- True role-specific gradient masking requires dual-model Comb support

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

# =============================================================================
# Multi-model checkpoint paths (for metadata / provenance)
# =============================================================================
SOLVER_CKPT_PATH = "c3_7B_12-2024_command_release"
VERIFIER_CKPT_PATH = "c3_7B_12-2024_command_release"

# =============================================================================
# Freeze configuration (logged as metadata; no gradient effect with single model)
# =============================================================================
# Set to ["verifier", "judge"] to freeze verifier model, train only solver
# Set to ["solver"] to freeze solver model, train only verifier
# Set to [] to train both models simultaneously
FREEZE_ROLES: list[str] = []

# =============================================================================
# Single-sidecar vLLM configuration (same pattern as sweep_math_debate_grpo.py)
# =============================================================================
_VLLM_SIDECAR = "vllm"
_VLLM_PORT = 8000
_VLLM_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"

# =============================================================================
# Shared training hyperparameters (MUST be identical between debate and baseline)
# =============================================================================
NUM_TRAINING_GPUS = 8
NUM_SAMPLING_GPUS = 16
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TOTAL_TRAIN_STEPS = 300
MAX_SEQUENCE_LENGTH = 4096
LEARNING_RATE = 3e-6
KL_BETA = 0.03
GENERATIONS_PER_PROMPT = 4
EXPORT_EVERY_STEPS = 20
HARD_UPDATE_REF_EVERY_STEPS = 20
SEED = 42

# Model and infrastructure settings
MODEL_TO_QUERY = "command-a-03-2025"
IS_STAGING = True
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"

CKPT_PATH = SOLVER_CKPT_PATH


class MathDebateMultiModelGRPO(sweep_base.Sweep):
    """SWEEP config for multi-model debate GRPO (single-sidecar approach).

    Uses the standard single-model debate config pattern with multi_model
    metadata logged to W&B for provenance. True dual-model training with
    separate sidecars per role is deferred until Comb supports multiple
    sampler endpoints.
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
        ckpt_path=CKPT_PATH,
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/debate-multimodel/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            # Multi-model metadata (stored in metadata to avoid RunConfig extra_forbidden)
            metadata=dict(
                multi_model=dict(
                    enabled=True,
                    solver_ckpt=SOLVER_CKPT_PATH,
                    verifier_ckpt=VERIFIER_CKPT_PATH,
                    freeze_roles=FREEZE_ROLES,
                    approach="single_sidecar",
                    note="True dual-sidecar deferred until Comb supports multiple sampler endpoints",
                ),
            ),
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
            n_gradient_accumulation_steps=1,
            lr_schedule={
                "kwargs": {
                    "peak_lr": LEARNING_RATE,
                    "end_lr": LEARNING_RATE,
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
            minizord=flink_zord.FlinkZordConfig(
                samplers=dict(
                    sampler_key=FlinkVllmSidecarSamplerConfig(
                        vllm_endpoint=EndpointResolverConfig(
                            job_name=_VLLM_SIDECAR,
                            job_port=_VLLM_PORT,
                        ),
                        export_every_steps=EXPORT_EVERY_STEPS,
                        export_dir=_VLLM_EXPORT_DIR,
                        temperature=1,
                    ),
                    eval_sampler_key=FlinkVllmSidecarSamplerConfig(
                        vllm_endpoint=EndpointResolverConfig(
                            job_name=_VLLM_SIDECAR,
                            job_port=_VLLM_PORT,
                        ),
                        export_every_steps=None,
                        export_dir=_VLLM_EXPORT_DIR,
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
                    sampler_endpoint_key="sampler_key",
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
                    DebateMetricStreamerConfig(
                        n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
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
        retries=0,
        kjobs_compute="",
        patch_kjobs_compute=dict(
            env={
                "JAX_COMPILATION_CACHE_DIR": "/data/1d/jax-cache/${USER}",
                "JAX_LOG_COMPILES": "1",
            },
        ),
        sidecars=[
            sweep.SidecarConfig(
                name=_VLLM_SIDECAR,
                repo=sweep.RepoConfig(
                    directory=sweep_base.get_current_repo_directory() + "/vinfer",
                    kind=sweep.RepoKinds.vllm,
                ),
                partition=f"gpu_{NUM_SAMPLING_GPUS}",
                command=" ".join(
                    [
                        "python -m vinfer.main",
                        f"--exports-glob-pattern={_VLLM_EXPORT_DIR}/*/_HF_EXPORT_IS_COMPLETE",
                        f"--port {_VLLM_PORT}",
                        "--gpus-per-vllm-worker=1",
                    ]
                ),
                ports=dict(web=_VLLM_PORT),
            )
        ],
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        """Default: single run with multi_model metadata."""
        return [{}]


if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(MathDebateMultiModelGRPO(), debugging_artefacts=[__file__])
