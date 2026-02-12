"""SWEEP config for math debate GRPO training.

Trains on debate trajectories (solver-verifier-judge) using multi-turn rollouts.
Uses 'math_debate' Comb environment via env_name_remap from base 'math' data items.

Usage:
    # Preview (no submission)
    uv run python configs/sweep_math_debate_grpo.py start

    # Submit to cluster
    uv run python configs/sweep_math_debate_grpo.py --submit start

OWNERS: Multiagent Debate RL Experiment
"""

from __future__ import annotations

from collections.abc import Iterable

import configs._image_reuse  # noqa: F401 — skip Docker builds when REUSE_IMAGE_TAG is set
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
# Moved to post_training for proper Ray serialization
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig
# GPUStatsStreamer disabled: @ray.remote at module scope + kubectl subprocesses
# destabilize Ray cluster and add memory pressure. Rewrite before re-enabling.

# vLLM sidecar configuration
_VLLM_SIDECAR = "vllm"
_VLLM_PORT = 8000
_VLLM_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"

# Shared training hyperparameters (MUST be identical between debate and baseline)
# Using 7B model for faster iteration - switch to 8x15B for production runs
NUM_TRAINING_GPUS = 8  # 24 total GPUs: 8 train + 16 vLLM
NUM_SAMPLING_GPUS = 16  # 7B fits on 1 GPU per vLLM worker (16 workers for throughput)
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
TOTAL_TRAIN_STEPS = 300  # Longer run for learning curves
MAX_SEQUENCE_LENGTH = 4096
LEARNING_RATE = 3e-6  # Standard for 7B GRPO
KL_BETA = 0.03
GENERATIONS_PER_PROMPT = 4
EXPORT_EVERY_STEPS = 20  # Was 5 — each export costs ~40s, so 5→20 saves ~600s over 100 steps
HARD_UPDATE_REF_EVERY_STEPS = 20  # Must match EXPORT_EVERY_STEPS
SEED = 42

# Model and infrastructure settings
MODEL_TO_QUERY = "command-a-03-2025"  # Staging model (no -pld-rl suffix)
IS_STAGING = True  # Use staging API for Hive/Blobheart (requires CO_API_KEY_STAGING)
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"

# 7B checkpoint for testing - switch to 8x15B for production
# Using named checkpoint alias (resolved by sweep infrastructure)
CKPT_PATH = "c3_7B_12-2024_command_release"

# NOTE: Using default data from run_config (AIME 2024/2025 validation sets)
# For production, set custom data paths and uncomment data_dir_dict override below


class MathDebateGRPO(sweep_base.Sweep):
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
        k8s_env_secrets_toml=K8S_SECRETS_PATH,  # Contains CO_API_KEY_STAGING (unlimited key)
        ckpt_path=CKPT_PATH,
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/debate/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            # v1.1 Phase 5: Enable gradient norm logging for debate training monitoring
            # Logs train/grad_norm and train/update_norm to W&B (maps to debate/grad/global_norm in dashboard)
            advanced_logging=dict(
                enabled=True,
                norm_granularity=["global"],  # Global gradient norm only (minimal overhead)
                norm_target_trees=["grad", "update"],  # Log both gradient and param update norms
                log_histograms=False,  # Disable histograms to reduce W&B overhead
            ),
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            total_train_steps=TOTAL_TRAIN_STEPS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            validation_every_steps=100,
            n_gradient_accumulation_steps=4,  # 32*4=128 effective batch
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
            # Using default data from run_config
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
                    # Remap "math" data items to use the multi-turn "math_debate" env
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
                actors_queue_batches=32,  # Must be >= GENERATIONS_PER_PROMPT * (TRAIN_BATCH_SIZE / grad_accum) = 4*4
                eval_actors_queue_batches=32,  # Must be >= GENERATIONS_PER_PROMPT * EVAL_BATCH_SIZE = 4*8
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
                    # Computes per-role rewards and zero-advantage detection metrics
                    # Phase 8: Reward shaping integration
                    # reward_shaping_strategy selects the shaping algorithm:
                    #   "" or omitted  = identity (passthrough, default -- no behavior change)
                    #   "reward_mixing" = alpha*G + (1-alpha)*local  (params: {"alpha": 0.5})
                    #   "difference_rewards" = D_i = G - G_{-i}  (no extra params)
                    #   "coma_advantage"     = COMA counterfactual advantage  (params: {"n_rollouts_per_prompt": N})
                    #   "potential_based"    = Ng et al. potential shaping    (params: {"gamma": 0.99, "potential_type": "debate_length"})
                    DebateMetricStreamerConfig(
                        n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
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
                log_train_generations_every_steps=10,  # More frequent for quick feedback
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
            ckpt=dict(force_at_init=False),  # Skip initial ckpt save (~60s)
            validation=dict(force_at_init=False),  # Skip initial eval to avoid rate limiting
            hf_export=dict(
                override_mesh_for_local_gathering=True,  # Use dedicated export mesh (tp=8) to avoid SPMD rematerialization
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
                    directory=sweep_base.get_current_repo_directory() + "/vinfer", kind=sweep.RepoKinds.vllm
                ),
                partition=f"gpu_{NUM_SAMPLING_GPUS}",
                command=" ".join(
                    [
                        "python -m vinfer.main",
                        f"--exports-glob-pattern={_VLLM_EXPORT_DIR}/*/_HF_EXPORT_IS_COMPLETE",
                        f"--port {_VLLM_PORT}",
                        "--gpus-per-vllm-worker=1",  # 7B model fits on 1 GPU
                    ]
                ),
                ports=dict(web=_VLLM_PORT),
            )
        ],
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        return [{}]


if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(MathDebateGRPO(), debugging_artefacts=[__file__])
