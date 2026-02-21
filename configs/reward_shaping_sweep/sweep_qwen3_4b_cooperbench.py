"""Qwen3-4B CooperBench: Agentic mode with DAPO loss and Docker-based verifiable rewards.

Agentic mode: Model interacts with code through OpenHands tools (bash, file editor)
across multiple turns. Each generation runs an agent loop in a Docker container
pointed at the vLLM sidecar. Much slower than single-turn (~5-15 min per rollout)
but produces dramatically better patches.

DAPO loss (default): Masks degenerate trajectories (timeouts, container errors, zero
reward, max turns), normalizes by max_sequence_length (no length bias), and clips
overlong trajectories. Set _LOSS_VARIATION to "grpo" or "gspo" to use standard
loss variants instead.

To disable agentic mode and use single-turn generation:
Set agentic_mode=False in actor config below.

Usage:
    # Preview (no submission)
    PYTHONPATH=/mnt/data/terry/repos/post_training/src:/mnt/data/terry/home/reward-training \
        /mnt/data/terry/repos/post_training/.venv/bin/python \
        configs/reward_shaping_sweep/sweep_qwen3_4b_cooperbench.py start

    # Submit to cluster
    PYTHONPATH=/mnt/data/terry/repos/post_training/src:/mnt/data/terry/home/reward-training \
        /mnt/data/terry/repos/post_training/.venv/bin/python \
        configs/reward_shaping_sweep/sweep_qwen3_4b_cooperbench.py --submit start
"""
from __future__ import annotations

from collections.abc import Iterable

import configs._image_reuse  # noqa: F401
import sweep

from post_training.canonicals import sweep_base
from post_training.flink import flink_zord
from post_training.flink.components import flink_reward_model
from post_training.flink.components.flink_comb_actor import FlinkCombActorConfig
from post_training.flink.components.flink_eval import FlinkEvalConfig
from post_training.flink.components.flink_input_data_preprocessors import CombItemsPreprocessorConfig
from post_training.flink.components.flink_learner_rloo import FlinkRlooLearnerConfig
from post_training.flink.components.flink_sampler_vllm_sidecar import FlinkVllmSidecarSamplerConfig
from post_training.flink.utils.endpoint_resolver import EndpointResolverConfig
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig
# Filters disabled for cold-start — see actor_outputs_streamers comment below
# from post_training.flink.components.flink_learning_filter.filtering_streamer import FilteringStreamerConfig
# from post_training.flink.components.flink_learning_filter.filter_dapo_degenerate import FilterDapoDegenerateConfig
# from post_training.flink.components.flink_learning_filter import FilterMode, FilterMultiplexerConfig
# from post_training.flink.components.flink_learning_filter.filter_zero_variance_group import FilterZeroVarianceGroupConfig

from configs.model_profiles import QWEN3_4B_INSTRUCT

# Model profile for Qwen3-4B
_PROFILE = QWEN3_4B_INSTRUCT
NUM_TRAINING_GPUS = _PROFILE.num_training_gpus      # 8
NUM_SAMPLING_GPUS = _PROFILE.num_sampling_gpus       # 8
MAX_SEQUENCE_LENGTH = _PROFILE.max_sequence_length   # 4096
CKPT_PATH = _PROFILE.ckpt_path

# Paths
RUN_CONFIG_PATH = "${HOME}/reward-training/run_configs/qwen3_4b_cooperbench.run"
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"
WANDB_PROJECT = "multiagent-debate-rl"
PRIORITY_CLASS = "dev-low"
SEED = 42

# vLLM sidecar
_VLLM_SIDECAR = "vllm"
_VLLM_PORT = 8000
_VLLM_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"

# Async Rollout-Train Decoupling:
# The Flink pipeline decouples rollout generation from gradient updates via queues.
# actors_queue_batches controls pipeline depth (how far ahead rollouts run).
# export_every_steps controls weight freshness for vLLM sidecar.
# Staleness = current_train_step - rollout_policy_step, bounded by queue depth.
# Target: 2-3 steps of staleness for 42% throughput improvement (Async-GRPO).

# Training params: 250 steps for stability validation (Phase 3)
_TOTAL_TRAIN_STEPS = 250
# Export weights every 2 steps to reduce export overhead.
# Rollouts using 1-step-old weights is acceptable (staleness ~2).
_EXPORT_EVERY_STEPS = 2
_TRAIN_BATCH_SIZE = 16  # 4 prompts x 4 gens/prompt for batch diversity (STAB-01)
_EVAL_BATCH_SIZE = 4
_GENERATIONS_PER_PROMPT = 4

# DAPO / Loss configuration
# To revert to standard GRPO/GSPO, change _LOSS_VARIATION and remove FilteringStreamerConfig
# from actor_outputs_streamers below. All other config stays the same.
_LOSS_VARIATION = "dapo"                    # "dapo" | "grpo" | "gspo" | "grpo_use_ref_policy_as_pi_old" | "gspo_use_ref_policy_as_pi_old"
_TEMPERATURE_TRAIN = 1.0                    # sampling temperature for training rollouts
_TEMPERATURE_EVAL = 0.6                     # sampling temperature for eval rollouts
_MAX_TURNS = 20                             # agentic_max_iterations
_DAPO_OVERLONG_COEFF = 0.5                  # halve gradient for overlong sequences
_DAPO_OVERLONG_THRESHOLD = 0.9              # trigger overlong clipping at 90% of max_sequence_length
_DAPO_MIN_VALID_PER_GROUP = 2               # minimum valid generations per prompt group
_DAPO_MAX_BATCH_MASK_RATIO = 0.5            # skip training step if >50% of batch is masked
_REDUNDANCY_FACTOR = 1.25                   # 25% extra rollouts for failure tolerance

# Phase 5: Docker Warmup and Rollout Speed Optimizations
# =====================================================
# 1. Container Pool: Docker containers persist across rollouts (no cold-start)
# 2. Async Decoupling: actors_queue_batches=32 keeps rollouts ahead of trainer
# 3. Weight Export: export_every_steps=2 with async background export
# 4. Trajectory Dispatch: Fine-grained interleaving via AgenticTrajectoryDispatcher
# 5. Redundant Rollouts: redundancy_factor=1.25 handles ~39% agentic failure rate
# 6. Rollout Timing: Per-task duration tracked for future sorted batching


class Qwen3_4bCooperBench(sweep_base.Sweep):
    settings: sweep.SweepSettings = sweep.SweepSettings(
        sweep_output_path="${HOME}/sweep_jobs/qwen3_4b_cooperbench/",
        cluster=sweep.Cluster.cw_us_east_04_prod,
    )
    fax: sweep.FaxConfig = sweep_base.PostTraining(
        partition=f"gpu_{NUM_TRAINING_GPUS}",
        queue=sweep.Queue.post_training_cohere_labs_queue,
        jobs_max_fanout=1,
        wandb_project=WANDB_PROJECT,
        priority_class=PRIORITY_CLASS,
        run_config=RUN_CONFIG_PATH,
        k8s_env_secrets_toml=K8S_SECRETS_PATH,
        ckpt_path=CKPT_PATH,
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/qwen3-4b-cooperbench/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            train_batch_size=_TRAIN_BATCH_SIZE,
            eval_batch_size=_EVAL_BATCH_SIZE,
            total_train_steps=_TOTAL_TRAIN_STEPS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            validation_every_steps=50,
            n_gradient_accumulation_steps=2,  # Effective batch of 32 (STAB-01)
            lr_schedule={
                "kwargs": {
                    "warmup_steps": 20,
                    "total_steps": _TOTAL_TRAIN_STEPS,
                    "peak_lr": 1e-6,
                    "end_lr": 1e-7,
                },
            },
            seed=SEED,
            objective={
                "loss": {
                    "kwargs": {
                        "rl_training_steps": 1,
                        "hard_update_ref_every_steps": _EXPORT_EVERY_STEPS,  # Must match export_every_steps
                        "preference": {
                            "loss_variation": _LOSS_VARIATION,
                            "beta": 0.03,
                            "avg_loglikelihood": False,
                            "generations_per_prompt": _GENERATIONS_PER_PROMPT,
                            # DAPO-specific loss parameters (ignored by GRPO/GSPO variants)
                            "dapo_overlong_coeff": _DAPO_OVERLONG_COEFF,
                            "dapo_overlong_threshold": _DAPO_OVERLONG_THRESHOLD,
                        },
                    },
                },
            },
            n_last_ckpts_to_keep=2,
            checkpoint_every_steps=50,
            minizord=flink_zord.FlinkZordConfig(
                samplers=dict(
                    sampler_key=FlinkVllmSidecarSamplerConfig(
                        vllm_endpoint=EndpointResolverConfig(
                            job_name=_VLLM_SIDECAR,
                            job_port=_VLLM_PORT,
                        ),
                        export_every_steps=_EXPORT_EVERY_STEPS,
                        export_dir=_VLLM_EXPORT_DIR,
                        temperature=0.8,  # Fallback if anneal not active
                        # Phase 3 (STAB-03): Linear temperature anneal 1.0 -> 0.6 over training
                        start_temperature=1.0,
                        end_temperature=0.6,
                        total_train_steps=_TOTAL_TRAIN_STEPS,
                    ),
                    eval_sampler_key=FlinkVllmSidecarSamplerConfig(
                        vllm_endpoint=EndpointResolverConfig(
                            job_name=_VLLM_SIDECAR,
                            job_port=_VLLM_PORT,
                        ),
                        export_every_steps=None,
                        export_dir=_VLLM_EXPORT_DIR,
                        temperature=_TEMPERATURE_EVAL,
                        p=0.95,
                        max_sequence_length_at_evaluation=MAX_SEQUENCE_LENGTH,
                    ),
                ),
                reward_models=dict(
                    reward_model_key=flink_reward_model.FlinkWrapRewardModelEndpointConfig(),
                ),
                input_data_preprocessor=CombItemsPreprocessorConfig(
                    env_name_remap={},
                    patch_data_item={},
                    patch_scenario_config={
                        "cooperbench": {
                            "docker_timeout": 300,
                        },
                    },
                ),
                actor=FlinkCombActorConfig(
                    sampler_endpoint_key="sampler_key",
                    agentic_mode=True,
                    agentic_max_iterations=_MAX_TURNS,
                    agentic_rollout_timeout=900,  # 15 min per rollout
                    agentic_vllm_base_url=f"http://{_VLLM_SIDECAR}:{_VLLM_PORT}/v1",
                    max_concurrent_trajectories=0,  # 0 = default to num_unrolls * redundancy_factor
                    agentic_redundancy_factor=_REDUNDANCY_FACTOR,
                    agentic_temperature=0.8,  # Phase 3 (STAB-03): configurable agent LLM temperature
                ),
                num_actors_per_batch_item=_GENERATIONS_PER_PROMPT,
                # Pipeline depth for async rollout-train decoupling.
                # Higher values = more overlap between rollout generation and training.
                # With 4 gens/prompt and batch_size=4, 32 queue batches means ~128 trajectories
                # can be buffered, allowing rollout workers to run 2-3 training steps ahead.
                actors_queue_batches=32,
                eval_actors_queue_batches=32,
                learner=FlinkRlooLearnerConfig(policy_gradient_loss="grpo"),
                actor_outputs_streamers=[
                    # NOTE: DapoFilter and ZeroVarianceFilter DISABLED for cold-start.
                    # At cold start, an untrained model produces mostly reward=0 trajectories.
                    # Both filters drop all-zero-reward prompt groups, causing the
                    # FilteringStreamer to loop forever and the learner to starve (no
                    # training steps execute). The DAPO loss handles zero-gradient items
                    # naturally — these filters are an optimization, not correctness.
                    # Re-enable once the model starts producing mixed-reward batches.
                    DebateMetricStreamerConfig(
                        n_rollouts_per_prompt=_GENERATIONS_PER_PROMPT,
                    ),
                ],
                eval=FlinkEvalConfig(
                    n_generation_steps=-1,
                    actor=FlinkCombActorConfig(
                        sampler_endpoint_key="eval_sampler_key",
                        patch_number_of_generation_per_prompt=1,
                        agentic_mode=True,
                        agentic_max_iterations=_MAX_TURNS,
                        agentic_rollout_timeout=900,
                        agentic_vllm_base_url=f"http://{_VLLM_SIDECAR}:{_VLLM_PORT}/v1",
                        max_concurrent_trajectories=0,
                        agentic_redundancy_factor=1.0,  # No redundancy for eval (single gen per prompt)
                        agentic_temperature=0.8,  # Phase 3 (STAB-03): match train temperature
                    ),
                ),
                log_train_generations_every_steps=1,
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
            hf_export=dict(override_mesh_for_local_gathering=_PROFILE.needs_mesh_override),
            model_export=dict(enabled=False),
            profile_every_steps=1,
            first_step_to_profile=1,
            read_extra_state=False,
        ),
        retries=0,
        kjobs_compute="",
        patch_kjobs_compute=dict(
            experimental_needs_docker_in_docker=True,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/data/1d/jax-cache/${USER}",
                "JAX_LOG_COMPILES": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1",
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
                        "env FAX_NUMBER_GPUS_PER_WORKER=1 VLLM_ATTENTION_BACKEND=FLASHINFER",
                        "python -m vinfer.main",
                        f"--exports-glob-pattern={_VLLM_EXPORT_DIR}/*/_HF_EXPORT_IS_COMPLETE",
                        f"--port {_VLLM_PORT}",
                        "--gpus-per-vllm-worker=1",
                        f"--max-model-len {MAX_SEQUENCE_LENGTH}",
                        "--enforce-eager",
                    ]
                ),
                ports=dict(web=_VLLM_PORT),
            )
        ],
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        return [{}]


if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(Qwen3_4bCooperBench(), debugging_artefacts=[__file__])
