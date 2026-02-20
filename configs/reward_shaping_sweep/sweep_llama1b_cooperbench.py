"""Llama 1B CooperBench: End-to-end pipeline validation with no filtering.

Runs 2 training steps with Llama-3.2-1B-Instruct on CooperBench data.
No FilteringStreamerConfig — all actor outputs reach the learner.
Format-based reward: 1.0 if model produces a git diff, 0.0 otherwise.

Usage:
    # Preview (no submission)
    uv run python configs/reward_shaping_sweep/sweep_llama1b_cooperbench.py start

    # Submit to cluster
    uv run python configs/reward_shaping_sweep/sweep_llama1b_cooperbench.py --submit start
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

from configs.model_profiles import LLAMA_1B_INSTRUCT

# Model profile for Llama 1B
_PROFILE = LLAMA_1B_INSTRUCT
NUM_TRAINING_GPUS = _PROFILE.num_training_gpus      # 4
NUM_SAMPLING_GPUS = _PROFILE.num_sampling_gpus       # 4
MAX_SEQUENCE_LENGTH = _PROFILE.max_sequence_length   # 4096
CKPT_PATH = _PROFILE.ckpt_path

# Paths
RUN_CONFIG_PATH = "${HOME}/reward-training/run_configs/llama_1b_cooperbench.run"
K8S_SECRETS_PATH = "${HOME}/repos/secrets_template.toml"
WANDB_PROJECT = "multiagent-debate-rl"
PRIORITY_CLASS = "dev-low"
SEED = 42

# vLLM sidecar
_VLLM_SIDECAR = "vllm"
_VLLM_PORT = 8000
_VLLM_EXPORT_DIR = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"

# Override: minimal steps for testing
_TOTAL_TRAIN_STEPS = 2
_EXPORT_EVERY_STEPS = 1
_TRAIN_BATCH_SIZE = 4
_EVAL_BATCH_SIZE = 4
_GENERATIONS_PER_PROMPT = 2


class Llama1bCooperBench(sweep_base.Sweep):
    settings: sweep.SweepSettings = sweep.SweepSettings(
        sweep_output_path="${HOME}/sweep_jobs/llama1b_cooperbench/",
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
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/llama1b-cooperbench/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            train_batch_size=_TRAIN_BATCH_SIZE,
            eval_batch_size=_EVAL_BATCH_SIZE,
            total_train_steps=_TOTAL_TRAIN_STEPS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            validation_every_steps=_TOTAL_TRAIN_STEPS,
            n_gradient_accumulation_steps=1,
            lr_schedule={
                "kwargs": {
                    "peak_lr": 3e-6,
                    "end_lr": 3e-6,
                },
            },
            seed=SEED,
            objective={
                "loss": {
                    "kwargs": {
                        "rl_training_steps": 1,
                        "hard_update_ref_every_steps": 1,
                        "preference": {
                            "loss_variation": "gspo_use_ref_policy_as_pi_old",
                            "beta": 0.03,
                            "avg_loglikelihood": False,
                            "generations_per_prompt": _GENERATIONS_PER_PROMPT,
                        },
                    },
                },
            },
            n_last_ckpts_to_keep=1,
            checkpoint_every_steps=_TOTAL_TRAIN_STEPS,
            minizord=flink_zord.FlinkZordConfig(
                samplers=dict(
                    sampler_key=FlinkVllmSidecarSamplerConfig(
                        vllm_endpoint=EndpointResolverConfig(
                            job_name=_VLLM_SIDECAR,
                            job_port=_VLLM_PORT,
                        ),
                        export_every_steps=_EXPORT_EVERY_STEPS,
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
                        max_sequence_length_at_evaluation=MAX_SEQUENCE_LENGTH,
                    ),
                ),
                reward_models=dict(
                    reward_model_key=flink_reward_model.FlinkWrapRewardModelEndpointConfig(),
                ),
                # CooperBench: no env_name_remap, no react_config patching
                input_data_preprocessor=CombItemsPreprocessorConfig(
                    env_name_remap={},
                    patch_data_item={},
                    patch_scenario_config={},
                ),
                actor=FlinkCombActorConfig(
                    sampler_endpoint_key="sampler_key",
                ),
                num_actors_per_batch_item=_GENERATIONS_PER_PROMPT,
                actors_queue_batches=32,
                eval_actors_queue_batches=32,
                learner=FlinkRlooLearnerConfig(policy_gradient_loss="grpo"),
                # NO FilteringStreamerConfig — all outputs reach learner.
                # FilterOnIdenticalReward drops everything when the model
                # always returns 0.0 reward (which Llama 1B will on cooperbench).
                actor_outputs_streamers=[
                    DebateMetricStreamerConfig(
                        n_rollouts_per_prompt=_GENERATIONS_PER_PROMPT,
                    ),
                ],
                eval=FlinkEvalConfig(
                    n_generation_steps=-1,
                    actor=FlinkCombActorConfig(
                        sampler_endpoint_key="eval_sampler_key",
                        patch_number_of_generation_per_prompt=1,
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
    sweep.cli.run_sweep_with_flags(Llama1bCooperBench(), debugging_artefacts=[__file__])
