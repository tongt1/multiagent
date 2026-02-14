"""SWEEP config for COMA advantage reward shaping (SmolLM-135M, MATH-500).

COMA (Counterfactual Multi-Agent) advantage: A_i = r_i - baseline_i.
The counterfactual baseline marginalizes over agent i's actions while
holding other agents fixed. n_rollouts_per_prompt=4 is required in BOTH
DebateMetricStreamerConfig AND reward_shaping_params (Pitfall 3/4).

Usage:
    # Preview (no submission)
    uv run python configs/reward_shaping_sweep/sweep_coma_advantage.py start

    # Submit to cluster
    uv run python configs/reward_shaping_sweep/sweep_coma_advantage.py --submit start

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
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig

from configs.reward_shaping_sweep._base import (
    NUM_TRAINING_GPUS,
    NUM_SAMPLING_GPUS,
    MAX_SEQUENCE_LENGTH,
    CKPT_PATH,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    TOTAL_TRAIN_STEPS,
    LEARNING_RATE,
    KL_BETA,
    GENERATIONS_PER_PROMPT,
    EXPORT_EVERY_STEPS,
    HARD_UPDATE_REF_EVERY_STEPS,
    SEED,
    N_GRADIENT_ACCUMULATION_STEPS,
    PRIORITY_CLASS,
    RUN_CONFIG_PATH,
    K8S_SECRETS_PATH,
    WANDB_PROJECT,
    _VLLM_SIDECAR,
    _VLLM_PORT,
    _VLLM_EXPORT_DIR,
)


class RewardShapingSweep(sweep_base.Sweep):
    settings: sweep.SweepSettings = sweep.SweepSettings(
        sweep_output_path="${HOME}/sweep_jobs/reward_shaping_comparison/",
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
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/reward-shaping-sweep/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            total_train_steps=TOTAL_TRAIN_STEPS,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            validation_every_steps=TOTAL_TRAIN_STEPS,
            n_gradient_accumulation_steps=N_GRADIENT_ACCUMULATION_STEPS,
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
            checkpoint_every_steps=TOTAL_TRAIN_STEPS,
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
                    DebateMetricStreamerConfig(
                        n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
                        reward_shaping_strategy="coma_advantage",
                        reward_shaping_params={"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT},
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
                        "--gpus-per-vllm-worker=1",
                    ]
                ),
                ports=dict(web=_VLLM_PORT),
            )
        ],
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        return [{}]


if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(RewardShapingSweep(), debugging_artefacts=[__file__])
