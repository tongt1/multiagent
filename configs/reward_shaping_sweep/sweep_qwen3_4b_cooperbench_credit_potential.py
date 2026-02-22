"""Qwen3-4B CooperBench: Potential-based reward shaping.

r' = r + gamma*Phi(s')-Phi(s) — preserves optimal policy (Ng et al. 1999).
Uses debate_length potential to penalize long interactions.
"""
from __future__ import annotations
from collections.abc import Iterable
import sweep
from post_training.canonicals import sweep_base
from post_training.flink import flink_zord
from post_training.flink.components import flink_reward_model
from post_training.flink.components.flink_comb_actor import FlinkCombActorConfig
from post_training.flink.components.flink_dual_rloo_learner import FlinkDualRlooLearnerConfig
from post_training.flink.components.flink_eval import FlinkEvalConfig
from post_training.flink.components.flink_input_data_preprocessors import CombItemsPreprocessorConfig
from post_training.flink.components.flink_sampler_vllm_sidecar import FlinkVllmSidecarSamplerConfig
from post_training.flink.utils.endpoint_resolver import EndpointResolverConfig
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig
from configs.model_profiles import QWEN3_4B_INSTRUCT_12GPU

_P = QWEN3_4B_INSTRUCT_12GPU; _V = "vllm"; _VP = 8000
_VD = "/data/1d/post-training/${USER}/${SWEEP_NAME}/${TRIAL_IDX}"
_S = 500; _E = 5; _B = 16; _G = 1; _T = 20; _R = 2.0

class Qwen3_4bCreditPotential(sweep_base.Sweep):
    settings: sweep.SweepSettings = sweep.SweepSettings(sweep_output_path="${HOME}/sweep_jobs/qwen3_4b_cooperbench/", cluster=sweep.Cluster.cw_us_east_04_prod)
    fax: sweep.FaxConfig = sweep_base.PostTraining(
        partition=f"gpu_{_P.num_training_gpus}", queue=sweep.Queue.post_training_cohere_labs_queue, jobs_max_fanout=1,
        wandb_project="multiagent-debate-rl", priority_class="dev-low",
        run_config="${HOME}/reward-training/run_configs/qwen3_4b_cooperbench.run",
        k8s_env_secrets_toml="${HOME}/repos/secrets_template.toml", ckpt_path=_P.ckpt_path,
        output_dir="s3://us-east-01a/30d/post-training/${USER}/multiagent-debate-rl/qwen3-4b-cooperbench-6gpu/${SWEEP_NAME}/${TRIAL_IDX}",
        patch_run_config=dict(
            train_batch_size=_B, eval_batch_size=4, total_train_steps=_S, max_sequence_length=_P.max_sequence_length,
            sharding={"n_tensor_parallel": 4}, validation_every_steps=100, n_gradient_accumulation_steps=2,
            lr_schedule={"kwargs": {"warmup_steps": 20, "total_steps": _S, "peak_lr": 1e-6, "end_lr": 1e-7}}, seed=42,
            objective={"loss": {"kwargs": {"rl_training_steps": 1, "hard_update_ref_every_steps": _E, "preference": {
                "loss_variation": "dapo", "use_reference_policy": False, "beta": 0.0, "avg_loglikelihood": False,
                "generations_per_prompt": _G, "std_normalised": False, "ppo_epsilon_low": 0.8, "ppo_epsilon_high": 0.28,
                "dapo_overlong_coeff": 0.5, "dapo_overlong_threshold": 0.9}}}},
            n_last_ckpts_to_keep=2, checkpoint_every_steps=100,
            minizord=flink_zord.FlinkZordConfig(
                samplers=dict(
                    sampler_key=FlinkVllmSidecarSamplerConfig(vllm_endpoint=EndpointResolverConfig(job_name=_V, job_port=_VP),
                        export_every_steps=_E, export_dir=_VD, temperature=0.8, start_temperature=1.0, end_temperature=0.6,
                        total_train_steps=_S, thinking_token_budget=4096),
                    eval_sampler_key=FlinkVllmSidecarSamplerConfig(vllm_endpoint=EndpointResolverConfig(job_name=_V, job_port=_VP),
                        export_every_steps=None, export_dir=_VD, temperature=0.6, p=0.95, max_sequence_length_at_evaluation=_P.max_sequence_length)),
                reward_models=dict(reward_model_key=flink_reward_model.FlinkWrapRewardModelEndpointConfig()),
                input_data_preprocessor=CombItemsPreprocessorConfig(env_name_remap={}, patch_data_item={},
                    patch_scenario_config={"cooperbench": {"docker_timeout": 60}}),
                actor=FlinkCombActorConfig(sampler_endpoint_key="sampler_key", agentic_mode=True, agentic_max_iterations=_T,
                    agentic_rollout_timeout=900, agentic_vllm_base_url=f"http://{_V}:{_VP}/v1",
                    max_concurrent_trajectories=0, agentic_redundancy_factor=_R, agentic_temperature=1.0, agentic_comms_mode=True),
                num_actors_per_batch_item=4, actors_queue_batches=32, eval_actors_queue_batches=32,
                learner=FlinkDualRlooLearnerConfig(policy_gradient_loss="grpo", solver_ckpt=_P.ckpt_path,
                    verifier_ckpt=_P.ckpt_path, freeze_roles=["verifier", "judge"]),
                actor_outputs_streamers=[DebateMetricStreamerConfig(n_rollouts_per_prompt=_G,
                    reward_shaping_strategy="potential_based",
                    reward_shaping_params={"gamma": 0.99, "potential_type": "debate_length", "penalty": 0.1})],
                eval=FlinkEvalConfig(n_generation_steps=-1, actor=FlinkCombActorConfig(
                    sampler_endpoint_key="eval_sampler_key", patch_number_of_generation_per_prompt=1,
                    agentic_mode=True, agentic_max_iterations=_T, agentic_rollout_timeout=900,
                    agentic_vllm_base_url=f"http://{_V}:{_VP}/v1", max_concurrent_trajectories=0,
                    agentic_redundancy_factor=1.0, agentic_temperature=0.8, agentic_comms_mode=True)),
                log_train_generations_every_steps=1, save_debug_data_every_steps=1).model_dump(),
            likelihood_evals=None, finetuning=dict(lora=dict(enabled=False, rank=8, alpha=8.0)),
            ckpt=dict(force_at_init=False), validation=dict(force_at_init=False),
            hf_export=dict(override_mesh_for_local_gathering=_P.needs_mesh_override),
            model_export=dict(enabled=False), profile_every_steps=1, first_step_to_profile=1, read_extra_state=False),
        retries=0, kjobs_compute="",
        patch_kjobs_compute=dict(experimental_needs_docker_in_docker=True, env={
            "JAX_COMPILATION_CACHE_DIR": "/data/1d/jax-cache/${USER}", "JAX_LOG_COMPILES": "1",
            "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"}),
        sidecars=[sweep.SidecarConfig(name=_V,
            repo=sweep.RepoConfig(directory=sweep_base.get_current_repo_directory() + "/vinfer", kind=sweep.RepoKinds.vllm),
            partition=f"gpu_{_P.num_sampling_gpus}",
            command=f"env FAX_NUMBER_GPUS_PER_WORKER=2 VLLM_ATTENTION_BACKEND=FLASHINFER python -m vinfer.main --exports-glob-pattern={_VD}/*/_HF_EXPORT_IS_COMPLETE --port {_VP} --gpus-per-vllm-worker=1 --max-model-len {_P.max_sequence_length} --enforce-eager",
            ports=dict(web=_VP))])

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        return [{}]

if __name__ == "__main__":
    sweep.cli.run_sweep_with_flags(Qwen3_4bCreditPotential(), debugging_artefacts=[__file__])
