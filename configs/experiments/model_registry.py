"""Model registry mapping model names to training infrastructure paths.

Maps model names to their training infrastructure configuration:
- Cohere models use Hive estimator (API-based)
- Open-source models use vLLM sidecar (self-hosted)

Usage:
    from configs.experiments.model_registry import get_model_config

    # Get model config
    model_cfg = get_model_config("command-a-03-2025")
    print(f"Estimator: {model_cfg.estimator_type}")  # "hive"
    print(f"GPUs: {model_cfg.num_training_gpus}")    # 64

    # List all supported models
    from configs.experiments.model_registry import SUPPORTED_MODELS
    print(list(SUPPORTED_MODELS.keys()))
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration for training infrastructure routing.

    Attributes:
        name: Canonical model name
        family: Model family ("command_r" or "llama")
        estimator_type: Training path ("hive" for Cohere API, "vllm" for open-source)
        hive_model_id: Model ID for Hive estimator (Cohere models only)
        hf_model_id: HuggingFace model ID (open-source models only)
        default_max_sequence_length: Model-specific context length
        num_training_gpus: GPU count for training
        num_sampling_gpus: GPU count for sampling
    """

    name: str
    family: str
    estimator_type: str
    hive_model_id: str | None = None
    hf_model_id: str | None = None
    default_max_sequence_length: int = 8192
    num_training_gpus: int = 64
    num_sampling_gpus: int = 64


# Supported models registry
SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # Cohere Command R family (Hive estimator)
    "command-r-03-2025": ModelConfig(
        name="command-r-03-2025",
        family="command_r",
        estimator_type="hive",
        hive_model_id="command-r-03-2025-pld-rl",
        default_max_sequence_length=8192,
        num_training_gpus=64,
        num_sampling_gpus=64,
    ),
    "command-r-plus-08-2024": ModelConfig(
        name="command-r-plus-08-2024",
        family="command_r",
        estimator_type="hive",
        hive_model_id="command-r-plus-08-2024-pld-rl",
        default_max_sequence_length=8192,
        num_training_gpus=128,
        num_sampling_gpus=128,
    ),
    "command-a-03-2025": ModelConfig(
        name="command-a-03-2025",
        family="command_r",
        estimator_type="hive",
        hive_model_id="command-a-03-2025-pld-rl",
        default_max_sequence_length=8192,
        num_training_gpus=64,
        num_sampling_gpus=64,
    ),
    # Meta Llama family (vLLM sidecar)
    "meta-llama/Llama-3-8b": ModelConfig(
        name="meta-llama/Llama-3-8b",
        family="llama",
        estimator_type="vllm",
        hf_model_id="meta-llama/Llama-3-8b",
        default_max_sequence_length=8192,
        num_training_gpus=8,
        num_sampling_gpus=8,
    ),
    "meta-llama/Llama-3-70b": ModelConfig(
        name="meta-llama/Llama-3-70b",
        family="llama",
        estimator_type="vllm",
        hf_model_id="meta-llama/Llama-3-70b",
        default_max_sequence_length=8192,
        num_training_gpus=64,
        num_sampling_gpus=64,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name.

    Args:
        model_name: Model name (must be in SUPPORTED_MODELS)

    Returns:
        ModelConfig for the specified model

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        )
    return SUPPORTED_MODELS[model_name]
