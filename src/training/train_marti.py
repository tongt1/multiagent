"""MARTI training entrypoint for multi-agent RL training jobs.

Implements REINFORCE/GRPO-style training:
1. Loads MARTI trajectory JSONL (pre-collected prompts, completions, rewards)
2. Trains model with reward-weighted maximum likelihood
3. Logs per-episode and per-agent metrics to Weights & Biases
4. Saves checkpoints in HuggingFace format

Supports both environment variable config (for kjobs/Ray) and CLI arguments (for local testing).

Usage:
    # Via environment variable (kjobs/Ray):
    TRAINING_CONFIG_JSON='...' python -m src.training.train_marti

    # Via CLI arguments (local testing):
    python -m src.training.train_marti \
        --pretrain-path HuggingFaceTB/SmolLM-135M \
        --trajectory-path data/sample_trajectories.jsonl \
        --save-path checkpoints/test_run \
        --wandb-project marti-training \
        --num-episodes 2 \
        --batch-size 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.training.training_config import (
    MultiAgentTrainingConfig,
    OpenRLHFConfig,
    RayJobConfig,
    TrainingAlgorithm,
    VLLMConfig,
)


def load_config_from_env() -> MultiAgentTrainingConfig:
    """Load training configuration from environment variable.

    Returns:
        Parsed MultiAgentTrainingConfig

    Raises:
        ValueError: If TRAINING_CONFIG_JSON not set or invalid
    """
    config_json = os.getenv("TRAINING_CONFIG_JSON")
    if not config_json:
        raise ValueError("TRAINING_CONFIG_JSON environment variable not set")

    logger.info("Loading training configuration from environment")

    try:
        config = MultiAgentTrainingConfig.model_validate_json(config_json)
        logger.info("Successfully parsed training configuration")
        return config
    except Exception as e:
        logger.error(f"Failed to parse training config: {e}")
        raise ValueError(f"Invalid training config JSON: {e}")


def load_config_from_args(args: argparse.Namespace) -> MultiAgentTrainingConfig:
    """Build training configuration from CLI arguments."""
    algorithm_map = {
        "reinforce": TrainingAlgorithm.REINFORCE,
        "grpo": TrainingAlgorithm.GRPO,
        "ppo": TrainingAlgorithm.PPO,
        "rloo": TrainingAlgorithm.RLOO,
    }

    return MultiAgentTrainingConfig(
        pretrain_path=args.pretrain_path,
        trajectory_path=args.trajectory_path,
        save_path=args.save_path,
        ray_config=RayJobConfig(num_nodes=1, num_gpus_per_node=1),
        vllm_config=VLLMConfig(num_engines=1, tensor_parallel_size=1),
        openrlhf_config=OpenRLHFConfig(
            algorithm=algorithm_map.get(args.algorithm, TrainingAlgorithm.REINFORCE),
            num_episodes=args.num_episodes,
            train_batch_size=args.batch_size,
            micro_train_batch_size=args.batch_size,
            rollout_batch_size=args.batch_size,
            micro_rollout_batch_size=args.batch_size,
            actor_learning_rate=args.learning_rate,
            normalize_reward=not args.no_reward_normalization,
        ),
        num_agents=5,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def validate_setup(config: MultiAgentTrainingConfig) -> None:
    """Validate training setup before starting."""
    logger.info("Validating training setup")

    # Check trajectory file exists
    trajectory_path = Path(config.trajectory_path)
    if not trajectory_path.exists():
        raise ValueError(f"Trajectory file not found: {config.trajectory_path}")

    logger.info(f"Trajectory file found: {config.trajectory_path}")

    # Check pretrain path (skip if remote/hub model name)
    if not config.pretrain_path.startswith(("s3://", "gs://", "azure://")) and "/" in config.pretrain_path:
        pretrain_path = Path(config.pretrain_path)
        if pretrain_path.exists():
            logger.info(f"Local pretrain path found: {config.pretrain_path}")
        else:
            logger.info(f"Pretrain path not local, assuming HuggingFace Hub model: {config.pretrain_path}")
    else:
        logger.info(f"Using model: {config.pretrain_path}")

    # Create save path
    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Setup validation complete")


def log_configuration(config: MultiAgentTrainingConfig) -> None:
    """Log training configuration for debugging."""
    logger.info("=" * 80)
    logger.info("MARTI Training Configuration")
    logger.info("=" * 80)
    logger.info(f"  Pretrain path: {config.pretrain_path}")
    logger.info(f"  Trajectory path: {config.trajectory_path}")
    logger.info(f"  Save path: {config.save_path}")
    logger.info(f"  Algorithm: {config.openrlhf_config.algorithm.value}")
    logger.info(f"  Episodes: {config.openrlhf_config.num_episodes}")
    logger.info(f"  Batch size: {config.openrlhf_config.micro_train_batch_size}")
    logger.info(f"  Learning rate: {config.openrlhf_config.actor_learning_rate}")
    logger.info(f"  Reward shaping: {config.reward_shaping.mode.value} (alpha={config.reward_shaping.alpha})")
    logger.info(f"  Normalize rewards: {config.openrlhf_config.normalize_reward}")
    if config.wandb_project:
        logger.info(f"  W&B project: {config.wandb_project}")
        if config.wandb_run_name:
            logger.info(f"  W&B run: {config.wandb_run_name}")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectories(trajectory_path: str) -> list[dict[str, Any]]:
    """Load trajectory entries from JSONL file.

    Each line is a JSON object with: input, output, reward, agent, metadata.
    """
    trajectories = []
    with open(trajectory_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                trajectories.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(trajectories)} trajectory entries from {trajectory_path}")
    return trajectories


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for MARTI trajectory entries."""

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.examples: list[dict[str, Any]] = []

        for entry in trajectories:
            input_data = entry.get("input", {})
            output_data = entry.get("output", {})
            reward = entry.get("reward", 0.0) or 0.0
            agent = entry.get("agent", "unknown")

            # Extract prompt and completion text
            prompt = input_data.get("problem", input_data.get("prompt", ""))
            completion = output_data.get("solution", output_data.get(
                "response", output_data.get("output", "")
            ))

            if isinstance(completion, dict):
                completion = json.dumps(completion)

            if not prompt or not completion:
                continue

            # Tokenize separately
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)

            # Truncate to max_length
            total = len(prompt_ids) + len(completion_ids)
            if total > max_length:
                max_prompt = max_length // 2
                if len(prompt_ids) > max_prompt:
                    prompt_ids = prompt_ids[:max_prompt]
                remaining = max_length - len(prompt_ids)
                completion_ids = completion_ids[:remaining]

            input_ids = prompt_ids + completion_ids
            # Labels: -100 for prompt tokens (don't train on prompt)
            labels = [-100] * len(prompt_ids) + completion_ids
            # Completion mask: 1.0 for completion tokens only
            completion_mask = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
            attention_mask = [1] * len(input_ids)

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "completion_mask": completion_mask,
                "reward": float(reward),
                "agent": agent,
            })

        logger.info(f"Prepared {len(self.examples)} training examples from trajectories")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict[str, Any]:
    """Pad and collate a batch of examples."""
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []
    completion_mask = []
    rewards = []
    agents = []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        completion_mask.append(ex["completion_mask"] + [0.0] * pad_len)
        rewards.append(ex["reward"])
        agents.append(ex["agent"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.float),
        "rewards": torch.tensor(rewards, dtype=torch.float),
        "agents": agents,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_per_example_loss(
    model: Any,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, Any]:
    """Compute per-example negative log-likelihood on completion tokens.

    Returns:
        (per_example_loss, model_outputs)
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    # Shift for next-token prediction
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    mask = batch["completion_mask"][:, 1:].contiguous()

    batch_size, seq_len, vocab_size = logits.shape

    # Per-token cross entropy
    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, seq_len)

    # Average over completion tokens per example
    num_tokens = mask.sum(dim=1).clamp(min=1)
    per_example_loss = (per_token_loss * mask).sum(dim=1) / num_tokens

    return per_example_loss, outputs


def run_training(config: MultiAgentTrainingConfig) -> dict[str, Any]:
    """Run the REINFORCE/GRPO training loop.

    Algorithm:
        REINFORCE: loss = mean(reward_i * NLL_i)
        GRPO: loss = mean((reward_i - mean_reward) * NLL_i)

    Higher reward trajectories receive more gradient signal.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Optional W&B
    wandb = None
    if config.wandb_project:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # Initialize W&B
    if wandb and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "algorithm": config.openrlhf_config.algorithm.value,
                "num_episodes": config.openrlhf_config.num_episodes,
                "batch_size": config.openrlhf_config.micro_train_batch_size,
                "learning_rate": config.openrlhf_config.actor_learning_rate,
                "reward_shaping": config.reward_shaping.mode.value,
                "reward_alpha": config.reward_shaping.alpha,
                "pretrain_path": config.pretrain_path,
                "num_agents": config.num_agents,
                "num_rounds": config.num_rounds,
                "normalize_reward": config.openrlhf_config.normalize_reward,
                "init_kl_coef": config.openrlhf_config.init_kl_coef,
            },
        )

    # Load model and tokenizer
    logger.info(f"Loading model from: {config.pretrain_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrain_path,
        torch_dtype=torch.float32,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params:,} parameters")

    # Load training data
    trajectories = load_trajectories(config.trajectory_path)
    dataset = TrajectoryDataset(
        trajectories,
        tokenizer,
        max_length=min(config.vllm_config.max_model_len, 512),
    )

    if len(dataset) == 0:
        raise ValueError("No valid training examples found in trajectories")

    pad_id = tokenizer.pad_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=config.openrlhf_config.micro_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.openrlhf_config.actor_learning_rate,
        weight_decay=0.01,
    )

    # Reference model for KL divergence (frozen)
    ref_model = None
    if config.openrlhf_config.init_kl_coef > 0:
        logger.info("Loading reference model for KL penalty")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.pretrain_path,
            torch_dtype=torch.float32,
        ).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Metrics tracking
    metrics: dict[str, Any] = {
        "total_steps": 0,
        "episode_losses": [],
        "episode_rewards": [],
    }

    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    num_episodes = config.openrlhf_config.num_episodes
    logger.info(
        f"Starting training: {num_episodes} episodes, "
        f"{len(dataset)} examples, "
        f"batch_size={config.openrlhf_config.micro_train_batch_size}"
    )

    # ---- Training loop ----
    global_step = 0

    for episode in range(num_episodes):
        model.train()
        episode_loss = 0.0
        episode_reward = 0.0
        episode_kl = 0.0
        episode_steps = 0
        agent_rewards: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(dataloader):
            # Move tensors to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            per_example_loss, outputs = compute_per_example_loss(model, batch_device)
            rewards = batch_device["rewards"]

            # Reward normalization
            if config.openrlhf_config.normalize_reward and rewards.numel() > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Compute RL loss
            algo = config.openrlhf_config.algorithm
            if algo == TrainingAlgorithm.GRPO:
                advantages = rewards - rewards.mean()
                loss = (per_example_loss * advantages).mean()
            else:
                # REINFORCE / default: reward-weighted NLL
                loss = (per_example_loss * rewards).mean()

            # KL penalty
            kl_div = torch.tensor(0.0, device=device)
            if ref_model is not None and config.openrlhf_config.init_kl_coef > 0:
                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                    )
                policy_lp = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
                ref_lp = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1)
                mask = batch_device["completion_mask"][:, 1:]
                kl_per_token = (policy_lp.exp() * (policy_lp - ref_lp)).sum(dim=-1)
                kl_div = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)
                loss = loss + config.openrlhf_config.init_kl_coef * kl_div

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            raw_reward = batch["rewards"].mean().item()
            episode_loss += loss.item()
            episode_reward += raw_reward
            episode_kl += kl_div.item()
            episode_steps += 1
            global_step += 1

            # Per-agent rewards
            for agent, r in zip(batch["agents"], batch["rewards"].tolist()):
                agent_rewards.setdefault(agent, []).append(r)

            # Step-level W&B logging
            if wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/reward_mean": raw_reward,
                    "train/kl_divergence": kl_div.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/learning_rate": config.openrlhf_config.actor_learning_rate,
                    "train/episode": episode,
                }, step=global_step)

            if batch_idx % max(1, len(dataloader) // 5) == 0:
                logger.info(
                    f"  Episode {episode + 1}/{num_episodes} "
                    f"batch {batch_idx}/{len(dataloader)}: "
                    f"loss={loss.item():.4f} reward={raw_reward:.4f} "
                    f"kl={kl_div.item():.4f}"
                )

        # ---- Episode summary ----
        avg_loss = episode_loss / max(episode_steps, 1)
        avg_reward = episode_reward / max(episode_steps, 1)
        avg_kl = episode_kl / max(episode_steps, 1)

        metrics["episode_losses"].append(avg_loss)
        metrics["episode_rewards"].append(avg_reward)
        metrics["total_steps"] += episode_steps

        # Episode-level W&B logging
        episode_metrics: dict[str, float] = {
            "episode/loss": avg_loss,
            "episode/reward_mean": avg_reward,
            "episode/kl_divergence": avg_kl,
            "episode/num": episode + 1,
        }

        for agent, r_list in agent_rewards.items():
            episode_metrics[f"agent/{agent}/reward_mean"] = sum(r_list) / len(r_list)
            episode_metrics[f"agent/{agent}/count"] = float(len(r_list))

        if wandb:
            wandb.log(episode_metrics, step=global_step)

        logger.info(
            f"Episode {episode + 1}/{num_episodes} complete: "
            f"loss={avg_loss:.4f} reward={avg_reward:.4f} kl={avg_kl:.4f}"
        )

        # Save checkpoint
        ckpt_dir = save_path / f"episode_{episode + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    # ---- Final save ----
    final_dir = save_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final model saved: {final_dir}")

    if wandb:
        wandb.finish()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local training."""
    parser = argparse.ArgumentParser(
        description="MARTI training — reward-weighted RL for multi-agent trajectories",
    )
    parser.add_argument("--pretrain-path", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--trajectory-path", required=True, help="Path to trajectory JSONL")
    parser.add_argument("--save-path", required=True, help="Output checkpoint directory")
    parser.add_argument("--algorithm", default="reinforce", choices=["reinforce", "grpo", "ppo", "rloo"])
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    parser.add_argument("--no-reward-normalization", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Main entrypoint for MARTI training."""
    logger.info("Starting MARTI training")

    try:
        # Try environment variable first (kjobs/Ray), fall back to CLI args
        config_json = os.getenv("TRAINING_CONFIG_JSON")
        if config_json:
            logger.info("Loading config from TRAINING_CONFIG_JSON environment variable")
            config = load_config_from_env()
        else:
            logger.info("No TRAINING_CONFIG_JSON found, using CLI arguments")
            args = parse_args()
            config = load_config_from_args(args)

        validate_setup(config)
        log_configuration(config)

        result = run_training(config)

        logger.info("Training complete!")
        logger.info(f"  Total steps: {result['total_steps']}")
        logger.info(f"  Final loss: {result['episode_losses'][-1]:.4f}")
        logger.info(f"  Final avg reward: {result['episode_rewards'][-1]:.4f}")

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
