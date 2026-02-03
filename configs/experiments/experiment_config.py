"""Experiment configuration dataclasses.

ExperimentConfig specifies only variable parameters (override-only pattern).
All other values come from Phase 2 SWEEP base configs.

Usage:
    # Create experiment with overrides
    exp = ExperimentConfig(
        name="debate_cmdR",
        mode="debate",
        model="command-a-03-2025",
        learning_rate=2e-6,
    )

    # Apply to SWEEP config
    sweep_config = load_sweep_config_dict()
    modified = exp.apply_to_sweep_config(sweep_config)

    # Validate
    errors = exp.validate()
    if errors:
        raise ValueError(f"Invalid config: {errors}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """Single experiment configuration (override-only pattern).

    Attributes:
        name: Experiment name (e.g., "debate_cmdR")
        mode: "debate" or "baseline"
        model: Model name (maps to ModelConfig via registry)
        learning_rate: Override base default (1.5e-6)
        train_batch_size: Override base default (64)
        total_train_steps: Override base default (500)
        kl_beta: Override base default (0.01)
        generations_per_prompt: Override base default (8)
        data_path: Override data path (None = use converted Phase 1 data)
        push_to_hub: Whether to push checkpoints to HuggingFace Hub
        hub_repo_id: HF Hub repo ID (required if push_to_hub=True)
        linked_experiment: Name of linked experiment for comparison
    """

    name: str
    mode: str
    model: str = "command-a-03-2025"
    learning_rate: float | None = None
    train_batch_size: int | None = None
    total_train_steps: int | None = None
    kl_beta: float | None = None
    generations_per_prompt: int | None = None
    data_path: str | None = None
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    linked_experiment: str | None = None

    def apply_to_sweep_config(self, sweep_config_dict: dict) -> dict:
        """Apply overrides to SWEEP config dict.

        Maps override fields to nested SWEEP config structure:
        - learning_rate -> lr_schedule.kwargs.peak_lr and end_lr
        - train_batch_size -> train_batch_size
        - total_train_steps -> total_train_steps
        - kl_beta -> objective.loss.kwargs.preference.beta
        - generations_per_prompt -> objective.loss.kwargs.preference.generations_per_prompt
                                    AND minizord.num_actors_per_batch_item (must sync)

        Args:
            sweep_config_dict: Base SWEEP config dict

        Returns:
            Modified config dict (mutates in place and returns)
        """
        # Learning rate
        if self.learning_rate is not None:
            if "lr_schedule" not in sweep_config_dict:
                sweep_config_dict["lr_schedule"] = {"kwargs": {}}
            if "kwargs" not in sweep_config_dict["lr_schedule"]:
                sweep_config_dict["lr_schedule"]["kwargs"] = {}
            sweep_config_dict["lr_schedule"]["kwargs"]["peak_lr"] = self.learning_rate
            sweep_config_dict["lr_schedule"]["kwargs"]["end_lr"] = self.learning_rate

        # Train batch size
        if self.train_batch_size is not None:
            sweep_config_dict["train_batch_size"] = self.train_batch_size

        # Total train steps
        if self.total_train_steps is not None:
            sweep_config_dict["total_train_steps"] = self.total_train_steps

        # KL beta
        if self.kl_beta is not None:
            if "objective" not in sweep_config_dict:
                sweep_config_dict["objective"] = {"loss": {"kwargs": {"preference": {}}}}
            if "loss" not in sweep_config_dict["objective"]:
                sweep_config_dict["objective"]["loss"] = {"kwargs": {"preference": {}}}
            if "kwargs" not in sweep_config_dict["objective"]["loss"]:
                sweep_config_dict["objective"]["loss"]["kwargs"] = {"preference": {}}
            if "preference" not in sweep_config_dict["objective"]["loss"]["kwargs"]:
                sweep_config_dict["objective"]["loss"]["kwargs"]["preference"] = {}
            sweep_config_dict["objective"]["loss"]["kwargs"]["preference"]["beta"] = self.kl_beta

        # Generations per prompt (MUST sync with num_actors_per_batch_item)
        if self.generations_per_prompt is not None:
            # Update preference loss config
            if "objective" not in sweep_config_dict:
                sweep_config_dict["objective"] = {"loss": {"kwargs": {"preference": {}}}}
            if "loss" not in sweep_config_dict["objective"]:
                sweep_config_dict["objective"]["loss"] = {"kwargs": {"preference": {}}}
            if "kwargs" not in sweep_config_dict["objective"]["loss"]:
                sweep_config_dict["objective"]["loss"]["kwargs"] = {"preference": {}}
            if "preference" not in sweep_config_dict["objective"]["loss"]["kwargs"]:
                sweep_config_dict["objective"]["loss"]["kwargs"]["preference"] = {}
            sweep_config_dict["objective"]["loss"]["kwargs"]["preference"]["generations_per_prompt"] = self.generations_per_prompt

            # Update minizord num_actors_per_batch_item (CRITICAL: must stay in sync)
            if "minizord" not in sweep_config_dict:
                sweep_config_dict["minizord"] = {}
            sweep_config_dict["minizord"]["num_actors_per_batch_item"] = self.generations_per_prompt

        return sweep_config_dict

    def validate(self) -> list[str]:
        """Validate experiment configuration.

        Returns:
            List of error strings (empty = valid)
        """
        errors = []

        # Mode must be debate or baseline
        if self.mode not in ["debate", "baseline"]:
            errors.append(f"mode must be 'debate' or 'baseline', got '{self.mode}'")

        # If push_to_hub, hub_repo_id must be set
        if self.push_to_hub and not self.hub_repo_id:
            errors.append("hub_repo_id must be set when push_to_hub is True")

        # If linked_experiment is set, it must be non-empty
        if self.linked_experiment is not None and not self.linked_experiment.strip():
            errors.append("linked_experiment must be a non-empty string if set")

        return errors


@dataclass
class ExperimentBatchConfig:
    """Batch of experiments configuration.

    Attributes:
        experiments: List of experiment configs
        batch_name: Optional batch name
        base_ckpt_path: Base model checkpoint S3 path (shared across experiments)
        gcs_bucket: GCS bucket for data upload
        max_retries: Retry count per stage (research recommendation)
        parallel: Whether independent experiments can run concurrently
    """

    experiments: list[ExperimentConfig]
    base_ckpt_path: str
    batch_name: str | None = None
    gcs_bucket: str = "your-bucket"
    max_retries: int = 3
    parallel: bool = True

    def validate(self) -> list[str]:
        """Validate batch configuration.

        Returns:
            List of error strings (empty = valid)
        """
        errors = []

        # Validate each experiment
        for i, exp in enumerate(self.experiments):
            exp_errors = exp.validate()
            for err in exp_errors:
                errors.append(f"Experiment {i} ({exp.name}): {err}")

        # Check for duplicate names
        names = [exp.name for exp in self.experiments]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            errors.append(f"Duplicate experiment names: {set(duplicates)}")

        # Check linked_experiment references valid names
        valid_names = set(names)
        for exp in self.experiments:
            if exp.linked_experiment and exp.linked_experiment not in valid_names:
                errors.append(
                    f"Experiment '{exp.name}' links to non-existent experiment '{exp.linked_experiment}'"
                )

        return errors

    def get_linked_pairs(self) -> list[tuple[ExperimentConfig, ExperimentConfig]]:
        """Return pairs of linked experiments for comparison stage.

        Returns:
            List of (experiment, linked_experiment) tuples
        """
        pairs = []
        exp_by_name = {exp.name: exp for exp in self.experiments}

        for exp in self.experiments:
            if exp.linked_experiment:
                linked = exp_by_name.get(exp.linked_experiment)
                if linked:
                    pairs.append((exp, linked))

        return pairs
