"""Reproducibility verification utilities for experiment tracking."""

import json
import subprocess
from pathlib import Path
from typing import Any, Optional


def verify_training_seed(experiment_dir: Path, expected_seed: int = 42) -> dict[str, Any]:
    """Verify training seed matches configured value across metadata and W&B.

    Args:
        experiment_dir: Path to experiment directory containing metadata.json
        expected_seed: Expected seed value (default: 42)

    Returns:
        Dict with keys:
        - seed_match: bool - Whether all seeds match expected value
        - configured_seed: int - The expected seed value
        - metadata_seed: int|None - Seed from metadata.json
        - wandb_seed: int|None - Seed from W&B config (if available)
        - warnings: list[str] - Any warnings encountered
    """
    warnings = []
    metadata_seed = None
    wandb_seed = None

    # Read metadata.json
    metadata_path = experiment_dir / "metadata.json"
    if not metadata_path.exists():
        warnings.append(f"metadata.json not found in {experiment_dir}")
        return {
            "seed_match": False,
            "configured_seed": expected_seed,
            "metadata_seed": None,
            "wandb_seed": None,
            "warnings": warnings,
        }

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Extract seed from metadata
    metadata_seed = metadata.get("seed")
    if metadata_seed is None:
        warnings.append("No 'seed' field found in metadata.json")

    # Attempt to retrieve W&B config if run ID exists
    wandb_run_id = metadata.get("wandb_run_id")
    if wandb_run_id:
        try:
            import wandb

            # Try to fetch the run
            api = wandb.Api()
            # Note: This requires WANDB_ENTITY and WANDB_PROJECT to be set or in metadata
            wandb_entity = metadata.get("wandb_entity", "")
            wandb_project = metadata.get("wandb_project", "")

            if wandb_entity and wandb_project:
                run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
                try:
                    run = api.run(run_path)
                    wandb_seed = run.config.get("seed")
                except Exception as e:
                    warnings.append(f"Could not fetch W&B run {run_path}: {str(e)}")
            else:
                warnings.append("W&B entity or project not found in metadata - skipping W&B verification")

        except ImportError:
            warnings.append("wandb not installed - skipping W&B seed verification")
        except Exception as e:
            warnings.append(f"Error accessing W&B: {str(e)}")

    # Determine if seeds match
    seed_match = True
    if metadata_seed is not None and metadata_seed != expected_seed:
        seed_match = False
    if wandb_seed is not None and wandb_seed != expected_seed:
        seed_match = False

    # If we couldn't verify some sources, still return True with warnings
    if metadata_seed is None and wandb_seed is None:
        seed_match = False

    return {
        "seed_match": seed_match,
        "configured_seed": expected_seed,
        "metadata_seed": metadata_seed,
        "wandb_seed": wandb_seed,
        "warnings": warnings,
    }


def save_config_snapshot(
    experiment_dir: Path,
    config_files: list[Path],
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save snapshot of configuration files for reproducibility.

    Args:
        experiment_dir: Path to experiment directory
        config_files: List of config file paths to snapshot
        extra_metadata: Optional additional metadata to save

    Returns:
        Path to snapshot directory
    """
    snapshot_dir = experiment_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy config files
    for config_file in config_files:
        if config_file.exists():
            dest = snapshot_dir / config_file.name
            with open(config_file, "r") as src_f, open(dest, "w") as dst_f:
                dst_f.write(src_f.read())
        else:
            # Log warning but don't fail
            print(f"Warning: Config file {config_file} not found, skipping")

    # Get git commit hash
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        # Git not available or not in a repo
        pass

    # Save snapshot metadata
    snapshot_metadata = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "git_commit": git_commit,
        "config_files": [str(f) for f in config_files],
    }

    if extra_metadata:
        snapshot_metadata["extra"] = extra_metadata

    metadata_path = snapshot_dir / "snapshot_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(snapshot_metadata, f, indent=2)

    return snapshot_dir


def validate_experiment_metadata(experiment_dir: Path) -> dict[str, Any]:
    """Validate experiment metadata.json for required fields and data integrity.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict with keys:
        - valid: bool - Whether metadata is valid
        - errors: list[str] - Validation errors
        - warnings: list[str] - Non-critical warnings
    """
    errors = []
    warnings = []

    # Check experiment_dir exists
    if not experiment_dir.exists():
        return {
            "valid": False,
            "errors": [f"Experiment directory does not exist: {experiment_dir}"],
            "warnings": [],
        }

    # Check metadata.json exists
    metadata_path = experiment_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "valid": False,
            "errors": [f"metadata.json not found in {experiment_dir}"],
            "warnings": [],
        }

    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON in metadata.json: {str(e)}"],
            "warnings": [],
        }

    # Validate required fields
    required_fields = ["experiment_id", "timestamp", "mode", "data_paths"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")

    # Check data file existence (if local paths)
    if "data_paths" in metadata:
        data_paths = metadata["data_paths"]
        if isinstance(data_paths, dict):
            for key, path_str in data_paths.items():
                # Check if it's a local path (not URL or cloud path)
                if isinstance(path_str, str) and not any(
                    path_str.startswith(prefix) for prefix in ["http://", "https://", "gs://", "s3://"]
                ):
                    data_path = Path(path_str)
                    if not data_path.exists():
                        warnings.append(f"Data file not found: {key} -> {path_str}")

    # Validate experiment_id format (should not be empty)
    if "experiment_id" in metadata:
        exp_id = metadata["experiment_id"]
        if not exp_id or not isinstance(exp_id, str):
            errors.append("experiment_id must be a non-empty string")

    # Validate timestamp format
    if "timestamp" in metadata:
        timestamp = metadata["timestamp"]
        try:
            # Try to parse as ISO format
            from datetime import datetime
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            warnings.append(f"timestamp is not in ISO format: {timestamp}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
