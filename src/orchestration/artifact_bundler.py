"""Artifact bundling and HuggingFace Hub upload for experiments.

Provides manifest generation, tar.gz bundling, and optional HF Hub upload.

Usage:
    from pathlib import Path
    from src.orchestration.artifact_bundler import create_experiment_bundle, upload_to_hub

    # Bundle experiment artifacts
    bundle_path = create_experiment_bundle(
        experiment_dir=Path("experiments/exp_001"),
        include_checkpoints=True
    )

    # Upload checkpoint to HuggingFace Hub (optional)
    hub_url = upload_to_hub(
        checkpoint_dir=Path("experiments/exp_001/checkpoints/ckpt-500"),
        repo_id="username/model-name",
        private=True
    )
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


def compute_data_hash(data_path: Path) -> str:
    """Compute SHA-256 hash of a data file for provenance.

    Args:
        data_path: Path to data file

    Returns:
        Hex digest string (64 chars) or "missing" if file doesn't exist
    """
    if not data_path.exists():
        return "missing"

    sha256 = hashlib.sha256()
    with open(data_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def create_experiment_manifest(
    experiment_dir: Path, config: dict | None = None
) -> dict:
    """Build manifest dict with full experiment provenance.

    Args:
        experiment_dir: Path to experiment directory
        config: Config dict (if None, load from experiment_dir/config.json)

    Returns:
        Manifest dict with experiment_id, created_at, config, data_provenance,
        eval_results, comparison, training
    """
    manifest: dict[str, Any] = {}

    # Basic metadata
    manifest["experiment_id"] = experiment_dir.name
    manifest["created_at"] = datetime.utcnow().isoformat() + "Z"

    # Config
    if config is not None:
        manifest["config"] = config
    else:
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                manifest["config"] = json.load(f)
        else:
            manifest["config"] = None

    # Data provenance
    data_provenance: dict[str, Any] = {
        "source": "MATH 500",
        "seed": 42,
        "files": {}
    }

    # Scan for .jsonl files in experiment outputs
    state_path = experiment_dir / "state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
            outputs = state.get("outputs", {})

            # Hash all .jsonl files referenced in stage outputs
            for stage, stage_outputs in outputs.items():
                if isinstance(stage_outputs, dict):
                    for key, value in stage_outputs.items():
                        if isinstance(value, str) and value.endswith(".jsonl"):
                            file_path = Path(value)
                            if not file_path.is_absolute():
                                file_path = experiment_dir / file_path

                            file_hash = compute_data_hash(file_path)
                            data_provenance["files"][value] = {
                                "path": value,
                                "hash": file_hash
                            }

    # Also scan for any .jsonl files directly in experiment_dir
    for jsonl_file in experiment_dir.glob("**/*.jsonl"):
        if jsonl_file.is_file():
            rel_path = str(jsonl_file.relative_to(experiment_dir))
            if rel_path not in data_provenance["files"]:
                file_hash = compute_data_hash(jsonl_file)
                data_provenance["files"][rel_path] = {
                    "path": rel_path,
                    "hash": file_hash
                }

    manifest["data_provenance"] = data_provenance

    # Eval results
    eval_results_path = experiment_dir / "eval_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            manifest["eval_results"] = json.load(f)
    else:
        manifest["eval_results"] = None

    # Comparison
    comparison_path = experiment_dir / "comparison.json"
    if comparison_path.exists():
        with open(comparison_path, "r") as f:
            manifest["comparison"] = json.load(f)
    else:
        manifest["comparison"] = None

    # Training (checkpoint paths)
    training: dict[str, Any] = {"checkpoints": []}
    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for ckpt_dir in sorted(checkpoints_dir.glob("ckpt-*")):
            if ckpt_dir.is_dir():
                training["checkpoints"].append(str(ckpt_dir.relative_to(experiment_dir)))

    manifest["training"] = training

    # Write manifest to experiment_dir
    manifest_path = experiment_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")

    return manifest


def create_experiment_bundle(
    experiment_dir: Path,
    output_path: Path | None = None,
    include_checkpoints: bool = True,
) -> Path:
    """Create tar.gz archive of experiment artifacts.

    Args:
        experiment_dir: Path to experiment directory
        output_path: Output bundle path (default: experiment_dir/bundle.tar.gz)
        include_checkpoints: Whether to include checkpoints/ directory

    Returns:
        Path to created bundle
    """
    if output_path is None:
        output_path = experiment_dir / "bundle.tar.gz"

    # Ensure manifest exists
    create_experiment_manifest(experiment_dir)

    print(f"Creating experiment bundle: {output_path}")

    # Files/directories to include (always)
    always_include = [
        "manifest.json",
        "config.json",
        "state.json",
        "metadata.json",
        "eval_results.json",
        "comparison.json",
    ]

    # Directories that might exist
    dir_include = [
        "eval_results",
        "comparison",
    ]

    files_added = 0

    with tarfile.open(output_path, "w:gz") as tar:
        # Add individual files
        for filename in always_include:
            file_path = experiment_dir / filename
            if file_path.exists():
                arcname = file_path.relative_to(experiment_dir)
                tar.add(file_path, arcname=arcname)
                print(f"  Added: {arcname}")
                files_added += 1

        # Add directories
        for dirname in dir_include:
            dir_path = experiment_dir / dirname
            if dir_path.exists() and dir_path.is_dir():
                arcname = dir_path.relative_to(experiment_dir)
                tar.add(dir_path, arcname=arcname)
                print(f"  Added: {arcname}/")
                files_added += 1

        # Add checkpoints if requested
        if include_checkpoints:
            checkpoints_dir = experiment_dir / "checkpoints"
            if checkpoints_dir.exists() and checkpoints_dir.is_dir():
                arcname = checkpoints_dir.relative_to(experiment_dir)
                tar.add(checkpoints_dir, arcname=arcname)
                print(f"  Added: {arcname}/")
                files_added += 1

        # Add any .md report files
        for md_file in experiment_dir.glob("*.md"):
            if md_file.is_file():
                arcname = md_file.relative_to(experiment_dir)
                tar.add(md_file, arcname=arcname)
                print(f"  Added: {arcname}")
                files_added += 1

    bundle_size = output_path.stat().st_size
    print(f"Bundle created: {output_path} ({bundle_size:,} bytes, {files_added} items)")

    return output_path


def upload_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    """Upload checkpoint to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to checkpoint directory
        repo_id: HuggingFace Hub repo ID (e.g., "username/model-name")
        commit_message: Commit message (default: auto-generated)
        private: Whether repo should be private

    Returns:
        HuggingFace Hub URL if successful, empty string on error
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Warning: huggingface_hub not installed")
        print("  To enable HuggingFace Hub upload, install: pip install huggingface-hub")
        return ""

    try:
        api = HfApi()

        # Create repo if it doesn't exist
        print(f"Creating/verifying repo: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )

        # Upload checkpoint
        message = commit_message or f"Upload checkpoint from {checkpoint_dir.name}"
        print(f"Uploading checkpoint to {repo_id}...")

        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=message
        )

        hub_url = f"https://huggingface.co/{repo_id}"
        print(f"Upload successful: {hub_url}")

        return hub_url

    except Exception as e:
        print(f"Warning: HuggingFace Hub upload failed: {e}")
        print("  Continuing without upload (non-blocking error)")
        return ""
