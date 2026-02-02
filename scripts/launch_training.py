#!/usr/bin/env python3
"""Training launcher: data conversion + SWEEP job submission.

This script orchestrates the complete training pipeline:
1. Convert Phase 1 trajectories to Comb JSONL format
2. (Optionally) Upload converted data to GCS
3. Submit SWEEP training jobs for debate and/or baseline

Usage:
    # Convert data only (no submission)
    python scripts/launch_training.py --mode debate --convert-only

    # Convert + preview SWEEP config (no submission)
    python scripts/launch_training.py --mode debate --preview

    # Convert + submit training
    python scripts/launch_training.py --mode debate --submit --ckpt-path s3://...

    # Convert + submit both debate and baseline
    python scripts/launch_training.py --mode both --submit --ckpt-path s3://...

    # Use custom data paths (skip conversion)
    python scripts/launch_training.py --mode debate --submit --data-path gs://bucket/path/debate/train.jsonl --ckpt-path s3://...
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def convert_data(
    input_dir: Path, output_dir: Path, mode: str, eval_split: float = 0.1
) -> tuple[int, Path, Path]:
    """Convert Phase 1 trajectories to Comb JSONL format.

    Args:
        input_dir: Directory containing Phase 1 trajectory JSONL files
        output_dir: Output directory for converted Comb JSONL
        mode: "debate" or "baseline"
        eval_split: Fraction of data for evaluation (default: 0.1)

    Returns:
        (num_converted, train_path, eval_path)
    """
    print(f"\n=== Converting {mode} trajectories to Comb format ===")

    # Import converter module
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data.comb_converter import convert_trajectories_to_comb
    except ImportError as e:
        print(f"Error: Cannot import comb_converter: {e}")
        print("Make sure multiagent/src/data/comb_converter.py exists")
        sys.exit(1)

    # Find trajectory files for the specified mode
    trajectory_dir = input_dir / mode
    if not trajectory_dir.exists():
        print(f"Error: Trajectory directory not found: {trajectory_dir}")
        print(f"Expected structure: {input_dir}/debate/ or {input_dir}/baseline/")
        return 0, None, None

    # Collect all JSONL files
    jsonl_files = list(trajectory_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No JSONL files found in {trajectory_dir}")
        return 0, None, None

    print(f"Found {len(jsonl_files)} trajectory files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    # Create output directory
    mode_output_dir = output_dir / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert all files to a temporary combined file
    temp_output = mode_output_dir / "all.jsonl"
    total_converted = 0

    for jsonl_file in jsonl_files:
        count = convert_trajectories_to_comb(jsonl_file, temp_output, mode=mode)
        total_converted += count
        print(f"  Converted {count} trajectories from {jsonl_file.name}")

    if total_converted == 0:
        print("Error: No trajectories were converted")
        return 0, None, None

    print(f"\nTotal trajectories converted: {total_converted}")

    # Split into train and eval
    train_path = mode_output_dir / "train.jsonl"
    eval_path = mode_output_dir / "eval.jsonl"

    # Read all converted lines
    with open(temp_output, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Calculate split point
    eval_count = int(len(lines) * eval_split)
    train_count = len(lines) - eval_count

    print(f"Splitting data: {train_count} train, {eval_count} eval")

    # Write train split
    with open(train_path, "w", encoding="utf-8") as f:
        for line in lines[:train_count]:
            f.write(line + "\n")

    # Write eval split
    with open(eval_path, "w", encoding="utf-8") as f:
        for line in lines[train_count:]:
            f.write(line + "\n")

    # Remove temp file
    temp_output.unlink()

    print(f"Train data: {train_path}")
    print(f"Eval data: {eval_path}")
    print(f"Train size: {train_path.stat().st_size / 1024:.1f} KB")
    print(f"Eval size: {eval_path.stat().st_size / 1024:.1f} KB")

    return total_converted, train_path, eval_path


def upload_to_gcs(local_path: Path, gcs_bucket: str, gcs_prefix: str) -> str:
    """Upload converted data to GCS.

    Args:
        local_path: Local file path
        gcs_bucket: GCS bucket name (e.g., "my-bucket")
        gcs_prefix: GCS path prefix (e.g., "multiagent-debate-rl/debate")

    Returns:
        Full GCS path (gs://...)
    """
    gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}/{local_path.name}"
    print(f"\nUploading {local_path} to {gcs_path}")

    try:
        subprocess.run(
            ["gsutil", "cp", str(local_path), gcs_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Uploaded successfully")
        return gcs_path
    except FileNotFoundError:
        print("Error: gsutil not found. Install Google Cloud SDK or skip upload with --no-upload")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to GCS: {e.stderr}")
        sys.exit(1)


def submit_sweep_job(mode: str, data_path: str, eval_path: str, ckpt_path: str, preview: bool = False):
    """Submit SWEEP training job.

    Args:
        mode: "debate" or "baseline"
        data_path: Path to training data (GCS or local)
        eval_path: Path to eval data (GCS or local)
        ckpt_path: Path to base model checkpoint
        preview: If True, preview config without submission
    """
    print(f"\n=== {'Previewing' if preview else 'Submitting'} {mode} SWEEP job ===")

    # Determine SWEEP config module
    if mode == "debate":
        config_module = "configs.sweep_math_debate_grpo"
        config_class = "MathDebateGRPO"
    elif mode == "baseline":
        config_module = "configs.sweep_math_baseline_grpo"
        config_class = "MathBaselineGRPO"
    else:
        print(f"Error: Invalid mode '{mode}'. Must be 'debate' or 'baseline'")
        sys.exit(1)

    print(f"Using SWEEP config: {config_module}")
    print(f"Training data: {data_path}")
    print(f"Eval data: {eval_path}")
    print(f"Checkpoint: {ckpt_path}")

    # Update config file with actual paths
    config_file = Path(__file__).parent.parent / "configs" / f"sweep_math_{mode}_grpo.py"
    if not config_file.exists():
        print(f"Error: SWEEP config not found: {config_file}")
        sys.exit(1)

    # Read config
    with open(config_file, "r", encoding="utf-8") as f:
        config_content = f.read()

    # Replace placeholder paths
    if mode == "debate":
        config_content = config_content.replace(
            'DEBATE_DATA_PATH = "gs://your-bucket/multiagent-debate-rl/debate/train.jsonl"',
            f'DEBATE_DATA_PATH = "{data_path}"'
        )
        config_content = config_content.replace(
            'DEBATE_EVAL_PATH = "gs://your-bucket/multiagent-debate-rl/debate/eval.jsonl"',
            f'DEBATE_EVAL_PATH = "{eval_path}"'
        )
    else:
        config_content = config_content.replace(
            'BASELINE_DATA_PATH = "gs://your-bucket/multiagent-debate-rl/baseline/train.jsonl"',
            f'BASELINE_DATA_PATH = "{data_path}"'
        )
        config_content = config_content.replace(
            'BASELINE_EVAL_PATH = "gs://your-bucket/multiagent-debate-rl/baseline/eval.jsonl"',
            f'BASELINE_EVAL_PATH = "{eval_path}"'
        )

    config_content = config_content.replace(
        'CKPT_PATH = "s3://us-east-01a/foundations-experiments/viraat_cohere_com/MM/8x15B/merges_posttrain/add_only_vit/26A91T_7bua4gxj_vit_ckpt7999_59rfb4uy_llm_ckpt6405/ckpt-0/"',
        f'CKPT_PATH = "{ckpt_path}"'
    )

    # Write updated config to temp file
    temp_config = config_file.with_suffix(".tmp.py")
    with open(temp_config, "w", encoding="utf-8") as f:
        f.write(config_content)

    try:
        # Import sweep (only when actually submitting/previewing)
        try:
            import sweep
        except ImportError:
            print("\nError: 'sweep' module not available")
            print("SWEEP is Cohere internal infrastructure. This script expects to run in the Cohere internal environment.")
            print("\nTo run SWEEP jobs:")
            print(f"  1. Ensure you're in the correct Python environment with sweep installed")
            print(f"  2. Run: uv run python {temp_config} {'start' if preview else '--submit start'}")
            sys.exit(1)

        # Build command
        cmd = ["uv", "run", "python", str(temp_config)]
        if not preview:
            cmd.append("--submit")
        cmd.append("start")

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"Error: SWEEP command failed with exit code {result.returncode}")
            sys.exit(1)

        print(f"\n{'Preview' if preview else 'Submission'} complete!")

    finally:
        # Clean up temp file
        if temp_config.exists():
            temp_config.unlink()


def save_experiment_metadata(
    experiment_id: str,
    mode: str,
    data_paths: dict,
    ckpt_path: Optional[str],
    conversion_stats: dict,
):
    """Save experiment metadata for tracking.

    Args:
        experiment_id: Unique experiment identifier
        mode: "debate", "baseline", or "both"
        data_paths: Dict of mode -> (train_path, eval_path)
        ckpt_path: Base model checkpoint path
        conversion_stats: Dict with conversion statistics
    """
    experiments_dir = Path(__file__).parent.parent / "experiments"
    experiment_dir = experiments_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "data_paths": data_paths,
        "ckpt_path": ckpt_path,
        "conversion_stats": conversion_stats,
    }

    metadata_file = experiment_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExperiment metadata saved: {metadata_file}")
    print(f"Experiment ID: {experiment_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch training pipeline: data conversion + SWEEP submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debate", "baseline", "both"],
        required=True,
        help="Training mode: debate (multi-agent), baseline (single-agent), or both",
    )

    # Execution control
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit to cluster (default: preview only)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert data, don't launch SWEEP",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Convert data + show SWEEP config preview (no submission)",
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override data path (skip conversion). For --mode both, provide comma-separated: debate_path,baseline_path",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("trajectories"),
        help="Phase 1 trajectory directory (default: trajectories/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/comb_training"),
        help="Converted Comb JSONL output (default: data/comb_training/)",
    )

    # GCS upload
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload converted data to GCS (requires gsutil)",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default="your-bucket",
        help="GCS bucket name (default: your-bucket)",
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="multiagent-debate-rl",
        help="GCS path prefix (default: multiagent-debate-rl)",
    )

    # Training config
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Base model checkpoint S3 path (required for --submit)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment identifier (default: auto-generated from timestamp)",
    )

    args = parser.parse_args()

    # Validation
    if args.submit and not args.ckpt_path and not args.data_path:
        parser.error("--ckpt-path is required when using --submit")

    if args.convert_only and args.submit:
        parser.error("Cannot use both --convert-only and --submit")

    # Generate experiment ID
    if not args.experiment_id:
        args.experiment_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"Experiment ID: {args.experiment_id}")
    print(f"Mode: {args.mode}")

    # Determine modes to process
    modes = ["debate", "baseline"] if args.mode == "both" else [args.mode]

    # Data conversion or use provided paths
    data_paths = {}
    conversion_stats = {}

    if args.data_path:
        # Use provided data paths (skip conversion)
        print("\nUsing provided data paths (skipping conversion)")
        if args.mode == "both":
            paths = args.data_path.split(",")
            if len(paths) != 2:
                parser.error("--data-path with --mode both requires comma-separated debate_path,baseline_path")
            data_paths["debate"] = (paths[0], paths[0].replace("train", "eval"))
            data_paths["baseline"] = (paths[1], paths[1].replace("train", "eval"))
        else:
            eval_path = args.data_path.replace("train", "eval")
            data_paths[args.mode] = (args.data_path, eval_path)
    else:
        # Convert data
        for mode in modes:
            count, train_path, eval_path = convert_data(
                args.input_dir, args.output_dir, mode
            )
            if count == 0:
                print(f"Warning: No trajectories converted for {mode}")
                continue

            conversion_stats[mode] = {"count": count}

            # Upload to GCS if requested
            if args.upload:
                train_gcs = upload_to_gcs(train_path, args.gcs_bucket, f"{args.gcs_prefix}/{mode}")
                eval_gcs = upload_to_gcs(eval_path, args.gcs_bucket, f"{args.gcs_prefix}/{mode}")
                data_paths[mode] = (train_gcs, eval_gcs)
            else:
                # Use local paths
                data_paths[mode] = (str(train_path.absolute()), str(eval_path.absolute()))

    # Save experiment metadata
    save_experiment_metadata(
        args.experiment_id,
        args.mode,
        data_paths,
        args.ckpt_path,
        conversion_stats,
    )

    # SWEEP submission
    if not args.convert_only:
        for mode in modes:
            if mode not in data_paths:
                print(f"Skipping {mode} (no data available)")
                continue

            train_path, eval_path = data_paths[mode]
            submit_sweep_job(
                mode,
                train_path,
                eval_path,
                args.ckpt_path,
                preview=(args.preview or not args.submit),
            )

    print("\n=== Pipeline complete ===")
    print(f"Experiment ID: {args.experiment_id}")


if __name__ == "__main__":
    main()
