#!/usr/bin/env python3
"""Convert HuggingFace LlamaForCausalLM weights to fax/megazord checkpoint format.

This script downloads safetensors weights from a HuggingFace model, applies
reverse weight transforms, and writes a valid fax checkpoint using zarr v2
format. No fax framework or GPU is required.

Usage:
    python tools/convert_hf_to_fax.py \
        --hf-model meta-llama/Llama-3.2-1B-Instruct \
        --output-dir /mnt/data/terry/scratch/llama-1b-ckpt/ckpt-0 \
        --dtype fp32
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd


# ---------------------------------------------------------------------------
# Zarr v2 writer (minimal, no zarr library dependency)
# ---------------------------------------------------------------------------

def _zarr_dtype_str(dtype: np.dtype) -> str:
    """Return zarr-compatible dtype string."""
    if dtype == np.float32:
        return "<f4"
    elif dtype == np.float16:
        return "<f2"
    elif hasattr(dtype, "name") and dtype.name == "bfloat16":
        # tensorstore-compatible extension
        return "bfloat16"
    else:
        return dtype.str


def write_zarr_array(path: str, array: np.ndarray, compressor_level: int = 1) -> None:
    """Write a numpy array as a zarr v2 array with zstd compression.

    Creates a directory at `path` containing:
      - .zarray  (JSON metadata)
      - 0 or 0.0... (single compressed chunk)
    """
    os.makedirs(path, exist_ok=True)

    dtype_str = _zarr_dtype_str(array.dtype)
    shape = list(array.shape) if array.ndim > 0 else []
    chunks = list(array.shape) if array.ndim > 0 else []

    # Determine fill_value
    if np.issubdtype(array.dtype, np.floating):
        fill_value = 0.0
    elif np.issubdtype(array.dtype, np.integer):
        fill_value = 0
    elif np.issubdtype(array.dtype, np.unsignedinteger):
        fill_value = 0
    else:
        fill_value = 0.0

    zarray_meta = {
        "chunks": chunks,
        "compressor": {
            "id": "zstd",
            "level": compressor_level,
        },
        "dimension_separator": ".",
        "dtype": dtype_str,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": shape,
        "zarr_format": 2,
    }

    with open(os.path.join(path, ".zarray"), "w") as f:
        json.dump(zarray_meta, f, indent=4)

    # Compress and write data as single chunk
    raw_bytes = array.tobytes(order="C")
    cctx = zstd.ZstdCompressor(level=compressor_level)
    compressed = cctx.compress(raw_bytes)

    # Chunk filename: "0" for scalar/1D, "0.0" for 2D, "0.0.0" for 3D, etc.
    if array.ndim <= 1:
        chunk_name = "0"
    else:
        chunk_name = ".".join(["0"] * array.ndim)

    with open(os.path.join(path, chunk_name), "wb") as f:
        f.write(compressed)


# ---------------------------------------------------------------------------
# HuggingFace model loading
# ---------------------------------------------------------------------------

def load_hf_model(hf_model: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load HF model weights and config.

    Args:
        hf_model: HF model name (e.g. 'meta-llama/Llama-3.2-1B-Instruct')
                  or local directory path.

    Returns:
        (weights_dict, config_dict)
    """
    # Register bfloat16 with numpy before safetensors tries to use it
    try:
        import ml_dtypes  # noqa: F401 — registers bfloat16 dtype with numpy
    except ImportError:
        pass
    from safetensors.numpy import load_file

    model_dir = Path(hf_model)
    if not model_dir.is_dir():
        # Download from HuggingFace Hub
        from huggingface_hub import snapshot_download
        print(f"Downloading {hf_model} from HuggingFace Hub...")
        model_dir = Path(snapshot_download(
            hf_model,
            allow_patterns=["*.safetensors", "config.json", "*.json"],
            ignore_patterns=["*.bin", "*.pt", "*.onnx"],
        ))
        print(f"Downloaded to {model_dir}")

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load all safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    weights = {}
    for sf_path in safetensor_files:
        print(f"Loading {sf_path.name}...")
        file_weights = load_file(str(sf_path))
        weights.update(file_weights)

    print(f"Loaded {len(weights)} weight tensors")
    return weights, config


# ---------------------------------------------------------------------------
# Weight transforms (HF -> fax internal)
# ---------------------------------------------------------------------------

def transform_weights(
    weights: dict[str, np.ndarray],
    config: dict,
    target_dtype: np.dtype,
) -> tuple[dict, int, int, int, int, int, int]:
    """Apply reverse transforms to convert HF weights to fax internal format.

    Returns:
        (fax_params_dict, dim, n_heads, head_dim, n_kv_heads, intermediate_size, n_layers)
    """
    dim = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", dim // n_heads)
    intermediate_size = config["intermediate_size"]
    n_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]

    print(f"\nModel config:")
    print(f"  hidden_size={dim}, n_heads={n_heads}, head_dim={head_dim}")
    print(f"  n_kv_heads={n_kv_heads}, intermediate_size={intermediate_size}")
    print(f"  n_layers={n_layers}, vocab_size={vocab_size}")
    print(f"  target_dtype={target_dtype}")

    def cast(arr: np.ndarray) -> np.ndarray:
        return arr.astype(target_dtype)

    # Build the fax pytree structure
    # Top-level keys use module __str__() names
    embed_key = f"input_embed,vocab-{vocab_size},emb_dim-{dim},pos_emb-rope"
    block_key = f"transf_block,dim-{dim},n_heads-{n_heads}"

    fax_params: dict[str, Any] = {}

    # --- Input embedding ---
    embed_w = cast(weights["model.embed_tokens.weight"])  # (V, D)
    assert embed_w.shape == (vocab_size, dim), f"embed shape {embed_w.shape}"
    fax_params[embed_key] = {
        "params": {
            "input_embed": embed_w,
        }
    }
    print(f"  input_embed: {embed_w.shape}")

    # --- Transformer layers ---
    block_params: dict[str, Any] = {}
    for i in range(n_layers):
        prefix = f"model.layers.{i}"

        # Q projection: (n_heads*head_dim, dim) -> (dim, n_heads, head_dim)
        q_w = cast(weights[f"{prefix}.self_attn.q_proj.weight"])  # (D, D)
        q_fax = q_w.T.reshape(dim, n_heads, head_dim)

        # K projection: (n_kv*head_dim, dim) -> (dim, n_kv, head_dim)
        k_w = cast(weights[f"{prefix}.self_attn.k_proj.weight"])
        k_fax = k_w.T.reshape(dim, n_kv_heads, head_dim)

        # V projection: (n_kv*head_dim, dim) -> (dim, n_kv, head_dim)
        v_w = cast(weights[f"{prefix}.self_attn.v_proj.weight"])
        v_fax = v_w.T.reshape(dim, n_kv_heads, head_dim)

        # Combined KV: (dim, 2, n_kv_heads, head_dim)
        kv_fax = np.stack([k_fax, v_fax], axis=1)

        # Output projection: (dim, n_heads*head_dim) -> (n_heads*head_dim, dim)
        o_w = cast(weights[f"{prefix}.self_attn.o_proj.weight"])
        attn_out_fax = o_w.T  # (n_heads*head_dim, dim)

        # Gate projection: (intermediate_size, dim) -> part of (dim, 2, intermediate_size)
        gate_w = cast(weights[f"{prefix}.mlp.gate_proj.weight"])
        # Up projection: (intermediate_size, dim) -> part of (dim, 2, intermediate_size)
        up_w = cast(weights[f"{prefix}.mlp.up_proj.weight"])
        # FFN expansion: stack [up.T, gate.T] -> (dim, 2, intermediate_size)
        ffn_exp_fax = np.stack([up_w.T, gate_w.T], axis=1)

        # Down projection: (dim, intermediate_size) -> (intermediate_size, dim)
        down_w = cast(weights[f"{prefix}.mlp.down_proj.weight"])
        ffn_red_fax = down_w.T  # (intermediate_size, dim)

        # Layer norms (identity transform)
        ln1_scale = cast(weights[f"{prefix}.input_layernorm.weight"])
        ln2_scale = cast(weights[f"{prefix}.post_attention_layernorm.weight"])

        block_params[f"rep_{i}"] = {
            "params": {
                "attn_layer": {
                    "q_proj": q_fax,
                    "kv_proj": kv_fax,
                    "attn_output": attn_out_fax,
                },
                "ffn_layer": {
                    "ffn_expansion": ffn_exp_fax,
                    "ffn_reduction": ffn_red_fax,
                },
                "rms_norm": {
                    "scale": ln1_scale,
                },
                "rms_norm_2": {
                    "scale": ln2_scale,
                },
            }
        }

        if i == 0:
            print(f"  Layer {i} shapes:")
            print(f"    q_proj:        {q_fax.shape}")
            print(f"    kv_proj:       {kv_fax.shape}")
            print(f"    attn_output:   {attn_out_fax.shape}")
            print(f"    ffn_expansion: {ffn_exp_fax.shape}")
            print(f"    ffn_reduction: {ffn_red_fax.shape}")
            print(f"    rms_norm:      {ln1_scale.shape}")
            print(f"    rms_norm_2:    {ln2_scale.shape}")

    fax_params[block_key] = block_params

    # --- Final RMS norm ---
    final_norm = cast(weights["model.norm.weight"])
    fax_params["rms_norm"] = {
        "params": {
            "rms_norm": {
                "scale": final_norm,
            }
        }
    }
    print(f"  final_rms_norm:  {final_norm.shape}")

    return fax_params, dim, n_heads, head_dim, n_kv_heads, intermediate_size, n_layers


# ---------------------------------------------------------------------------
# Checkpoint writer
# ---------------------------------------------------------------------------

def _write_pytree_zarr(base_dir: str, pytree: dict, path_prefix: str = "") -> None:
    """Recursively write a pytree of numpy arrays as zarr arrays under base_dir."""
    for key, value in pytree.items():
        current_path = os.path.join(path_prefix, str(key)) if path_prefix else str(key)
        if isinstance(value, dict):
            _write_pytree_zarr(base_dir, value, current_path)
        elif isinstance(value, np.ndarray):
            zarr_path = os.path.join(base_dir, current_path, "gda")
            write_zarr_array(zarr_path, value)
        else:
            raise TypeError(f"Unexpected type {type(value)} at path {current_path}")


def _jax_dtype_str(dtype: np.dtype) -> str:
    """Return JAX-compatible dtype string for pruned_model_state.json."""
    if dtype == np.float32:
        return "float32"
    elif dtype == np.float16:
        return "float16"
    elif hasattr(dtype, "name") and dtype.name == "bfloat16":
        return "bfloat16"
    elif dtype == np.int32:
        return "int32"
    elif dtype == np.uint32:
        return "uint32"
    else:
        return dtype.name


def _build_shape_dtype_tree(pytree: dict) -> dict:
    """Build the pruned_model_state.json structure from a pytree."""
    result = {}
    for key, value in pytree.items():
        if isinstance(value, dict):
            result[key] = _build_shape_dtype_tree(value)
        elif isinstance(value, np.ndarray):
            result[key] = {
                "__jax_ShapeDtypeStruct__": True,
                "shape": list(value.shape),
                "dtype": _jax_dtype_str(value.dtype),
            }
    return result


def _build_shard_map(pytree: dict) -> dict:
    """Build model_shard_map.json with null sharding specs (single-GPU)."""
    result = {}
    for key, value in pytree.items():
        if isinstance(value, dict):
            result[key] = _build_shard_map(value)
        elif isinstance(value, np.ndarray):
            # Null sharding for each dimension
            result[key] = [None] * value.ndim
    return result


def write_fax_checkpoint(
    output_dir: str,
    fax_params: dict,
    config: dict,
) -> None:
    """Write a complete fax checkpoint to output_dir.

    Creates the structure:
        output_dir/
            zord-0/
                params/...
                random_key/gda/
                step/gda/
                pruned_model_state.json
                model_shard_map.json
                zord_metadata.json
            run_config.json
            _CHECKPOINT_IS_COMPLETE
    """
    zord_dir = os.path.join(output_dir, "zord-0")
    params_dir = os.path.join(zord_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # --- Write parameter arrays ---
    print("\nWriting parameter arrays...")
    _write_pytree_zarr(params_dir, fax_params)
    print("  Done writing parameter arrays.")

    # --- Write random_key (uint32 array of shape [2]) ---
    random_key = np.array([0, 0], dtype=np.uint32)
    write_zarr_array(os.path.join(zord_dir, "random_key", "gda"), random_key)

    # --- Write step (int32 scalar, value 0) ---
    step = np.int32(0)
    write_zarr_array(os.path.join(zord_dir, "step", "gda"), step)

    # --- Write pruned_model_state.json ---
    # Wrap params in trainable_params / non_trainable_params structure
    model_state = {
        "trainable_params": fax_params,
        "non_trainable_params": {},
    }
    shape_dtype_tree = _build_shape_dtype_tree(model_state)
    # Add random_key and step
    shape_dtype_tree["random_key"] = {
        "__jax_ShapeDtypeStruct__": True,
        "shape": [2],
        "dtype": "uint32",
    }
    shape_dtype_tree["step"] = {
        "__jax_ShapeDtypeStruct__": True,
        "shape": [],
        "dtype": "int32",
    }
    with open(os.path.join(zord_dir, "pruned_model_state.json"), "w") as f:
        json.dump(shape_dtype_tree, f)
    print("  Wrote pruned_model_state.json")

    # --- Write model_shard_map.json ---
    shard_map = {
        "trainable_params": _build_shard_map(fax_params),
        "non_trainable_params": {},
    }
    shard_map["random_key"] = [None]
    shard_map["step"] = []
    with open(os.path.join(zord_dir, "model_shard_map.json"), "w") as f:
        json.dump(shard_map, f)
    print("  Wrote model_shard_map.json")

    # --- Write zord_metadata.json ---
    with open(os.path.join(zord_dir, "zord_metadata.json"), "w") as f:
        json.dump({"total_hosts": 1}, f)
    print("  Wrote zord_metadata.json")

    # --- Write run_config.json ---
    # Build arch config from HF config
    arch_config = {
        "embedding_dim": config["hidden_size"],
        # fax ffn_dim = 2 * HF intermediate_size for swiglu (gate + up combined)
        "ffn_dim": config["intermediate_size"] * 2,
        "n_heads": config["num_attention_heads"],
        "n_layers": config["num_hidden_layers"],
        "vocab_size": config["vocab_size"],
        "head_dim": config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
        "max_positional_embedding_length": config.get("max_position_embeddings", 131072),
        "use_bias": False,
        "positional_embedding_type": "rope",
        "rope_base_frequency": config.get("rope_theta", 500000.0),
        "rope_style": "split",
        "transformer_block_type": "vanilla",
        "rank_fraction": None,
        "activation_fn": "swiglu",
        "architecture": "LinearTransformer",
        "share_input_output_emb": config.get("tie_word_embeddings", True),
        "norm_type": "rms_norm",
        "n_kv_heads": config.get("num_key_value_heads", config["num_attention_heads"]),
        "use_qk_norm": False,
        "init": "glorot_uniform",
        "mup": {
            "enable": False,
            "base_embedding_dim": None,
            "base_ffn_dim": None,
            "base_n_heads": None,
            "attn_scale": 1.0,
            "logit_scale": 1.0,
            "input_emb_scale": 1.0,
            "input_emb_init_std": 1.0,
            "output_emb_init_std": 1.0,
            "hidden_init_std": 1.0,
        },
        "norm_eps": config.get("rms_norm_eps", 1e-5),
        "layer_switch": 1,
        "sliding_window_size": 4096,
        "moe": {
            "n_experts": 1,
            "n_expert_groups": 1,
            "n_shared_experts": 0,
            "shared_expert_combination_strategy": "average",
            "router": "token_choice",
            "expert_selection_fn": "softmax",
        },
    }
    run_config = {
        "ckpt": {"format": {}},
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f)
    print("  Wrote run_config.json")

    # --- Write arch_config.json ---
    with open(os.path.join(output_dir, "arch_config.json"), "w") as f:
        json.dump(arch_config, f, indent=2)
    print("  Wrote arch_config.json")

    # --- Write metadata.json ---
    # fax expects: config_history keyed by step, with arch_config_dict, run_config_dict, fax_commit_hash
    # Also requires: n_tokens_seen, current_step at top level
    metadata = {
        "ckpt_version": 1,
        "config_history": {
            "null": {
                "arch_config_dict": arch_config,
                "run_config_dict": run_config,
                "fax_commit_hash": "00000000",
            }
        },
        "current_step": 0,
        "max_file_idx_seen": None,
        "accumulated_rows_for_datasets": None,
        "prefetched_batch": None,
        "latest_release_tag": None,
        "n_tokens_seen": None,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Wrote metadata.json")

    # --- Write sentinel file ---
    sentinel = os.path.join(output_dir, "_CHECKPOINT_IS_COMPLETE")
    with open(sentinel, "w") as f:
        pass
    print("  Wrote _CHECKPOINT_IS_COMPLETE")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_checkpoint(output_dir: str) -> bool:
    """Validate the checkpoint structure and shapes."""
    print("\nValidating checkpoint...")
    zord_dir = os.path.join(output_dir, "zord-0")
    ok = True

    # Check sentinel
    if not os.path.exists(os.path.join(output_dir, "_CHECKPOINT_IS_COMPLETE")):
        print("  FAIL: _CHECKPOINT_IS_COMPLETE missing")
        ok = False

    # Check metadata files
    for fname in ["pruned_model_state.json", "model_shard_map.json", "zord_metadata.json"]:
        fpath = os.path.join(zord_dir, fname)
        if not os.path.exists(fpath):
            print(f"  FAIL: {fname} missing")
            ok = False
        else:
            with open(fpath) as f:
                data = json.load(f)
            print(f"  OK: {fname} ({len(json.dumps(data))} bytes)")

    # Check zarr arrays
    params_dir = os.path.join(zord_dir, "params")
    zarr_count = 0
    for root, dirs, files in os.walk(params_dir):
        if ".zarray" in files:
            zarr_count += 1
            zarray_path = os.path.join(root, ".zarray")
            with open(zarray_path) as f:
                meta = json.load(f)
            rel = os.path.relpath(root, params_dir)
            print(f"  OK: {rel} shape={meta['shape']} dtype={meta['dtype']}")

    # Check step and random_key
    for name in ["step", "random_key"]:
        gda_path = os.path.join(zord_dir, name, "gda")
        if os.path.exists(os.path.join(gda_path, ".zarray")):
            zarr_count += 1
            print(f"  OK: {name}/gda")
        else:
            print(f"  FAIL: {name}/gda missing")
            ok = False

    print(f"\nTotal zarr arrays: {zarr_count}")
    if ok:
        print("Checkpoint validation PASSED")
    else:
        print("Checkpoint validation FAILED")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace LlamaForCausalLM to fax/megazord checkpoint format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Llama 3.2 1B Instruct (requires HF login + license acceptance)
  python tools/convert_hf_to_fax.py \\
      --hf-model meta-llama/Llama-3.2-1B-Instruct \\
      --output-dir /mnt/data/terry/scratch/llama-1b-ckpt/ckpt-0

  # Convert from local directory
  python tools/convert_hf_to_fax.py \\
      --hf-model /path/to/local/model \\
      --output-dir /mnt/data/terry/scratch/llama-1b-ckpt/ckpt-0

  # Validate only (skip conversion)
  python tools/convert_hf_to_fax.py \\
      --validate-only /mnt/data/terry/scratch/llama-1b-ckpt/ckpt-0
        """,
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace model name or local directory path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the fax checkpoint (e.g. .../ckpt-0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Target dtype for checkpoint weights (default: fp32)",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        metavar="CKPT_DIR",
        help="Only validate an existing checkpoint (skip conversion)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.validate_only:
        ok = validate_checkpoint(args.validate_only)
        sys.exit(0 if ok else 1)

    if not args.hf_model or not args.output_dir:
        print("Error: --hf-model and --output-dir are required for conversion", file=sys.stderr)
        sys.exit(1)

    # Determine target dtype
    if args.dtype == "fp32":
        target_dtype = np.float32
    elif args.dtype == "bf16":
        try:
            import ml_dtypes
            target_dtype = ml_dtypes.bfloat16
        except ImportError:
            print("Error: ml_dtypes package required for bf16. Install: pip install ml_dtypes",
                  file=sys.stderr)
            sys.exit(1)

    # Load HF model
    weights, config = load_hf_model(args.hf_model)

    # Transform weights
    fax_params, *_ = transform_weights(weights, config, target_dtype)

    # Free HF weights to save memory
    del weights

    # Write checkpoint
    write_fax_checkpoint(args.output_dir, fax_params, config)

    # Validate
    validate_checkpoint(args.output_dir)

    print(f"\nCheckpoint written to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Upload to GCS:")
    print(f"     gsutil -m cp -r {args.output_dir}/ gs://cohere-dev-central-2/users/terry/llama-1b-instruct/ckpt-0/")
    print(f"  2. Update model_profiles.py with the LLAMA_1B_INSTRUCT profile")


if __name__ == "__main__":
    main()
