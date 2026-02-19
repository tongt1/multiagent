#!/usr/bin/env python3
"""Patch a Llama tokenizer to add Cohere special token aliases.

The fax/flink RLVR pipeline uses Cohere special token strings
(e.g. <|START_OF_TURN_TOKEN|>, <|USER_TOKEN|>) in COMB-formatted data.
When using a Llama tokenizer, these strings are treated as regular text
(tokenized into 8-12 subword pieces each), causing prompts to balloon
in length and become semantically wrong.

This script adds Cohere special token strings as added_tokens in the
Llama tokenizer.json, mapping them to Llama's reserved token IDs.
Where a natural mapping exists (e.g. <|END_OF_TURN_TOKEN|> → <|eot_id|>),
the Cohere token REPLACES the Llama reserved token at that ID.
Other Cohere tokens are mapped to unused reserved slots.

Usage:
    python tools/patch_llama_tokenizer.py \
        --input /tmp/llama_eot_tokenizer.json \
        --output /tmp/llama_eot_tokenizer_patched.json

    # Then upload:
    gsutil cp /tmp/llama_eot_tokenizer_patched.json \
        gs://cohere-dev-central-2/users/terry/llama-1b-instruct-eot/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys


# Mapping: Cohere token string → Llama token ID to use.
# Tokens with a natural Llama equivalent share the same ID.
# Others use reserved slots starting at 128011.
COHERE_TO_LLAMA_ID = {
    # Tokens with natural Llama equivalents (share the ID):
    "<BOS_TOKEN>":              128000,  # same as <|begin_of_text|>
    "<EOS_TOKEN>":              128001,  # same as <|end_of_text|>
    "<PAD>":                    128004,  # same as <|finetune_right_pad_id|>
    "<|END_OF_TURN_TOKEN|>":    128009,  # same as <|eot_id|>
    "<UNK>":                    128002,  # reserved_special_token_0

    # Tokens without Llama equivalents (use reserved slots):
    "<|START_OF_TURN_TOKEN|>":  128011,  # reserved_special_token_3
    "<|USER_TOKEN|>":           128012,  # reserved_special_token_4
    "<|CHATBOT_TOKEN|>":        128013,  # reserved_special_token_5
    "<|SYSTEM_TOKEN|>":         128014,  # reserved_special_token_6
    "<|START_RESPONSE|>":       128015,  # reserved_special_token_7
    "<|END_RESPONSE|>":         128016,  # reserved_special_token_8
    "<|START_ACTION|>":         128017,  # reserved_special_token_9
    "<|END_ACTION|>":           128018,  # reserved_special_token_10
    "<|START_TOOL_RESULT|>":    128019,  # reserved_special_token_11
    "<|END_TOOL_RESULT|>":      128020,  # reserved_special_token_12
    "<|START_THINKING|>":       128021,  # reserved_special_token_13
    "<|END_THINKING|>":         128022,  # reserved_special_token_14
    "<EOP_TOKEN>":              128023,  # reserved_special_token_15
}


def patch_tokenizer(input_path: str, output_path: str) -> None:
    """Add Cohere special tokens to a Llama tokenizer.json."""
    with open(input_path) as f:
        data = json.load(f)

    added_tokens = data.get("added_tokens", [])

    # Build ID → index map for existing added_tokens
    id_to_idx: dict[int, int] = {}
    content_to_idx: dict[str, int] = {}
    for i, tok in enumerate(added_tokens):
        id_to_idx[tok["id"]] = i
        content_to_idx[tok["content"]] = i

    n_replaced = 0
    n_added = 0

    for cohere_str, target_id in COHERE_TO_LLAMA_ID.items():
        # Skip if the Cohere token already exists
        if cohere_str in content_to_idx:
            print(f"  SKIP {cohere_str} (already at id={added_tokens[content_to_idx[cohere_str]]['id']})")
            continue

        new_entry = {
            "id": target_id,
            "content": cohere_str,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }

        if target_id in id_to_idx:
            # Replace the existing entry at this ID (e.g. reserved token)
            old_content = added_tokens[id_to_idx[target_id]]["content"]
            print(f"  REPLACE id={target_id}: {old_content!r} → {cohere_str!r}")
            added_tokens[id_to_idx[target_id]] = new_entry
            n_replaced += 1
        else:
            # Add new entry
            print(f"  ADD id={target_id}: {cohere_str!r}")
            added_tokens.append(new_entry)
            n_added += 1

    # Sort by ID for cleanliness
    added_tokens.sort(key=lambda t: t["id"])
    data["added_tokens"] = added_tokens

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {n_replaced} replaced, {n_added} added")
    print(f"Total added_tokens: {len(added_tokens)}")
    print(f"Output: {output_path}")


def verify_tokenizer(path: str) -> None:
    """Verify the patched tokenizer works correctly."""
    # Import here to avoid requiring datatools just for patching
    from datatools.tokenizer.bpe import BPTokenizer

    tok = BPTokenizer(path)

    print("\n=== Verification ===")

    # Check key token IDs
    print(f"bos_token_id: {tok.bos_token_id}")
    print(f"eos_token_id: {tok.eos_token_id}")
    print(f"end_of_turn_token_id: {tok.end_of_turn_token_id}")
    print(f"start_of_turn_token_id: {tok.start_of_turn_token_id}")

    # Check encoding of Cohere special tokens
    tests = [
        ("<|START_OF_TURN_TOKEN|>", 1),
        ("<|END_OF_TURN_TOKEN|>", 1),
        ("<EOS_TOKEN>", 1),
        ("<BOS_TOKEN>", 1),
        ("<|USER_TOKEN|>", 1),
        ("<|CHATBOT_TOKEN|>", 1),
        ("<|SYSTEM_TOKEN|>", 1),
        ("<PAD>", 1),
        ("<UNK>", 1),
        ("<|START_RESPONSE|>", 1),
        ("<|END_RESPONSE|>", 1),
    ]

    all_ok = True
    for token_str, expected_len in tests:
        encoded = tok.encode(token_str, add_special_tokens=False)
        status = "OK" if len(encoded) == expected_len else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {status}: encode({token_str!r}) = {encoded} (len={len(encoded)}, expected={expected_len})")

    # Check the _EOS_MISSING_TEXT_TOKENS scenario (the "Bad assumptions" check)
    eos_missing = tok.encode("<|END_OF_TURN_TOKEN|><EOS_TOKEN>", add_special_tokens=False)
    status = "OK" if len(eos_missing) == 2 else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  {status}: encode('<|END_OF_TURN_TOKEN|><EOS_TOKEN>') = {eos_missing} (len={len(eos_missing)}, expected=2)")

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Patch Llama tokenizer with Cohere special tokens")
    parser.add_argument("--input", required=True, help="Input tokenizer.json path")
    parser.add_argument("--output", required=True, help="Output tokenizer.json path")
    parser.add_argument("--verify", action="store_true", help="Verify the output tokenizer after patching")
    args = parser.parse_args()

    print(f"Patching {args.input} → {args.output}")
    patch_tokenizer(args.input, args.output)

    if args.verify:
        verify_tokenizer(args.output)


if __name__ == "__main__":
    main()
