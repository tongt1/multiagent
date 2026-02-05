"""Role mask computation from trajectory text via marker string parsing.

This module computes per-role boolean token masks by parsing raw debate trajectory TEXT
for marker strings (verification prompt, final answer prompt), then mapping character
offsets to token positions. This implements CONTEXT.md Decision #1: text marker parsing,
NOT special token boundary detection.

Turn structure (from CONTEXT.md):
- Turn 0 (user: problem) → Turn 1 (chatbot: solver)
- Turn 2 (user: verify prompt) → Turn 3 (chatbot: verifier)
- Turn 4 (user: final answer prompt) → Turn 5 (chatbot: judge)

Only chatbot turns (1, 3, 5) get role labels. User turns (0, 2, 4) are injected prompts
excluded from ALL role masks.

Short debates (< 3 chatbot turns): assign all chatbot tokens to solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Configurable Marker Strings
# ============================================================================
#
# These marker strings are derived from the Comb conversation template and
# MathDebateNextSpeakerSelector turn structure. They detect role boundaries
# by searching for known prompts injected between chatbot turns.
#
# NOTE: The exact marker strings may vary based on the Comb conversation
# template configuration. If role masks consistently produce zero verifier/judge
# tokens, inspect actual rendered trajectories and update these constants.

DEFAULT_VERIFICATION_MARKERS: list[str] = [
    "verify",
    "check your",
    "review the",
]
"""Character strings that indicate the start of verifier role.

These appear in the injected user prompt (Turn 2) that transitions from
solver to verifier. Case-insensitive matching.
"""

DEFAULT_FINAL_ANSWER_MARKERS: list[str] = [
    "final answer",
    "provide your final",
    "give your final",
]
"""Character strings that indicate the start of judge role.

These appear in the injected user prompt (Turn 4) that transitions from
verifier to judge. Case-insensitive matching.
"""


# ============================================================================
# Marker Search
# ============================================================================


def find_marker_in_text(text: str, markers: list[str]) -> int:
    """Search text for any marker string and return character offset.

    Args:
        text: Full trajectory text to search
        markers: List of marker strings to search for

    Returns:
        Character offset of the FIRST marker found, or -1 if no marker found.
        Case-insensitive matching.

    Example:
        >>> find_marker_in_text("Hello. Please verify this.", ["verify", "check"])
        15  # Offset of "verify" in the text
    """
    text_lower = text.lower()

    earliest_offset = -1
    for marker in markers:
        offset = text_lower.find(marker.lower())
        if offset >= 0:
            if earliest_offset < 0 or offset < earliest_offset:
                earliest_offset = offset

    return earliest_offset


# ============================================================================
# Character-to-Token Offset Mapping
# ============================================================================


def map_char_offset_to_token_index(
    text: str,
    char_offset: int,
    tokenizer: Any,
) -> int:
    """Map character position in text to corresponding token index.

    Strategies (in order of preference):
    1. SentencePiece encode_as_serialized_proto() with offset information (fast)
    2. Prefix tokenization fallback (slower but always works)

    Args:
        text: Full trajectory text
        char_offset: Character position to map
        tokenizer: Duck-typed tokenizer with encode() method

    Returns:
        Token index corresponding to character offset. If char_offset falls
        inside a multi-character token, returns the token index containing it.

    Implementation note:
        The SentencePiece proto method is preferred but may not be available
        on all tokenizer wrappers (e.g., BPTokenizer). The prefix fallback
        is conservative and works with any tokenizer.encode() method.
    """
    # Strategy A: Try SentencePiece proto offsets (fast path)
    try:
        # Attempt to use encode_as_serialized_proto if available
        # This provides exact character-to-token offset mapping
        if hasattr(tokenizer, 'encode_as_serialized_proto'):
            import sentencepiece_pb2

            serialized = tokenizer.encode_as_serialized_proto(text, add_special_tokens=False)
            proto = sentencepiece_pb2.SentencePieceText()
            proto.ParseFromString(serialized)

            # Find token containing or immediately after char_offset
            for i, piece in enumerate(proto.pieces):
                if piece.begin <= char_offset < piece.end:
                    return i  # Offset falls inside this token
                if piece.begin >= char_offset:
                    return i  # Offset is at start of this token

            return len(proto.pieces)  # Offset is after all tokens

    except (AttributeError, ImportError, Exception) as e:
        # Proto method not available or failed - fall back to prefix search
        if isinstance(e, AttributeError):
            # Only log once per execution
            if not hasattr(map_char_offset_to_token_index, '_fallback_warned'):
                logger.debug(
                    "Tokenizer doesn't support encode_as_serialized_proto, using prefix fallback"
                )
                map_char_offset_to_token_index._fallback_warned = True

    # Strategy B: Prefix tokenization fallback (always works)
    prefix = text[:char_offset]

    # Handle edge case: empty prefix
    if not prefix:
        return 0

    # Tokenize prefix to find approximate token boundary
    # This is conservative: may be off by 1 at subword boundaries
    try:
        if hasattr(tokenizer, 'encode'):
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            if hasattr(prefix_tokens, '__len__'):
                return len(prefix_tokens)
            elif hasattr(prefix_tokens, 'ids'):
                # HuggingFace tokenizers return Encoding object
                return len(prefix_tokens.ids)
    except Exception as e:
        logger.warning(f"Prefix tokenization failed: {e}")
        return 0

    return 0


# ============================================================================
# Role Parsing
# ============================================================================


def parse_trajectory_roles(
    trajectory_text: str,
    tokenizer: Any,
    max_seq_len: int,
    verification_markers: list[str] | None = None,
    final_answer_markers: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Parse trajectory text to extract per-role token masks.

    This is the core parsing function. It searches for marker strings to find
    role boundaries, maps character offsets to token indices, and builds
    boolean masks for each role.

    Args:
        trajectory_text: Raw rendered debate trajectory text
        tokenizer: Duck-typed tokenizer with encode() method
        max_seq_len: Maximum sequence length (for mask shape)
        verification_markers: Marker strings for verifier role (None = use defaults)
        final_answer_markers: Marker strings for judge role (None = use defaults)

    Returns:
        Dict of boolean masks {"solver": [T], "verifier": [T], "judge": [T]}
        where T is the number of tokens in trajectory_text (up to max_seq_len).
        Returns None if parsing fails catastrophically.

    Short debates (< 3 chatbot turns):
        If verification or final answer markers are not found, assumes short
        debate and assigns all tokens to solver. Verifier and judge masks are
        all-False.

    User prompt exclusion:
        This function does NOT exclude user prompts - that happens in the
        compute_role_masks_from_trajectory() wrapper which intersects with
        prompt_mask.
    """
    if verification_markers is None:
        verification_markers = DEFAULT_VERIFICATION_MARKERS
    if final_answer_markers is None:
        final_answer_markers = DEFAULT_FINAL_ANSWER_MARKERS

    try:
        # Tokenize full trajectory
        tokens = tokenizer.encode(trajectory_text, add_special_tokens=False)

        # Handle different tokenizer return types
        if hasattr(tokens, '__len__'):
            T = min(len(tokens), max_seq_len)
        elif hasattr(tokens, 'ids'):
            # HuggingFace tokenizers
            T = min(len(tokens.ids), max_seq_len)
        else:
            logger.warning(f"Unknown tokenizer output type: {type(tokens)}")
            return None

        # Initialize masks (all False)
        masks = {
            "solver": np.zeros(T, dtype=np.int32),
            "verifier": np.zeros(T, dtype=np.int32),
            "judge": np.zeros(T, dtype=np.int32),
        }

        # Find character offsets of markers
        verify_char_offset = find_marker_in_text(trajectory_text, verification_markers)
        final_char_offset = find_marker_in_text(trajectory_text, final_answer_markers)

        # Check for short debate (markers not found)
        if verify_char_offset < 0 or final_char_offset < 0:
            logger.debug(
                f"Short debate detected (verify={verify_char_offset}, final={final_char_offset}), "
                "assigning all tokens to solver"
            )
            masks["solver"][:] = 1
            return masks

        # Map character offsets to token indices
        verify_token = map_char_offset_to_token_index(
            trajectory_text, verify_char_offset, tokenizer
        )
        final_token = map_char_offset_to_token_index(
            trajectory_text, final_char_offset, tokenizer
        )

        # Clamp to sequence length
        verify_token = min(verify_token, T)
        final_token = min(final_token, T)

        # Assign tokens to roles based on boundaries
        # NOTE: This assumes first chatbot turn starts at token 0
        # In practice, there may be a user prompt before solver tokens.
        # The caller (compute_role_masks_from_trajectory) handles this by
        # intersecting with prompt_mask to exclude user tokens.

        if verify_token >= final_token:
            # Degenerate case: final marker appears before verify marker
            logger.warning(
                f"Final marker ({final_token}) before verify marker ({verify_token}), "
                "assigning all to solver"
            )
            masks["solver"][:] = 1
            return masks

        # Full debate structure:
        # [0 : verify_token] = solver (includes initial user prompt, will be masked out)
        # [verify_token : final_token] = verifier (includes verify prompt, will be masked out)
        # [final_token : T] = judge (includes final prompt, will be masked out)

        masks["solver"][0:verify_token] = 1
        masks["verifier"][verify_token:final_token] = 1
        masks["judge"][final_token:T] = 1

        return masks

    except Exception as e:
        logger.error(f"Failed to parse trajectory: {e}", exc_info=True)
        return None


# ============================================================================
# Single-Sample Entry Point
# ============================================================================


def compute_role_masks_from_trajectory(
    trajectory_text: str,
    prompt_mask: np.ndarray,
    tokenizer: Any,
    max_seq_len: int,
    verification_markers: list[str] | None = None,
    final_answer_markers: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute role masks from trajectory text with prompt mask intersection.

    This is the single-sample entry point. It calls parse_trajectory_roles()
    to get initial role masks, then intersects with prompt_mask to exclude
    user-injected tokens.

    Args:
        trajectory_text: Raw rendered debate trajectory text
        prompt_mask: Boolean mask [T] where 1 = user/system token, 0 = chatbot token
        tokenizer: Duck-typed tokenizer with encode() method
        max_seq_len: Maximum sequence length
        verification_markers: Marker strings for verifier role (None = use defaults)
        final_answer_markers: Marker strings for judge role (None = use defaults)

    Returns:
        Dict of boolean masks {"solver": [T], "verifier": [T], "judge": [T]}
        where each mask is 1 only for chatbot-generated tokens in that role.
        Returns None if parsing fails.

    Shape note:
        GRPO computes loss on [B, T-1] (next-token prediction). Caller should
        slice masks[:, :-1] to match GRPO objective shape.
    """
    # Parse trajectory to get initial role assignments
    role_masks = parse_trajectory_roles(
        trajectory_text=trajectory_text,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        verification_markers=verification_markers,
        final_answer_markers=final_answer_markers,
    )

    if role_masks is None:
        return None

    # Intersect with prompt_mask to exclude user/system tokens
    # prompt_mask is 1 for user tokens, 0 for chatbot tokens
    # We want to EXCLUDE user tokens, so multiply by (1 - prompt_mask)
    chatbot_mask = 1 - prompt_mask

    # Ensure shape compatibility
    T = min(len(chatbot_mask), max_seq_len)
    for role in role_masks:
        # Truncate role mask to match chatbot_mask length
        role_masks[role] = role_masks[role][:T]
        # Exclude user tokens
        role_masks[role] = role_masks[role] * chatbot_mask[:len(role_masks[role])]

    return role_masks


# ============================================================================
# Batch Entry Point
# ============================================================================


def compute_batch_role_masks(
    batch_trajectories: list[str],
    batch_prompt_mask: np.ndarray,
    tokenizer: Any,
    max_seq_len: int,
    verification_markers: list[str] | None = None,
    final_answer_markers: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute role masks for a batch of trajectories.

    Args:
        batch_trajectories: List of trajectory strings [B]
        batch_prompt_mask: Boolean mask [B, T] where 1 = user/system token
        tokenizer: Duck-typed tokenizer
        max_seq_len: Maximum sequence length
        verification_markers: Marker strings for verifier role
        final_answer_markers: Marker strings for judge role

    Returns:
        Dict of boolean masks {"solver": [B, T], "verifier": [B, T], "judge": [B, T]}
        Returns None if all samples fail to parse (partial failures return partial masks).

    Partial failure handling:
        If some trajectories fail to parse, they get all-zero masks (no role labels).
        This allows training to continue with reduced per-role signal rather than crashing.
    """
    B = len(batch_trajectories)
    T = batch_prompt_mask.shape[1] if len(batch_prompt_mask.shape) > 1 else len(batch_prompt_mask)

    # Initialize batch masks
    batch_masks = {
        "solver": np.zeros((B, T), dtype=np.int32),
        "verifier": np.zeros((B, T), dtype=np.int32),
        "judge": np.zeros((B, T), dtype=np.int32),
    }

    successful_parses = 0

    for i, trajectory_text in enumerate(batch_trajectories):
        # Get prompt mask for this sample
        if len(batch_prompt_mask.shape) > 1:
            prompt_mask = batch_prompt_mask[i]
        else:
            # Batch size 1 case
            prompt_mask = batch_prompt_mask

        # Compute role masks for this sample
        sample_masks = compute_role_masks_from_trajectory(
            trajectory_text=trajectory_text,
            prompt_mask=prompt_mask,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            verification_markers=verification_markers,
            final_answer_markers=final_answer_markers,
        )

        if sample_masks is not None:
            # Copy into batch masks
            for role in batch_masks:
                mask_len = min(len(sample_masks[role]), T)
                batch_masks[role][i, :mask_len] = sample_masks[role][:mask_len]
            successful_parses += 1
        else:
            # Parse failed - leave all-zero masks for this sample
            logger.warning(f"Failed to parse trajectory {i}, using zero masks")

    if successful_parses == 0:
        logger.error("All trajectories failed to parse")
        return None

    if successful_parses < B:
        logger.warning(
            f"Partial batch parse success: {successful_parses}/{B} trajectories parsed"
        )

    return batch_masks
