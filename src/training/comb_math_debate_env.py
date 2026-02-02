"""Custom Comb environment for math debate training.

This module provides a custom Comb environment that extends the existing math environment
with multi-speaker debate support. The debate structure uses a single model playing multiple
roles (solver, verifier, judge) through prompt engineering, matching RLVR's single-model
training paradigm.

The environment is registered as "math_debate" and discovered by CombItemsPreprocessor at
training time. Reward computation uses the same SmartAnswerValidator as the base math env.
"""

import dataclasses
import time
from typing import Any

try:
    from pydantic import Field
    from datatools.tokenizer.bpe import SPECIAL_TOKENS
    from datatools.types import agent_trajectory as data_model
    from hive import load_estimator
    from validators.smart_answer_validator import smart_answer_validator

    from comb import interface
    from comb.registry import register_builder
    from comb.shared import speakers as shared_speakers

    # Import ground truth verification for dual reward logging
    from src.evaluation.math_verifier import verify_math_answer

    COMB_AVAILABLE = True
except ImportError:
    # Comb not installed - provide minimal stubs for local testing
    COMB_AVAILABLE = False

    # Create stub classes that allow the module to be imported
    class StubBase:
        pass

    class interface:
        class ScenarioConfig(StubBase):
            pass
        class ConversationSetup(StubBase):
            pass
        class NextSpeakerSelector(StubBase):
            pass
        class Scenario(StubBase):
            pass
        class ScenarioBuilder(StubBase):
            pass
        class ConversationState:
            def __init__(self, is_trackable=True, extra=None):
                self.is_trackable = is_trackable
                self.extra = extra or {}
        class NextSpeakerSelectorOutput:
            def __init__(self, next_speaker_id, state):
                self.next_speaker_id = next_speaker_id
                self.state = state
        class RewardOutput:
            def __init__(self, reward, text_info=None, metrics=None):
                self.reward = reward
                self.text_info = text_info or {}
                self.metrics = metrics or {}

    class data_model:
        class Role:
            user = "user"
            chatbot = "chatbot"
        class Turn:
            pass
        class TextContent:
            pass
        class ReasoningContent:
            pass
        class ValidatorAnnotation:
            pass
        MISSING = "MISSING"

    def register_builder(name):
        """Stub decorator when Comb is not available."""
        def decorator(cls):
            cls._comb_env_name = name
            return cls
        return decorator

    def Field(default_factory=None, **kwargs):
        """Stub Field function when pydantic is not available."""
        if default_factory:
            return default_factory()
        return None

    SPECIAL_TOKENS = []


class MathDebateScenarioConfig(interface.ScenarioConfig):
    """Configuration for the math debate scenario.

    Extends the base ScenarioConfig with debate-specific parameters.
    """

    use_format_reward: bool = False
    hive_estimator_for_validators: str = Field(
        default_factory=lambda: "hive_estimator.command.CommandEstimator" if COMB_AVAILABLE else "default"
    )
    hive_estimator_config: dict[str, Any] = Field(default_factory=dict)
    num_debate_rounds: int = 3  # Number of solver-verifier exchanges before final answer


class MathDebateConversationSetup(interface.ConversationSetup):
    """Conversation setup for math debate environment."""

    scenario_config: MathDebateScenarioConfig
    validator_annotation: Any  # data_model.ValidatorAnnotation


class MathDebateNextSpeakerSelector(interface.NextSpeakerSelector):
    """Multi-turn speaker selector for debate structure.

    Implements the debate turn-taking pattern:
    - Turn 0 (user): Math problem → chatbot (solver) speaks
    - Turn 1 (chatbot): Initial solution → user (inject verification prompt)
    - Turn 2 (user): Verification prompt → chatbot (verifier) speaks
    - Turn 3 (chatbot): Verification response → user (inject final answer prompt)
    - Turn 4 (user): Final answer prompt → chatbot (final answer) speaks
    - Turn 5 (chatbot): Final answer → DONE

    The debate structure is encoded in conversation turns, not separate speakers.
    In RLVR training, there's only ONE model (the chatbot) being trained - it learns
    to play all roles through the prompts injected between its turns.
    """

    def __call__(
        self,
        turns: list[Any],  # list[data_model.Turn]
        state: interface.ConversationState
    ) -> interface.NextSpeakerSelectorOutput:
        """Select next speaker based on current turn count and debate state."""

        # Initialize debate state if needed
        if not hasattr(state, 'extra') or state.extra is None:
            state.extra = {}
        if 'debate_round' not in state.extra:
            state.extra['debate_round'] = 0
            state.extra['phase'] = 'solve'

        # Get current speaker from last turn
        last_turn_role = turns[-1].role if hasattr(turns[-1], 'role') else getattr(turns[-1], 'role', data_model.Role.user)

        # Determine next speaker based on last turn and debate state
        if last_turn_role == data_model.Role.user:
            # After user turn, chatbot speaks
            return interface.NextSpeakerSelectorOutput(
                next_speaker_id=data_model.Role.chatbot,
                state=state,
            )

        # After chatbot turn, check if we should continue debate or end
        debate_round = state.extra.get('debate_round', 0)
        phase = state.extra.get('phase', 'solve')

        # Debate pattern: solve -> verify -> final
        # Count chatbot turns to determine when to end
        chatbot_turn_count = sum(1 for t in turns if getattr(t, 'role', data_model.Role.user) == data_model.Role.chatbot)

        # Minimal debate: 2 chatbot turns (initial solve + verification/final)
        # Extended debate: more rounds add solve-verify pairs
        # For simplicity, terminate after 2+ chatbot turns (minimal debate achieved)
        if chatbot_turn_count >= 2:
            # End conversation after sufficient turns
            return interface.NextSpeakerSelectorOutput(
                next_speaker_id=None,
                state=state,
            )

        # Continue debate - inject next user prompt for verification or final answer
        if phase == 'solve':
            state.extra['phase'] = 'verify'
        elif phase == 'verify':
            state.extra['debate_round'] = debate_round + 1
            state.extra['phase'] = 'solve'

        # Return user turn (injected prompt) followed by chatbot
        return interface.NextSpeakerSelectorOutput(
            next_speaker_id=data_model.Role.user,  # Will be followed by chatbot
            state=state,
        )


@dataclasses.dataclass
class MathDebateScenario(interface.Scenario):
    """Math debate scenario with reward computation.

    Reward is computed on the LAST chatbot turn (final answer) using the same
    SmartAnswerValidator as the base math environment. The debate is a process
    that produces a final answer - reward is based on final answer correctness only.
    """

    def compute_format_reward(self, turns: list[Any]) -> float:
        """Compute reward based on completion format.

        Gives 1.0 if last turn has ReasoningContent + TextContent with no special tokens.
        Otherwise 0.0. This matches the base math environment behavior.
        """
        if not COMB_AVAILABLE:
            return 0.0

        last_turn = turns[-1]
        if (
            len(last_turn.contents) == 2
            and isinstance(last_turn.contents[0], data_model.ReasoningContent)
            and isinstance(last_turn.contents[1], data_model.TextContent)
            and all(token not in last_turn.contents[0].text for token in SPECIAL_TOKENS)
            and all(token not in last_turn.contents[1].text for token in SPECIAL_TOKENS)
        ):
            return 1.0
        else:
            return 0.0

    async def compute_reward(
        self,
        turns: list[Any],
        state: interface.ConversationState
    ) -> interface.RewardOutput:
        """Compute reward for the debate trajectory.

        Extracts the answer from the LAST chatbot turn and compares to ground truth
        using SmartAnswerValidator (same as base math env). Reward is binary: 1.0 or 0.0.

        The debate process (solver-verifier exchanges) is encoded in the trajectory,
        but reward is based solely on whether the final answer is correct.
        """
        if not COMB_AVAILABLE:
            # Fallback for local testing without Comb
            return interface.RewardOutput(
                reward=0.0,
                text_info={"error": "Comb not available"},
                metrics={
                    "correctness_score": 0.0,
                    "ground_truth_reward": 0.0,
                    "reward_agreement": 0.0,
                    "format_score": 0.0,
                    "validator_call_time": 0.0,
                },
            )

        # Default metrics structure for all error returns
        default_metrics = {
            "correctness_score": 0.0,
            "ground_truth_reward": 0.0,
            "reward_agreement": 0.0,
            "format_score": 0.0,
            "validator_call_time": 0.0,
        }

        # Find last chatbot turn
        chatbot_turns = [t for t in turns if t.role == data_model.Role.chatbot]
        if not chatbot_turns:
            return interface.RewardOutput(
                reward=0.0,
                text_info={"error": "No chatbot turns found"},
                metrics=default_metrics,
            )

        last_chatbot_turn = chatbot_turns[-1]

        # Basic validation
        if not last_chatbot_turn.contents:
            return interface.RewardOutput(
                reward=0.0,
                text_info={"error": "Empty chatbot turn"},
                metrics=default_metrics,
            )

        # Ensure last content is TextContent
        if not isinstance(last_chatbot_turn.contents[-1], data_model.TextContent):
            return interface.RewardOutput(
                reward=0.0,
                text_info={"error": "Last content not TextContent"},
                metrics=default_metrics,
            )

        # Extract parameters
        conv_setup: MathDebateConversationSetup = self.conversation_setup
        hive_estimator_class = conv_setup.scenario_config.hive_estimator_for_validators
        hive_estimator_config = conv_setup.scenario_config.hive_estimator_config

        # Get question from first user turn
        user_turns = [t for t in turns if t.role == data_model.Role.user]
        question = user_turns[0].contents[0].text if user_turns else ""

        # Get gold answer from validator annotation
        gold_answer = conv_setup.validator_annotation.spec.arguments["gold_answer"]

        # Extract answer from last chatbot turn
        extracted_answer = smart_answer_validator.SmartAnswerValidation._extract_value(
            last_chatbot_turn.contents[-1].text
        )

        # Compute correctness using SmartAnswerValidator
        start_time = time.perf_counter()
        deciding_model = load_estimator(
            cls_name=hive_estimator_class,
            usage="comb/validator",
            **hive_estimator_config,
        )
        await deciding_model._setup(skip_ready_check=True)

        try:
            (truth_value, decisions) = await smart_answer_validator.SmartAnswerValidation.single_compare_values(
                question=question,
                correct_answer=gold_answer,
                extracted_answer=extracted_answer,
                deciding_model=deciding_model,
                catch_exceptions=False,
            )
        finally:
            await deciding_model.cleanup()

        end = time.perf_counter()
        validator_call_time = end - start_time

        # Compute scores
        correctness_score = truth_value
        format_score = self.compute_format_reward(turns) * int(
            conv_setup.scenario_config.use_format_reward
        )

        # Compute ground truth binary reward for dual reward logging
        ground_truth_reward = 0.0
        if COMB_AVAILABLE:
            # Use symbolic equivalence verification
            gt_is_correct = verify_math_answer(extracted_answer, str(gold_answer))
            ground_truth_reward = 1.0 if gt_is_correct else 0.0

        # Compute reward agreement: 1 if BEE and GT agree, 0 otherwise
        # Cast BEE truth_value to binary (>= 0.5 -> 1.0, else 0.0)
        bee_binary = 1.0 if correctness_score >= 0.5 else 0.0
        reward_agreement = 1.0 if bee_binary == ground_truth_reward else 0.0

        # Build output
        metrics: dict[str, float | None] = {
            "correctness_score": correctness_score,
            "ground_truth_reward": ground_truth_reward,
            "reward_agreement": reward_agreement,
            "format_score": format_score,
            "validator_call_time": validator_call_time,
        }
        text_info: dict[str, str] = {
            "correct_answer": str(gold_answer),
            "extracted_answer": str(extracted_answer),
            "decisions": str(decisions),
        }

        return interface.RewardOutput(
            reward=correctness_score + format_score,
            text_info=text_info,
            metrics=metrics,
        )


@register_builder("math_debate")
class MathDebateScenarioBuilder(interface.ScenarioBuilder):
    """Builder for math debate scenarios.

    Registered as "math_debate" for Comb discovery. The builder creates a single-model
    setup where the chatbot plays all debate roles through prompt engineering.
    """

    async def __call__(
        self,
        data_item: Any,  # CommandDataCombItem
        scenario_config: MathDebateScenarioConfig | None = None,
        sampler_fn: Any = None,
    ) -> MathDebateScenario:
        """Build a math debate scenario from data item."""

        if not COMB_AVAILABLE:
            raise ImportError("Comb is not installed. Cannot build scenario.")

        # Setup configuration
        scenario_config = scenario_config or self.get_default_scenario_config(data_item=data_item)

        # Build data and sampler setup using parent class methods
        data_setup = self.build_data_setup(
            data_item=data_item,
            preamble_override=scenario_config.preamble_override,
            render_setting_override=scenario_config.render_setting_override,
        )
        sampler_setup = self.build_sampler_setup(
            data_item=data_item,
            scenario_config=scenario_config,
            sampler_fn=sampler_fn
        )
        conversation_setup = MathDebateConversationSetup(
            scenario_config=scenario_config,
            validator_annotation=data_item.validator_annotation,
        )

        # Create speakers - only chatbot (single model for all roles)
        speakers = {
            data_model.Role.chatbot: shared_speakers.DefaultChatbotSpeaker(
                data_setup=data_setup,
                sampler_setup=sampler_setup,
                conversation_setup=conversation_setup,
            ),
        }

        # Create next speaker selector for multi-turn debate
        next_speaker_selector = MathDebateNextSpeakerSelector()

        return MathDebateScenario(
            speakers=speakers,
            next_speaker_selector=next_speaker_selector,
            conversation_setup=conversation_setup,
            initial_state=interface.ConversationState(is_trackable=True, extra={}),
            initial_turns=self.get_initial_turns(data_item=data_item),
        )

    def get_default_scenario_config(self, data_item: Any) -> MathDebateScenarioConfig:
        """Get default configuration for math debate scenario."""
        del data_item  # unused
        return MathDebateScenarioConfig()
