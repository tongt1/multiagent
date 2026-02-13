"""Tests for custom Comb math debate environment.

These tests verify the debate environment structure and logic.
Tests work with or without Comb installation using conditional imports.
"""

from pathlib import Path

import pytest

# Import our module directly - it handles missing Comb gracefully
from src.training.comb_math_debate_env import (
    MathDebateScenarioConfig,
    MathDebateConversationSetup,
    MathDebateNextSpeakerSelector,
    MathDebateScenario,
    MathDebateScenarioBuilder,
    COMB_AVAILABLE,
)

# Get the appropriate interface module
if COMB_AVAILABLE:
    try:
        from comb import interface
        from datatools.types import agent_trajectory as data_model
    except ImportError:
        COMB_AVAILABLE = False

# If Comb still not available after import, use stubs
if not COMB_AVAILABLE:
    # Use stub interface from our module
    from src.training import comb_math_debate_env
    interface = comb_math_debate_env.interface
    data_model = comb_math_debate_env.data_model


class TestMathDebateScenarioConfig:
    """Tests for debate scenario configuration."""

    def test_debate_scenario_config_defaults(self):
        """Test that MathDebateScenarioConfig has correct defaults."""
        config = MathDebateScenarioConfig()

        assert hasattr(config, 'num_debate_rounds')
        assert config.num_debate_rounds == 3
        assert hasattr(config, 'use_format_reward')
        assert config.use_format_reward is False


class TestMathDebateNextSpeakerSelector:
    """Tests for debate turn-taking logic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.selector = MathDebateNextSpeakerSelector()

        # Mock turn class
        class MockTurn:
            def __init__(self, role):
                self.role = role

        self.MockTurn = MockTurn

        # Use proper ConversationState
        self.ConversationState = interface.ConversationState

    def test_next_speaker_selector_initial_turn(self):
        """Test that after initial user turn, chatbot speaks."""
        # Initial user turn (math problem)
        turns = [self.MockTurn(data_model.Role.user)]
        state = self.ConversationState(is_trackable=True, extra={})

        result = self.selector(turns, state)

        assert result.next_speaker_id == data_model.Role.chatbot
        assert state.extra.get('debate_round') == 0
        assert state.extra.get('phase') == 'solve'

    def test_next_speaker_selector_minimal_debate(self):
        """Test minimal debate sequence: solve -> verify -> final."""
        # Turn 0: User (problem)
        # Turn 1: Chatbot (initial solution)
        # Turn 2: User (verification prompt - injected)
        # Turn 3: Chatbot (verification/final answer)
        # Should terminate after 2 chatbot turns

        # After first chatbot turn
        turns = [
            self.MockTurn(data_model.Role.user),
            self.MockTurn(data_model.Role.chatbot),
        ]
        state = self.ConversationState(is_trackable=True, extra={'debate_round': 0, 'phase': 'solve'})

        result = self.selector(turns, state)

        # Should continue (inject user verification prompt)
        assert result.next_speaker_id == data_model.Role.user

        # After second chatbot turn (verification/final)
        turns.append(self.MockTurn(data_model.Role.user))  # Injected prompt
        turns.append(self.MockTurn(data_model.Role.chatbot))  # Verification/final

        result = self.selector(turns, state)

        # Should terminate (enough chatbot turns)
        assert result.next_speaker_id is None

    def test_next_speaker_selector_turn_sequence(self):
        """Test full turn sequence pattern."""
        state = self.ConversationState(is_trackable=True, extra={})
        turns = []

        # Turn 0: User problem
        turns.append(self.MockTurn(data_model.Role.user))
        result = self.selector(turns, state)
        assert result.next_speaker_id == data_model.Role.chatbot

        # Turn 1: Chatbot solver
        turns.append(self.MockTurn(data_model.Role.chatbot))
        result = self.selector(turns, state)
        # Should continue (only 1 chatbot turn so far)
        assert result.next_speaker_id == data_model.Role.user

        # Turn 2: User verification prompt (injected)
        turns.append(self.MockTurn(data_model.Role.user))
        result = self.selector(turns, state)
        assert result.next_speaker_id == data_model.Role.chatbot

        # Turn 3: Chatbot verifier/final
        turns.append(self.MockTurn(data_model.Role.chatbot))
        result = self.selector(turns, state)
        # Should terminate (2+ chatbot turns)
        assert result.next_speaker_id is None

    def test_next_speaker_selector_terminates(self):
        """Test that selector terminates after sufficient turns."""
        # Create a long conversation
        turns = [
            self.MockTurn(data_model.Role.user),
            self.MockTurn(data_model.Role.chatbot),
            self.MockTurn(data_model.Role.user),
            self.MockTurn(data_model.Role.chatbot),
            self.MockTurn(data_model.Role.user),
            self.MockTurn(data_model.Role.chatbot),
        ]
        state = self.ConversationState(is_trackable=True, extra={'debate_round': 2, 'phase': 'final'})

        result = self.selector(turns, state)

        # Should terminate with 3 chatbot turns
        assert result.next_speaker_id is None


class TestMathDebateScenarioStructure:
    """Tests for debate scenario structure."""

    def test_register_builder_decorator_present(self):
        """Test that @register_builder('math_debate') decorator is present."""
        # Read source file and verify decorator
        source_file = Path(__file__).parent.parent / "src" / "training" / "comb_math_debate_env.py"
        content = source_file.read_text()

        assert '@register_builder("math_debate")' in content

    def test_module_structure(self):
        """Test that all expected classes exist."""
        # Verify classes exist
        assert MathDebateScenarioBuilder is not None
        assert MathDebateScenarioConfig is not None
        assert MathDebateNextSpeakerSelector is not None
        assert MathDebateScenario is not None
        assert MathDebateConversationSetup is not None

        # Verify builder has __call__ method
        assert hasattr(MathDebateScenarioBuilder, '__call__')

        # Verify scenario has compute_reward method
        assert hasattr(MathDebateScenario, 'compute_reward')

    def test_builder_has_env_name(self):
        """Test that builder has been decorated with environment name."""
        # The decorator should set _comb_env_name attribute
        assert hasattr(MathDebateScenarioBuilder, '_comb_env_name')
        assert MathDebateScenarioBuilder._comb_env_name == "math_debate"


class TestMathDebateScenario:
    """Tests for debate scenario reward computation."""

    def test_compute_reward_method_signature(self):
        """Test that compute_reward has correct signature."""
        import inspect

        scenario = MathDebateScenario()
        sig = inspect.signature(scenario.compute_reward)

        # Should have turns and state parameters
        params = list(sig.parameters.keys())
        assert 'turns' in params
        assert 'state' in params

    def test_compute_format_reward_exists(self):
        """Test that compute_format_reward method exists."""
        scenario = MathDebateScenario()
        assert hasattr(scenario, 'compute_format_reward')
        assert callable(scenario.compute_format_reward)


class TestDebateStateManagement:
    """Tests for debate state tracking."""

    def test_debate_state_initialization(self):
        """Test that debate state is properly initialized."""
        selector = MathDebateNextSpeakerSelector()

        class MockTurn:
            def __init__(self, role):
                self.role = role

        turns = [MockTurn(data_model.Role.user)]
        state = interface.ConversationState(is_trackable=True, extra=None)

        result = selector(turns, state)

        # State should be initialized
        assert state.extra is not None
        assert 'debate_round' in state.extra
        assert 'phase' in state.extra
        assert state.extra['debate_round'] == 0
        assert state.extra['phase'] == 'solve'

    def test_debate_phase_transitions(self):
        """Test that debate phase transitions correctly."""
        selector = MathDebateNextSpeakerSelector()

        class MockTurn:
            def __init__(self, role):
                self.role = role

        # After solve phase, should transition to verify
        turns = [
            MockTurn(data_model.Role.user),
            MockTurn(data_model.Role.chatbot),
        ]
        state = interface.ConversationState(is_trackable=True, extra={'debate_round': 0, 'phase': 'solve'})

        result = selector(turns, state)

        # Phase should update
        assert state.extra['phase'] == 'verify'


class TestIntegration:
    """Integration tests for the complete environment."""

    def test_config_and_selector_integration(self):
        """Test that config and selector work together."""
        config = MathDebateScenarioConfig()
        selector = MathDebateNextSpeakerSelector()

        # Config should specify debate rounds
        assert config.num_debate_rounds > 0

        # Selector should be callable
        assert callable(selector)

    def test_builder_instantiation(self):
        """Test that builder can be instantiated."""
        builder = MathDebateScenarioBuilder()
        assert builder is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
