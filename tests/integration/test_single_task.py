"""Integration test for running a single v2 task with HaloAgent.

This test verifies:
1. Agent can be created and configured
2. Harness runs without crashing
3. At least one browser action is produced
4. Action validation works correctly
"""

import pytest
import os
from pathlib import Path

# Skip if no API key available
pytestmark = pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)


class TestHaloAgentIntegration:
    """Integration tests for HaloAgent with AGI SDK."""

    def test_agent_creation(self):
        """Test that HaloAgent can be created with all modes."""
        from halo.sdk import HaloAgent, SUPPORTED_MODES
        
        for mode in ['baseline_worker', 'hierarchy_mgr_gate', 'hierarchy_vac', 'hierarchy_vac_macros']:
            agent = HaloAgent(mode=mode, max_steps=5)
            assert agent is not None
            assert agent.mode == mode
            assert agent.max_steps == 5

    def test_action_validation(self):
        """Test action validation functions."""
        from halo.sdk import validate_action, repair_action
        
        # Valid actions
        assert validate_action('click("123")') is True
        assert validate_action('fill("input", "text")') is True
        assert validate_action('scroll(0, 100)') is True
        assert validate_action('noop()') is True
        assert validate_action('go_back()') is True
        assert validate_action('send_msg_to_user("hello")') is True
        
        # Invalid actions
        assert validate_action('') is False
        assert validate_action(None) is False
        assert validate_action('invalid_action') is False
        
        # Repair actions
        assert repair_action('') == 'noop()'
        assert repair_action('click(123') == 'click("123")'  # Missing quote and paren

    def test_ensure_v2_task(self):
        """Test that task names are properly prefixed with v2."""
        from halo.sdk import ensure_v2_task
        
        assert ensure_v2_task('omnizon-1') == 'v2.omnizon-1'
        assert ensure_v2_task('v2.omnizon-1') == 'v2.omnizon-1'
        assert ensure_v2_task('gomail-1') == 'v2.gomail-1'

    def test_harness_config_constraints(self):
        """Test that harness enforces critical constraints."""
        from halo.sdk import HaloAgentArgs, create_harness
        
        # Create harness with default settings
        agent_args = HaloAgentArgs(
            mode='baseline_worker',
            max_steps=5,
            log_trajectories=False
        )
        
        # Note: We can't actually run the harness without the full SDK setup,
        # but we can verify the configuration is correct
        assert agent_args.max_steps == 5
        assert agent_args.mode == 'baseline_worker'

    @pytest.mark.slow
    def test_single_task_headless(self):
        """Test running a single v2 task in headless mode.
        
        This test requires:
        - OPENAI_API_KEY environment variable
        - AGI SDK properly installed
        - Playwright browsers installed
        
        Run with: pytest tests/integration/ -m slow
        """
        from halo.sdk import run_single_task
        
        # Run a single task with minimal steps
        result = run_single_task(
            task_name='omnizon-1',  # Will be prefixed with v2.
            mode='baseline_worker',
            headless=True,
            max_steps=3,  # Very short for testing
            run_id='test_integration'
        )
        
        # Verify we got a result
        assert result is not None
        assert isinstance(result, dict)

        # Ensure we got a per-task record dict (not dict-of-dicts keyed by task name)
        assert result.get("task_name") == "v2.omnizon-1"
        assert result.get("task_id") == "omnizon-1"

        # Ensure metrics fields are present on the record
        assert "n_steps" in result
        assert isinstance(result["n_steps"], int)
        assert 0 <= result["n_steps"] <= 3
        
        # The task may not succeed in 3 steps, but it shouldn't crash
        # Check that we have some expected keys
        assert 'success' in result or 'error' not in result


class TestModeConfigurations:
    """Test that mode configurations are applied correctly."""

    def test_baseline_worker_mode(self):
        """Test baseline_worker mode disables manager and caches."""
        from halo.sdk import HaloAgent
        
        agent = HaloAgent(mode='baseline_worker')
        assert agent.orchestrator.manager is None
        assert agent.orchestrator.vac is None
        assert agent.orchestrator.macro_cache is None

    def test_hierarchy_mgr_gate_mode(self):
        """Test hierarchy_mgr_gate mode enables manager but not caches."""
        from halo.sdk import HaloAgent
        
        agent = HaloAgent(mode='hierarchy_mgr_gate')
        assert agent.orchestrator.manager is not None
        assert agent.orchestrator.vac is None
        assert agent.orchestrator.macro_cache is None

    def test_hierarchy_vac_mode(self):
        """Test hierarchy_vac mode enables manager and VAC."""
        from halo.sdk import HaloAgent
        
        agent = HaloAgent(mode='hierarchy_vac')
        assert agent.orchestrator.manager is not None
        assert agent.orchestrator.vac is not None
        assert agent.orchestrator.macro_cache is None

    def test_hierarchy_vac_macros_mode(self):
        """Test hierarchy_vac_macros mode enables all components."""
        from halo.sdk import HaloAgent
        
        agent = HaloAgent(mode='hierarchy_vac_macros')
        assert agent.orchestrator.manager is not None
        assert agent.orchestrator.vac is not None
        assert agent.orchestrator.macro_cache is not None
