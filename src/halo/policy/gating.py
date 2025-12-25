"""Gating logic for HALO Agent.

Decides when to escalate from worker to manager.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from ..obs.page_type import is_high_stakes_page


@dataclass
class GatingDecision:
    """Result of gating check."""
    should_call_manager: bool
    reason: str


def check_manager_gate(
    obs: Dict,
    action_history: List[str],
    loop_detected: bool = False,
    consecutive_errors: int = 0,
    page_type: str = "unknown"
) -> GatingDecision:
    """Check if manager should be called.

    Gating rules:
    1. Error present -> call manager
    2. Loop detected -> call manager
    3. High-stakes page (checkout/cart/login) -> call manager
    4. Too many consecutive errors -> call manager

    Args:
        obs: Current observation
        action_history: History of actions
        loop_detected: Whether loop was detected
        consecutive_errors: Number of consecutive errors
        page_type: Current page type

    Returns:
        GatingDecision
    """
    last_error = obs.get('last_action_error', '')

    # Rule 1: Error present
    if last_error:
        return GatingDecision(
            should_call_manager=True,
            reason=f"Action error: {last_error[:100]}"
        )

    # Rule 2: Loop detected
    if loop_detected:
        return GatingDecision(
            should_call_manager=True,
            reason="Loop detected - same state repeated"
        )

    # Rule 3: High-stakes page
    if is_high_stakes_page(page_type):
        return GatingDecision(
            should_call_manager=True,
            reason=f"High-stakes page: {page_type}"
        )

    # Rule 4: Too many consecutive errors (even if current step has no error)
    if consecutive_errors >= 2:
        return GatingDecision(
            should_call_manager=True,
            reason=f"Multiple consecutive errors: {consecutive_errors}"
        )

    # No manager needed
    return GatingDecision(
        should_call_manager=False,
        reason="Normal operation"
    )


class GatingController:
    """Stateful gating controller."""

    def __init__(
        self,
        error_threshold: int = 2,
        loop_threshold: int = 3,
        always_high_stakes: bool = True
    ):
        self.error_threshold = error_threshold
        self.loop_threshold = loop_threshold
        self.always_high_stakes = always_high_stakes
        self.consecutive_errors = 0
        self.manager_calls = 0

    def check(
        self,
        obs: Dict,
        action_history: List[str],
        loop_detected: bool,
        page_type: str
    ) -> GatingDecision:
        """Check gating and update state."""
        last_error = obs.get('last_action_error', '')

        # Update error count
        if last_error:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        decision = check_manager_gate(
            obs=obs,
            action_history=action_history,
            loop_detected=loop_detected,
            consecutive_errors=self.consecutive_errors,
            page_type=page_type
        )

        if decision.should_call_manager:
            self.manager_calls += 1

        return decision

    def reset(self):
        """Reset state for new episode."""
        self.consecutive_errors = 0
        self.manager_calls = 0
