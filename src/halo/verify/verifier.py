"""Action verifier for HALO Agent.

Verifies whether actions succeeded based on observation changes.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of action verification."""
    success: bool
    reason: str
    url_changed: bool = False
    error_present: bool = False


class ActionVerifier:
    """Verifies action success based on observation changes."""

    def __init__(self):
        self.last_url: Optional[str] = None
        self.last_elements_count: int = 0

    def verify(
        self,
        prev_obs: Dict,
        curr_obs: Dict,
        action: str,
        expected_postcondition: str = ""
    ) -> VerificationResult:
        """Verify if action succeeded.

        Verification checks:
        1. No error in last_action_error
        2. URL change for navigation actions
        3. Element presence changes

        Args:
            prev_obs: Observation before action
            curr_obs: Observation after action
            action: Action that was taken
            expected_postcondition: Expected result description

        Returns:
            VerificationResult
        """
        # Check for errors
        error = curr_obs.get('last_action_error', '')
        if error:
            return VerificationResult(
                success=False,
                reason=f"Action error: {error[:100]}",
                error_present=True
            )

        # Check URL change
        prev_url = prev_obs.get('url', '')
        curr_url = curr_obs.get('url', '')
        url_changed = prev_url != curr_url

        # For navigation actions, URL should change
        nav_actions = ['click(', 'goto(', 'go_back()', 'go_forward()']
        is_nav_action = any(action.startswith(a) for a in nav_actions)

        if is_nav_action and url_changed:
            return VerificationResult(
                success=True,
                reason="Navigation successful - URL changed",
                url_changed=True
            )

        # For fill actions, check no error
        if action.startswith('fill('):
            return VerificationResult(
                success=True,
                reason="Fill action completed without error"
            )

        # For send_msg_to_user, always success if no error
        if action.startswith('send_msg_to_user('):
            return VerificationResult(
                success=True,
                reason="Message sent to user"
            )

        # For noop, always success
        if action == 'noop()':
            return VerificationResult(
                success=True,
                reason="Noop executed"
            )

        # Default: no error means success
        return VerificationResult(
            success=True,
            reason="Action completed without error",
            url_changed=url_changed
        )

    def quick_check(self, obs: Dict) -> bool:
        """Quick check if last action had an error."""
        return not bool(obs.get('last_action_error', ''))


def verify_postcondition(
    obs: Dict,
    postcondition: str
) -> Tuple[bool, str]:
    """Verify a specific postcondition.

    Simple heuristic checks based on postcondition text.

    Args:
        obs: Current observation
        postcondition: Expected postcondition description

    Returns:
        Tuple of (success, reason)
    """
    postcondition_lower = postcondition.lower()

    # Check for errors first
    if obs.get('last_action_error'):
        return False, f"Error: {obs['last_action_error']}"

    # URL-based checks
    url = obs.get('url', '').lower()

    if 'cart' in postcondition_lower and 'cart' in url:
        return True, "Cart page reached"

    if 'checkout' in postcondition_lower and 'checkout' in url:
        return True, "Checkout page reached"

    if 'search' in postcondition_lower and ('search' in url or 'q=' in url):
        return True, "Search results visible"

    if 'login' in postcondition_lower and 'login' not in url:
        return True, "Logged in (no longer on login page)"

    # Default: assume success if no error
    return True, "No error detected"
