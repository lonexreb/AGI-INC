"""Loop detection for HALO Agent.

Detects repeated states that indicate the agent is stuck.
"""

import logging
from typing import List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class LoopDetector:
    """Detects loops via repeated StateKey patterns."""

    def __init__(self, window_size: int = 5, threshold: int = 2):
        """Initialize loop detector.

        Args:
            window_size: Number of recent states to track
            threshold: Number of repetitions to trigger loop detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.state_history: deque = deque(maxlen=window_size * 2)
        self.action_history: deque = deque(maxlen=window_size * 2)

    def add_state(self, state_key: str, action: str):
        """Add state to history.

        Args:
            state_key: Current state key string
            action: Action taken
        """
        self.state_history.append(state_key)
        self.action_history.append(action)

    def is_loop(self) -> bool:
        """Check if current state indicates a loop.

        Returns:
            True if loop detected
        """
        if len(self.state_history) < self.threshold:
            return False

        # Check for repeated states
        recent = list(self.state_history)[-self.window_size:]

        # Count occurrences of most recent state
        current_state = recent[-1] if recent else None
        if current_state:
            count = sum(1 for s in recent if s == current_state)
            if count >= self.threshold:
                logger.warning(f"Loop detected: state repeated {count} times")
                return True

        # Check for action-state cycles
        if len(self.action_history) >= 4:
            recent_actions = list(self.action_history)[-4:]
            # Detect A-B-A-B pattern
            if (recent_actions[0] == recent_actions[2] and
                recent_actions[1] == recent_actions[3]):
                logger.warning("Loop detected: action cycle A-B-A-B")
                return True

        return False

    def get_loop_info(self) -> Optional[str]:
        """Get information about detected loop.

        Returns:
            Loop description or None
        """
        if not self.is_loop():
            return None

        recent = list(self.state_history)[-self.window_size:]
        current = recent[-1] if recent else "unknown"
        count = sum(1 for s in recent if s == current)

        return f"State repeated {count} times in last {len(recent)} steps"

    def reset(self):
        """Reset loop detector state."""
        self.state_history.clear()
        self.action_history.clear()

    def get_repetition_count(self, state_key: str) -> int:
        """Get number of times a state has been seen recently.

        Args:
            state_key: State key to check

        Returns:
            Number of occurrences in recent history
        """
        return sum(1 for s in self.state_history if s == state_key)
