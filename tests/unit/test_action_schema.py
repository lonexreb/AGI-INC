"""Test action schema validation."""

import pytest
from typing import Dict, Any


def validate_action(action: Dict[str, Any]) -> bool:
    """Validate action schema.

    Args:
        action: Action dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(action, dict):
        return False

    action_type = action.get("type")
    if action_type not in ["click", "type", "scroll", "wait", "done"]:
        return False

    # Type-specific validation
    if action_type == "click":
        return "target" in action
    elif action_type == "type":
        return "target" in action and "text" in action
    elif action_type == "scroll":
        return "direction" in action
    elif action_type == "wait":
        return "duration" in action or "condition" in action
    elif action_type == "done":
        return True

    return False


def test_click_action():
    """Test click action validation."""
    valid_click = {"type": "click", "target": "button#submit"}
    assert validate_action(valid_click) is True

    invalid_click = {"type": "click"}  # Missing target
    assert validate_action(invalid_click) is False


def test_type_action():
    """Test type action validation."""
    valid_type = {"type": "type", "target": "input#email", "text": "user@example.com"}
    assert validate_action(valid_type) is True

    invalid_type_no_target = {"type": "type", "text": "hello"}
    assert validate_action(invalid_type_no_target) is False

    invalid_type_no_text = {"type": "type", "target": "input#name"}
    assert validate_action(invalid_type_no_text) is False


def test_scroll_action():
    """Test scroll action validation."""
    valid_scroll = {"type": "scroll", "direction": "down"}
    assert validate_action(valid_scroll) is True

    invalid_scroll = {"type": "scroll"}  # Missing direction
    assert validate_action(invalid_scroll) is False


def test_wait_action():
    """Test wait action validation."""
    valid_wait_duration = {"type": "wait", "duration": 1000}
    assert validate_action(valid_wait_duration) is True

    valid_wait_condition = {"type": "wait", "condition": "element_visible"}
    assert validate_action(valid_wait_condition) is True

    invalid_wait = {"type": "wait"}  # Missing both duration and condition
    assert validate_action(invalid_wait) is False


def test_done_action():
    """Test done action validation."""
    valid_done = {"type": "done"}
    assert validate_action(valid_done) is True

    done_with_reason = {"type": "done", "reason": "Task completed"}
    assert validate_action(done_with_reason) is True


def test_invalid_action_types():
    """Test that invalid action types are rejected."""
    invalid_types = [
        {"type": "invalid_action"},
        {"type": "hover"},  # Not in our supported actions
        {"type": ""},
        {},
        {"action": "click"},  # Wrong key
    ]

    for invalid_action in invalid_types:
        assert validate_action(invalid_action) is False


def test_malformed_actions():
    """Test that malformed actions are rejected."""
    malformed_actions = [
        None,
        [],
        "click",
        123,
        {"type": None},
        {"type": ["click"]},
    ]

    for malformed in malformed_actions:
        assert validate_action(malformed) is False


def test_action_with_required_fields():
    """Test actions have required fields based on type."""
    # Each action type has specific required fields
    actions_with_requirements = [
        ({"type": "click", "target": "btn"}, True),
        ({"type": "click"}, False),  # Missing target
        ({"type": "type", "target": "input", "text": "test"}, True),
        ({"type": "type", "target": "input"}, False),  # Missing text
        ({"type": "scroll", "direction": "up"}, True),
        ({"type": "scroll"}, False),  # Missing direction
    ]

    for action, expected in actions_with_requirements:
        assert validate_action(action) is expected
