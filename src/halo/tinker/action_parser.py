"""Action parsing and validation for browser automation.

Shared logic for decoding model-generated text into valid browser action strings,
and validating element IDs (BIDs) against the current page state.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Valid action patterns for strict validation
VALID_ACTION_PATTERNS = [
    r'^click\("[^"]+"\)$',
    r'^fill\("[^"]+",\s*"[^"]*"\)$',
    r'^select_option\("[^"]+",\s*"[^"]*"\)$',
    r'^scroll\(-?\d+,\s*-?\d+\)$',
    r'^go_back\(\)$',
    r'^go_forward\(\)$',
    r'^goto\("[^"]+"\)$',
    r'^send_msg_to_user\("[^"]*"\)$',
    r'^noop\(\)$',
    r'^hover\("[^"]+"\)$',
    r'^press\("[^"]+"\)$',
    r'^focus\("[^"]+"\)$',
]

# Action extraction patterns (more lenient, for finding actions in free text)
ACTION_EXTRACTION_PATTERNS = [
    r'(click\("[^"]+"\))',
    r'(fill\("[^"]+",\s*"[^"]*"\))',
    r'(select_option\("[^"]+",\s*"[^"]*"\))',
    r'(scroll\(-?\d+,\s*-?\d+\))',
    r'(go_back\(\))',
    r'(go_forward\(\))',
    r'(goto\("[^"]+"\))',
    r'(send_msg_to_user\("[^"]*"\))',
    r'(noop\(\))',
    r'(hover\("[^"]+"\))',
    r'(press\("[^"]+"\))',
]

VALID_PREFIXES = [
    'click(', 'fill(', 'select_option(', 'scroll(',
    'go_back()', 'go_forward()', 'goto(', 'send_msg_to_user(',
    'noop()', 'hover(', 'press(', 'focus('
]

# Action grammar description for model prompts
ACTION_GRAMMAR = """Available actions (use exact syntax):
- click("bid") - Click element with browser ID
- fill("bid", "text") - Fill text input with value
- select_option("bid", "option") - Select dropdown option
- scroll(x, y) - Scroll page by x,y pixels (positive y scrolls down)
- go_back() - Navigate back
- go_forward() - Navigate forward
- goto("url") - Navigate to URL
- send_msg_to_user("message") - Send message to complete task
- noop() - Do nothing this step
- hover("bid") - Hover over element
- press("key") - Press keyboard key (e.g., "Enter", "Tab")
"""


def parse_action_from_text(text: str) -> str:
    """Parse an action string from model-generated text.

    Handles:
    - JSON: {"action": "click(\"a1b2\")", ...}
    - Raw action: click("a1b2")
    - Markdown code blocks wrapping JSON or actions

    Args:
        text: Raw text from model output

    Returns:
        Parsed action string (defaults to "noop()" on failure)
    """
    text = text.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()

    # Try JSON parse
    try:
        if "{" in text:
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                action = parsed.get("action", "")
                if action:
                    return action
    except (json.JSONDecodeError, KeyError):
        pass

    # Try direct JSON parse (whole string)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed["action"]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try regex extraction of action patterns
    for pattern in ACTION_EXTRACTION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    logger.debug(f"Failed to parse action from text: {text[:200]}")
    return "noop()"


def validate_action(
    action: str,
    valid_bids: Optional[Set[str]] = None,
) -> Tuple[str, bool]:
    """Validate an action string and repair if needed.

    Checks:
    1. Action has a valid prefix (click, fill, scroll, etc.)
    2. Element ID (bid) exists in the valid_bids set

    Args:
        action: Action string to validate
        valid_bids: Set of valid element IDs on current page

    Returns:
        Tuple of (possibly repaired action, was_repaired)
    """
    action = action.strip()

    # Check valid prefix
    has_valid_prefix = any(action.startswith(p) for p in VALID_PREFIXES)
    if not has_valid_prefix:
        return "noop()", True

    # Check bid validity
    if valid_bids:
        bid_match = re.match(
            r'^(?:click|fill|select_option|hover|press|focus)\("([^"]+)"',
            action,
        )
        if bid_match:
            bid = bid_match.group(1)
            if bid not in valid_bids:
                logger.debug(f"Invalid bid {bid}, valid: {list(valid_bids)[:10]}")
                return "scroll(0, 300)", True

    return action, False


def is_valid_action(action: str) -> bool:
    """Check if an action string matches any valid pattern (strict)."""
    if not action or not isinstance(action, str):
        return False
    action = action.strip()
    for pattern in VALID_ACTION_PATTERNS:
        if re.match(pattern, action):
            return True
    # Lenient: accept actions starting with valid prefixes
    return any(action.startswith(p) for p in VALID_PREFIXES)
