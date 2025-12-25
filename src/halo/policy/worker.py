"""Worker policy for HALO Agent.

Uses gpt-4o-mini for fast action decisions.
Enforces strict element ID validation to prevent hallucination.
"""

import json
import logging
import re
from typing import Dict, Optional, Any, List, Set

logger = logging.getLogger(__name__)


# Action grammar reference
ACTION_GRAMMAR = """
Available actions (use exact syntax):
- click("bid") - Click element with browser ID
- fill("bid", "text") - Fill text input with value
- select_option("bid", "option") - Select dropdown option
- scroll(x, y) - Scroll page by x,y pixels
- go_back() - Navigate back
- go_forward() - Navigate forward
- goto("url") - Navigate to URL
- send_msg_to_user("message") - Send message to complete task
- noop() - Do nothing this step
"""


class WorkerPolicy:
    """Fast worker policy using gpt-4o-mini.
    
    Features:
    - Strict element ID validation (only uses IDs from observation)
    - Structured JSON output with rationale and confidence
    - Temperature=0 for determinism
    - Automatic repair for invalid IDs
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = None
        self.invalid_id_count = 0
        self.total_action_count = 0

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def get_action(
        self,
        obs_summary: str,
        goal: str,
        action_history: list = None,
        manager_guidance: Optional[Dict] = None,
        valid_bids: Optional[Set[str]] = None,
        last_action_error: str = ""
    ) -> Dict[str, Any]:
        """Get action from worker policy.

        Args:
            obs_summary: Summarized observation text
            goal: Task goal
            action_history: List of previous actions
            manager_guidance: Optional guidance from manager
            valid_bids: Set of valid element IDs from current observation
            last_action_error: Error from last action if any

        Returns:
            Dict with keys: action, rationale, confidence, raw_response
        """
        action_history = action_history or []
        valid_bids = valid_bids or set()
        self.total_action_count += 1
        
        # Debug: log valid_bids count
        if valid_bids:
            logger.debug(f"Worker has {len(valid_bids)} valid bids: {sorted(valid_bids)[:5]}...")
        else:
            logger.warning("Worker received EMPTY valid_bids set!")

        # Build system prompt with strict ID enforcement
        system_prompt = f"""You are a browser automation agent. Complete the task by interacting with page elements.

{ACTION_GRAMMAR}

CRITICAL - ELEMENT IDs:
- Element IDs are SHORT alphanumeric codes like "a1", "b2c", "x7y8" from the Actionable Elements list.
- NEVER use semantic names like "search", "button", "submit" - these will FAIL.
- Example: click("a1b2") is correct, click("search") is WRONG.

STRATEGY:
1. Look at the Actionable Elements list and find elements matching your goal.
2. Use click() to interact with buttons/links, fill() for text inputs.
3. For shopping: find search box → type product → click search → click product → add to cart.
4. For email: click compose → fill to/subject/body → click send.
5. Only use scroll(0, 300) if you need to see more elements.
6. Only send_msg_to_user after seeing explicit success confirmation.

Output JSON: {{"action": "click(\"a1b2\")", "rationale": "clicking search button", "confidence": 0.9}}
"""

        # Build user prompt
        user_parts = [f"# Goal\n{goal}"]
        
        # Add last action error prominently if present
        if last_action_error:
            user_parts.append(f"# ⚠️ LAST ACTION ERROR\n{last_action_error}\nYou must try a different action!")
        
        user_parts.append(obs_summary)

        if action_history:
            recent = action_history[-5:]  # Last 5 actions
            user_parts.append(f"# Recent Actions\n" + "\n".join(recent))

        if manager_guidance:
            user_parts.append(
                f"# Manager Guidance\n"
                f"Subgoal: {manager_guidance.get('subgoal', 'N/A')}\n"
                f"Skill: {manager_guidance.get('skill', 'N/A')}"
            )

        # Remind about valid IDs with emphasis
        if valid_bids:
            bid_list = ', '.join(sorted(valid_bids)[:30])
            user_parts.append(f"# VALID ELEMENT IDs - You MUST use one of these:\n{bid_list}")

        user_parts.append('# Your Next Action\nRespond with JSON: {"action": "...", "rationale": "...", "confidence": 0.0-1.0}')

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n\n".join(user_parts)}
                ],
                temperature=self.temperature,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            action = result.get("action", "noop()")
            rationale = result.get("rationale", "")
            confidence = result.get("confidence", 0.5)

            # Validate action format and element IDs
            action, was_repaired = self._validate_and_repair_action(action, valid_bids)
            if was_repaired:
                self.invalid_id_count += 1
                logger.warning(f"Repaired invalid action. Invalid ID rate: {self.invalid_id_count}/{self.total_action_count}")

            return {
                "action": action,
                "rationale": rationale,
                "confidence": confidence,
                "raw_response": content,
                "was_repaired": was_repaired
            }

        except Exception as e:
            logger.error(f"Worker policy error: {e}")
            return {
                "action": "noop()",
                "rationale": f"Error: {e}",
                "confidence": 0.0,
                "raw_response": "",
                "was_repaired": True
            }

    def _extract_bid_from_action(self, action: str) -> Optional[str]:
        """Extract element ID from action string."""
        # Match patterns like click("123"), click('123'), fill("456", "text"), fill('456', 'text')
        match = re.match(r'^(?:click|fill|select_option|hover|press|focus)\(["\']([^"\']+)["\']', action)
        if match:
            return match.group(1)
        return None

    def _validate_and_repair_action(self, action: str, valid_bids: Set[str]) -> tuple:
        """Validate action and repair if needed.
        
        Returns:
            Tuple of (action, was_repaired)
        """
        action = action.strip()
        was_repaired = False

        # Basic format validation
        valid_prefixes = [
            'click(', 'fill(', 'select_option(', 'scroll(',
            'go_back()', 'go_forward()', 'goto(', 'send_msg_to_user(',
            'noop()', 'hover(', 'press(', 'focus('
        ]

        has_valid_prefix = any(action.startswith(p) for p in valid_prefixes)
        if not has_valid_prefix:
            logger.warning(f"Invalid action format: {action}")
            return self._get_fallback_action(valid_bids), True

        # Check if action uses an element ID
        bid = self._extract_bid_from_action(action)
        if bid:
            # Validate bid format - real bids are alphanumeric like "a1b2", not words like "search"
            is_semantic_name = bid.isalpha() and len(bid) > 3  # Words like "search", "button"
            is_valid_bid = bid in valid_bids if valid_bids else True
            
            if is_semantic_name or (valid_bids and not is_valid_bid):
                logger.warning(f"Invalid element ID '{bid}' - using fallback (valid_bids: {len(valid_bids) if valid_bids else 0})")
                fallback = self._get_fallback_action(valid_bids)
                return fallback, True

        return action, was_repaired

    def _get_fallback_action(self, valid_bids: Set[str]) -> str:
        """Get a safe fallback action when element ID is invalid."""
        # If we have valid bids, try clicking the first one instead of scrolling
        if valid_bids:
            first_bid = sorted(valid_bids)[0]
            return f'click("{first_bid}")'
        # Only scroll if no valid bids available
        return "scroll(0, 300)"

    def get_invalid_id_rate(self) -> float:
        """Get the rate of invalid ID usage."""
        if self.total_action_count == 0:
            return 0.0
        return self.invalid_id_count / self.total_action_count


def create_worker_policy(model: str = "gpt-4o-mini") -> WorkerPolicy:
    """Factory function for worker policy."""
    return WorkerPolicy(model=model)
