"""VLM Policy Client for Qwen3-VL-8B.

This module provides a clean wrapper for vLLM's OpenAI-compatible API
with vision capabilities. It handles batched action sampling with image inputs.

Usage:
    client = VLLMPolicyClient(base_url="http://localhost:8000/v1")

    # Sample multiple actions (for training)
    actions = client.sample_actions(screenshot, goal, n=8, temperature=0.7)

    # Sample single action (for eval, greedy)
    action = client.sample_single(screenshot, goal)
"""

import base64
import json
import logging
import re
from typing import List, Optional, Dict, Any, Set

logger = logging.getLogger(__name__)

# Action grammar for VLM prompts
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


class VLLMPolicyClient:
    """VLM Policy Client for Qwen3-VL-8B via vLLM.

    Handles:
    - Screenshot encoding (base64)
    - Chat completion format with images
    - Batched action sampling (n > 1)
    - Greedy sampling for evaluation
    - Action parsing and validation

    The client uses vLLM's OpenAI-compatible API which supports
    multimodal inputs via the chat completions endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_tokens: int = 300,
        timeout: float = 60.0,
    ):
        """Initialize VLM policy client.

        Args:
            base_url: vLLM OpenAI-compatible API base URL
            model: Model name (must match what vLLM is serving)
            max_tokens: Maximum tokens to generate per response
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = None

        # Stats
        self.total_requests = 0
        self.failed_requests = 0
        self.invalid_action_count = 0

    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="dummy",  # vLLM doesn't require real key
                timeout=self.timeout,
            )
        return self._client

    def _encode_image(self, screenshot: bytes) -> str:
        """Encode screenshot bytes to base64 data URL."""
        b64 = base64.b64encode(screenshot).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def _build_system_prompt(self) -> str:
        """Build system prompt for the VLM."""
        return f"""You are a browser automation agent controlling a web browser.
You see a screenshot of the current page and must decide what action to take.

{ACTION_GRAMMAR}

CRITICAL RULES:
1. ONLY use element IDs (bid values) that you can see in the screenshot or are given to you.
2. NEVER invent element IDs. If unsure, use scroll(0, 300) to see more content.
3. Do NOT call send_msg_to_user() unless you see clear confirmation the task is complete.
4. Respond with ONLY a JSON object containing your action.

Output format:
{{"action": "your_action_here", "rationale": "brief explanation"}}
"""

    def _build_user_message(
        self,
        screenshot: bytes,
        goal: str,
        action_history: Optional[List[str]] = None,
        valid_bids: Optional[Set[str]] = None,
        last_action_error: Optional[str] = None,
        axtree_summary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build user message content with image and text."""
        content = []

        # Add screenshot image
        content.append({
            "type": "image_url",
            "image_url": {"url": self._encode_image(screenshot)}
        })

        # Build text parts
        text_parts = [f"# Goal\n{goal}"]

        if last_action_error:
            text_parts.append(f"# ⚠️ LAST ACTION ERROR\n{last_action_error}")

        if action_history:
            recent = action_history[-5:]
            text_parts.append(f"# Recent Actions\n" + "\n".join(f"- {a}" for a in recent))

        if valid_bids:
            # Show available element IDs
            bids_str = ", ".join(sorted(valid_bids)[:50])
            text_parts.append(f"# Available Element IDs\n{bids_str}")

        if axtree_summary:
            # Optionally include AXTree summary for additional context
            text_parts.append(f"# Page Structure\n{axtree_summary[:2000]}")

        text_parts.append("# Your Next Action\nRespond with JSON only: {\"action\": \"...\", \"rationale\": \"...\"}")

        content.append({
            "type": "text",
            "text": "\n\n".join(text_parts)
        })

        return content

    def sample_actions(
        self,
        screenshot: bytes,
        goal: str,
        n: int = 8,
        temperature: float = 0.7,
        action_history: Optional[List[str]] = None,
        valid_bids: Optional[Set[str]] = None,
        last_action_error: Optional[str] = None,
        axtree_summary: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sample n actions from VLM given screenshot and goal.

        Used during training to get multiple action candidates for GRPO.

        Args:
            screenshot: PNG screenshot bytes
            goal: Task goal string
            n: Number of actions to sample
            temperature: Sampling temperature (higher = more diverse)
            action_history: List of previous actions in episode
            valid_bids: Set of valid element IDs on current page
            last_action_error: Error message from last action (if any)
            axtree_summary: Optional AXTree text summary

        Returns:
            List of dicts with keys: action, rationale, raw_response
        """
        self.total_requests += 1
        client = self._get_client()

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_message(
                    screenshot, goal, action_history, valid_bids,
                    last_action_error, axtree_summary
                )
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                n=n,
            )

            results = []
            for choice in response.choices:
                raw = choice.message.content.strip()
                parsed = self._parse_response(raw)
                action = parsed.get("action", "noop()")

                # Validate and repair action
                action, was_repaired = self._validate_action(action, valid_bids)
                if was_repaired:
                    self.invalid_action_count += 1

                results.append({
                    "action": action,
                    "rationale": parsed.get("rationale", ""),
                    "raw_response": raw,
                    "was_repaired": was_repaired,
                })

            return results

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"VLM sample_actions failed: {e}")
            # Return fallback actions
            return [{"action": "noop()", "rationale": f"Error: {e}", "raw_response": "", "was_repaired": True} for _ in range(n)]

    def sample_single(
        self,
        screenshot: bytes,
        goal: str,
        temperature: float = 0.0,
        action_history: Optional[List[str]] = None,
        valid_bids: Optional[Set[str]] = None,
        last_action_error: Optional[str] = None,
        axtree_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sample single action from VLM (for evaluation, greedy).

        Args:
            screenshot: PNG screenshot bytes
            goal: Task goal string
            temperature: Sampling temperature (0 for greedy/deterministic)
            action_history: List of previous actions in episode
            valid_bids: Set of valid element IDs on current page
            last_action_error: Error message from last action (if any)
            axtree_summary: Optional AXTree text summary

        Returns:
            Dict with keys: action, rationale, raw_response, was_repaired
        """
        results = self.sample_actions(
            screenshot=screenshot,
            goal=goal,
            n=1,
            temperature=temperature,
            action_history=action_history,
            valid_bids=valid_bids,
            last_action_error=last_action_error,
            axtree_summary=axtree_summary,
        )
        return results[0] if results else {"action": "noop()", "rationale": "No response", "raw_response": "", "was_repaired": True}

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from VLM."""
        try:
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            # Try direct JSON parse
            if response.startswith("{"):
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    return json.loads(json_match.group())

            return json.loads(response)

        except json.JSONDecodeError:
            # Fallback: try to extract action directly
            action_patterns = [
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

            for pattern in action_patterns:
                match = re.search(pattern, response)
                if match:
                    return {"action": match.group(1), "rationale": "Extracted from response"}

            logger.warning(f"Failed to parse VLM response: {response[:200]}")
            return {"action": "noop()", "rationale": "Failed to parse response"}

    def _validate_action(self, action: str, valid_bids: Optional[Set[str]] = None) -> tuple:
        """Validate action and repair if needed.

        Returns:
            Tuple of (action, was_repaired)
        """
        action = action.strip()

        # Check valid action prefixes
        valid_prefixes = [
            'click(', 'fill(', 'select_option(', 'scroll(',
            'go_back()', 'go_forward()', 'goto(', 'send_msg_to_user(',
            'noop()', 'hover(', 'press(', 'focus('
        ]

        has_valid_prefix = any(action.startswith(p) for p in valid_prefixes)
        if not has_valid_prefix:
            return "noop()", True

        # Extract and validate bid if present
        if valid_bids:
            bid_match = re.match(r'^(?:click|fill|select_option|hover|press|focus)\("([^"]+)"', action)
            if bid_match:
                bid = bid_match.group(1)
                if bid not in valid_bids:
                    logger.debug(f"Invalid bid {bid}, valid: {list(valid_bids)[:10]}")
                    return "scroll(0, 300)", True

        return action, False

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "invalid_action_count": self.invalid_action_count,
            "failure_rate": self.failed_requests / max(1, self.total_requests),
            "invalid_action_rate": self.invalid_action_count / max(1, self.total_requests),
        }

    def health_check(self) -> bool:
        """Check if vLLM server is available.

        Returns:
            True if server responds, False otherwise
        """
        try:
            client = self._get_client()
            models = client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"vLLM health check: available models = {available}")
            return self.model in available or len(available) > 0
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False


def create_vlm_client(
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    **kwargs
) -> VLLMPolicyClient:
    """Factory function to create VLM policy client.

    Args:
        base_url: vLLM API base URL
        model: Model name being served
        **kwargs: Additional arguments for VLLMPolicyClient

    Returns:
        Configured VLLMPolicyClient
    """
    return VLLMPolicyClient(base_url=base_url, model=model, **kwargs)
