"""Manager policy for HALO Agent.

Uses gpt-4o for strategic decisions on errors/loops/high-stakes.
"""

import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ManagerDecision:
    """Manager's strategic decision."""
    subgoal: str
    skill: str
    stop_condition: str
    risk: str  # 'low', 'med', 'high'
    reasoning: str = ""


class ManagerPolicy:
    """Strategic manager policy using gpt-4o."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def get_decision(
        self,
        obs_summary: str,
        goal: str,
        action_history: List[str],
        last_error: str = "",
        loop_detected: bool = False,
        page_type: str = "unknown"
    ) -> ManagerDecision:
        """Get strategic decision from manager.

        Called only when:
        - last_action_error is present
        - Loop detected
        - High-stakes page (checkout, submit)

        Args:
            obs_summary: Summarized observation
            goal: Task goal
            action_history: Previous actions
            last_error: Error message if any
            loop_detected: Whether a loop was detected
            page_type: Current page type

        Returns:
            ManagerDecision with subgoal, skill, stop_condition, risk
        """
        system_prompt = """You are a strategic manager for a browser automation agent.
Your worker has encountered a situation requiring your guidance.

Analyze the situation and provide:
1. subgoal: Immediate objective (1 sentence)
2. skill: Suggested skill/approach (search_and_filter, fill_form, navigate, retry, or custom)
3. stop_condition: When worker should stop this subgoal
4. risk: low, med, or high

Output ONLY valid JSON:
{"subgoal": "...", "skill": "...", "stop_condition": "...", "risk": "low|med|high", "reasoning": "..."}
"""

        user_parts = [f"# Goal\n{goal}", obs_summary]

        # Add context about why manager was called
        context_parts = []
        if last_error:
            context_parts.append(f"Error: {last_error}")
        if loop_detected:
            context_parts.append("Loop detected - same state repeated")
        if page_type in ['checkout', 'cart', 'login']:
            context_parts.append(f"High-stakes page: {page_type}")

        if context_parts:
            user_parts.append("# Why Manager Called\n" + "\n".join(context_parts))

        if action_history:
            recent = action_history[-10:]
            user_parts.append("# Recent Actions\n" + "\n".join(recent))

        user_parts.append("# Your Strategic Decision\nProvide JSON guidance for the worker.")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n\n".join(user_parts)}
                ],
                temperature=0.2,
                max_tokens=512,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return ManagerDecision(
                subgoal=result.get("subgoal", "Continue with task"),
                skill=result.get("skill", "navigate"),
                stop_condition=result.get("stop_condition", "Goal completed"),
                risk=result.get("risk", "low"),
                reasoning=result.get("reasoning", "")
            )

        except Exception as e:
            logger.error(f"Manager policy error: {e}")
            return ManagerDecision(
                subgoal="Recover from error and continue",
                skill="retry",
                stop_condition="Error resolved",
                risk="med",
                reasoning=f"Manager error: {e}"
            )


def create_manager_policy(model: str = "gpt-4o") -> ManagerPolicy:
    """Factory function for manager policy."""
    return ManagerPolicy(model=model)
