"""Tinker-compatible browser environment wrapping the REAL benchmark.

This module implements Tinker's Env ABC so that the REAL benchmark browser
can be driven by Tinker's RL training loop. Each BrowserEnv instance manages
a single Playwright browser session for one episode.

Architecture:
    Tinker's rollout system (async) calls:
        1. initial_observation() -> (ModelInput, StopCondition)
        2. step(action_tokens)   -> StepResult    (in a loop)
    We bridge to REAL's synchronous Playwright browser via ThreadPoolExecutor.
"""

import asyncio
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Set

import tinker
from tinker_cookbook.rl.types import Env, StepResult
from tinker_cookbook.completers import StopCondition

from ..constants import DEFAULT_MODEL
from ..obs import extract_actionable_nodes, summarize_observation
from ..rl.progress import DenseRewardCalculator, score_progress
from .action_parser import (
    ACTION_GRAMMAR,
    parse_action_from_text,
    validate_action,
)

logger = logging.getLogger(__name__)

# Shared thread pool for blocking browser operations
_browser_executor = ThreadPoolExecutor(max_workers=8)

# System prompt for the VLM
SYSTEM_PROMPT = f"""You are a browser automation agent controlling a web browser.
You see a screenshot of the current page and must decide what action to take.

{ACTION_GRAMMAR}

CRITICAL RULES:
1. ONLY use element IDs (bid values) that appear in the Available Element IDs list.
2. NEVER invent element IDs. If unsure, use scroll(0, 300) to see more content.
3. Do NOT call send_msg_to_user() unless you see clear confirmation the task is complete.
4. Respond with ONLY a JSON object containing your action.

Output format:
{{"action": "your_action_here", "rationale": "brief explanation"}}
"""


def _ensure_v2_task(task_name: str) -> str:
    """Ensure task name has v2. prefix."""
    if not task_name.startswith("v2."):
        return f"v2.{task_name}"
    return task_name


class BrowserEnv(Env):
    """Tinker-compatible wrapper around a REAL benchmark browser environment.

    Each instance owns one Playwright browser session. Created by
    BrowserGroupBuilder.make_envs() and discarded after one episode.

    The environment encodes observations as multimodal ModelInput
    (screenshot image + text context) and decodes model output tokens
    back into browser action strings.
    """

    def __init__(
        self,
        task_name: str,
        renderer: Any,  # tinker_cookbook.renderers.Renderer (Qwen3VL)
        reward_calculator: DenseRewardCalculator,
        max_steps: int = 70,
        headless: bool = True,
        task_seed: Optional[int] = None,
    ):
        self.task_name = _ensure_v2_task(task_name)
        self.renderer = renderer
        self.reward_calculator = reward_calculator
        self.max_steps = max_steps
        self.headless = headless
        self.task_seed = task_seed

        # State (initialised in initial_observation)
        self._gym_env = None
        self._obs: Optional[Dict] = None
        self._goal: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._success: bool = False
        self._action_history: List[str] = []

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    async def initial_observation(self):
        """Create browser env, reset, and return first (ModelInput, StopCondition)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_browser_executor, self._sync_initial_obs)

    async def step(self, action: list[int]) -> StepResult:
        """Decode action tokens, execute in browser, return StepResult."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _browser_executor, partial(self._sync_step, action)
        )

    # ------------------------------------------------------------------
    # Synchronous implementations (run inside thread pool)
    # ------------------------------------------------------------------

    def _sync_initial_obs(self):
        """Blocking: create gymnasium env, reset, build ModelInput."""
        self._gym_env = self._create_gym_env()
        obs, info = self._gym_env.reset(seed=self.task_seed)

        self._obs = obs
        self._goal = self._extract_goal(obs)
        self._step_count = 0
        self._done = False
        self._success = False
        self._action_history = []
        self.reward_calculator.reset()

        model_input = self._obs_to_model_input(obs)
        stop_condition = self.renderer.get_stop_sequences()
        return model_input, stop_condition

    def _sync_step(self, action_tokens: list[int]) -> StepResult:
        """Blocking: decode tokens → action string → browser step → StepResult."""
        self._step_count += 1

        # 1. Decode tokens to text
        response_message, _parse_ok = self.renderer.parse_response(action_tokens)
        action_text = response_message.get("content", "") if isinstance(response_message, dict) else str(response_message)

        # 2. Parse action string
        action_str = parse_action_from_text(action_text)

        # 3. Validate against current valid BIDs
        valid_bids = self._extract_valid_bids(self._obs)
        action_str, was_repaired = validate_action(action_str, valid_bids)

        # 4. Execute in browser
        try:
            next_obs, _env_reward, terminated, truncated, info = self._gym_env.step(action_str)
        except Exception as e:
            logger.warning(f"Browser step failed: {e}")
            next_obs = self._obs  # keep same obs on error
            terminated = False
            truncated = False
            info = {"error": str(e)}

        done = terminated or truncated or self._step_count >= self.max_steps
        self._success = info.get("success", False)
        last_action_error = next_obs.get("last_action_error", "") if isinstance(next_obs, dict) else ""

        # 5. Compute dense reward
        reward_components = self.reward_calculator.compute_reward(
            obs=self._obs,
            action=action_str,
            next_obs=next_obs,
            done=done,
            success=self._success,
            last_action_error=last_action_error,
        )
        step_reward = reward_components["total"]

        # Small bonus for valid (non-repaired) action format
        if not was_repaired:
            step_reward += 0.1

        # 6. Update state for next step
        self._obs = next_obs
        self._action_history.append(action_str)
        self._done = done

        # 7. Build next observation
        if not done:
            next_model_input = self._obs_to_model_input(next_obs)
        else:
            next_model_input = tinker.ModelInput.empty()

        next_stop_condition = self.renderer.get_stop_sequences()

        metrics = {
            "progress": reward_components.get("progress", 0.0),
            "was_repaired": float(was_repaired),
            "success": float(self._success),
        }

        return StepResult(
            reward=step_reward,
            episode_done=done,
            next_observation=next_model_input,
            next_stop_condition=next_stop_condition,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_gym_env(self):
        """Create a BrowserGym environment for the task."""
        from agisdk.REAL import browsergym

        env = browsergym.make(
            task_name=self.task_name,
            headless=self.headless,
            use_axtree=True,
            use_screenshot=True,
            use_html=False,
            viewport={"width": 1280, "height": 720},
        )
        return env

    def _obs_to_model_input(self, obs: Dict) -> tinker.ModelInput:
        """Convert a REAL benchmark observation to a Tinker ModelInput.

        Builds a multimodal ModelInput with:
        1. Text tokens (system prompt + user context) via the renderer
        2. An ImageChunk for the screenshot (inserted between system and user text)

        The renderer only handles text Messages, so we construct the
        ModelInput manually to interleave image chunks.
        """
        screenshot_bytes = self._get_screenshot_bytes(obs)

        # Build compact text context
        text_parts = [f"# Goal\n{self._goal}"]

        last_action_error = obs.get("last_action_error", "")
        if last_action_error:
            text_parts.append(f"# Last Action Error\n{last_action_error}")

        if self._action_history:
            recent = self._action_history[-5:]
            text_parts.append("# Recent Actions\n" + "\n".join(f"- {a}" for a in recent))

        valid_bids = self._extract_valid_bids(obs)
        if valid_bids:
            bids_str = ", ".join(sorted(valid_bids)[:50])
            text_parts.append(f"# Available Element IDs\n{bids_str}")

        # Optional AXTree summary for extra structure
        obs_summary = summarize_observation(obs, self._goal)
        if obs_summary:
            # Include a compact snippet
            text_parts.append(f"# Page Structure\n{obs_summary[:2000]}")

        text_parts.append('# Your Next Action\nRespond with JSON only: {"action": "...", "rationale": "..."}')
        user_text = "\n\n".join(text_parts)

        # Build text-only messages for the renderer to tokenize
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        text_model_input = self.renderer.build_generation_prompt(messages)

        # If we have a screenshot, insert an ImageChunk after the system
        # prompt tokens and before the user text tokens.
        if screenshot_bytes:
            # Qwen3-VL uses ~1176 tokens for a 1280x720 image
            image_chunk = tinker.types.ImageChunk(
                data=screenshot_bytes,
                format="png",
                expected_tokens=1176,
            )
            # Insert image chunk after first text chunk (system prompt)
            new_chunks = []
            inserted = False
            for chunk in text_model_input.chunks:
                new_chunks.append(chunk)
                if not inserted:
                    new_chunks.append(image_chunk)
                    inserted = True
            if not inserted:
                # Fallback: prepend image
                new_chunks = [image_chunk] + new_chunks
            return tinker.ModelInput(chunks=new_chunks)

        return text_model_input

    def _extract_goal(self, obs: Dict) -> str:
        """Extract task goal text from observation."""
        goal_obj = obs.get("goal_object", [])
        if isinstance(goal_obj, list):
            return " ".join(
                item.get("text", "")
                for item in goal_obj
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(goal_obj)

    def _get_screenshot_bytes(self, obs: Dict) -> bytes:
        """Convert observation screenshot to PNG bytes."""
        screenshot = obs.get("screenshot")
        if screenshot is None:
            return b""
        if isinstance(screenshot, bytes):
            return screenshot
        # numpy array → PNG bytes
        try:
            from PIL import Image

            img = Image.fromarray(screenshot)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"Screenshot conversion failed: {e}")
            return b""

    def _extract_valid_bids(self, obs: Optional[Dict]) -> Set[str]:
        """Extract valid element IDs from the accessibility tree."""
        if obs is None:
            return set()
        axtree = obs.get("axtree_object")
        if axtree:
            elements = extract_actionable_nodes(axtree, top_k=50)
            return {str(e["bid"]) for e in elements}
        return set()

    def __del__(self):
        """Clean up browser resources."""
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
