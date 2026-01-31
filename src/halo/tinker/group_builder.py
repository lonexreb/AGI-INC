"""GRPO group builder for browser RL training.

Creates groups of BrowserEnv instances for the same task so that
Tinker's training loop can compute group-relative advantages (GRPO).
"""

import logging
from typing import Any, Optional, Sequence

from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Env,
    Metrics,
    Trajectory,
)

from ..rl.progress import DenseRewardCalculator
from .browser_env import BrowserEnv, _ensure_v2_task

logger = logging.getLogger(__name__)


class BrowserGroupBuilder(EnvGroupBuilder):
    """Creates a group of BrowserEnv instances for GRPO training.

    All environments in the group run the same task (optionally with
    the same seed) so that group-relative advantage computation is
    meaningful — each trajectory starts from the same initial state.
    """

    def __init__(
        self,
        task_name: str,
        num_envs: int,
        renderer: Any,
        max_steps: int = 70,
        headless: bool = True,
        task_seed: Optional[int] = None,
        reward_config: Optional[dict] = None,
    ):
        self.task_name = _ensure_v2_task(task_name)
        self.num_envs = num_envs
        self.renderer = renderer
        self.max_steps = max_steps
        self.headless = headless
        self.task_seed = task_seed
        self.reward_config = reward_config or {}

    async def make_envs(self) -> Sequence[Env]:
        """Create N parallel BrowserEnv instances for the same task."""
        envs = []
        for _ in range(self.num_envs):
            reward_calc = DenseRewardCalculator(**self.reward_config)
            env = BrowserEnv(
                task_name=self.task_name,
                renderer=self.renderer,
                reward_calculator=reward_calc,
                max_steps=self.max_steps,
                headless=self.headless,
                task_seed=self.task_seed,
            )
            envs.append(env)
        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        """Compute additional group-level rewards after all rollouts complete.

        Per-step rewards are already captured in each Transition.reward
        (via DenseRewardCalculator inside BrowserEnv.step).
        Here we add episode-level bonuses visible to the whole group:
        - success_bonus: +1.0 if the trajectory completed the task
        """
        results = []
        for traj, env in zip(trajectory_group, env_group):
            group_reward = 0.0
            metrics: Metrics = {}

            # Check success from last transition
            if traj.transitions:
                last = traj.transitions[-1]
                if last.metrics.get("success", 0.0) > 0.5:
                    group_reward += 1.0
                    metrics["task_success"] = 1.0
                else:
                    metrics["task_success"] = 0.0

            results.append((group_reward, metrics))
        return results

    def logging_tags(self) -> list[str]:
        """Tags used by Tinker's logging to aggregate metrics."""
        # e.g. "v2.omnizon-1" → ["omnizon", "real"]
        parts = self.task_name.replace("v2.", "").split("-")
        site = parts[0] if parts else "unknown"
        return [site, "real"]
