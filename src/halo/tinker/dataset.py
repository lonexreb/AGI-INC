"""Dataset and builder for REAL benchmark RL training.

Provides the RLDataset / RLDatasetBuilder abstractions that Tinker's
training loop expects. Each batch maps to one task; the dataset cycles
through the configured task list.
"""

import logging
from typing import Any, Optional, Sequence

import chz
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

from .browser_env import _ensure_v2_task
from .group_builder import BrowserGroupBuilder

logger = logging.getLogger(__name__)


class BrowserDataset(RLDataset):
    """Dataset of REAL benchmark tasks for RL training.

    Each call to get_batch(index) returns a single-element list containing
    a BrowserGroupBuilder for one task. Tasks cycle round-robin.
    """

    def __init__(
        self,
        task_names: list[str],
        renderer: Any,
        num_envs_per_group: int = 8,
        max_steps: int = 70,
        headless: bool = True,
        reward_config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.task_names = [_ensure_v2_task(t) for t in task_names]
        self.renderer = renderer
        self.num_envs_per_group = num_envs_per_group
        self.max_steps = max_steps
        self.headless = headless
        self.reward_config = reward_config or {}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.task_names)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        task_name = self.task_names[index % len(self.task_names)]
        task_seed = (self.seed + index) if self.seed is not None else None

        builder = BrowserGroupBuilder(
            task_name=task_name,
            num_envs=self.num_envs_per_group,
            renderer=self.renderer,
            max_steps=self.max_steps,
            headless=self.headless,
            task_seed=task_seed,
            reward_config=self.reward_config,
        )
        return [builder]


@chz.chz
class BrowserDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optional test) BrowserDatasets.

    Designed to be passed as ``dataset_builder`` to
    ``tinker_cookbook.rl.train.Config``.
    """

    train_tasks: list[str]
    test_tasks: list[str] = chz.field(default_factory=list)
    renderer_name: str = "qwen3_instruct"
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    num_envs_per_group: int = 8
    max_steps: int = 70
    headless: bool = True
    seed: int = 42

    # Reward shaping weights
    progress_weight: float = 1.0
    novelty_bonus: float = 0.2
    loop_penalty: float = -0.5
    action_error_penalty: float = -0.2
    success_bonus: float = 1.0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)

        reward_config = {
            "progress_weight": self.progress_weight,
            "novelty_bonus": self.novelty_bonus,
            "loop_penalty": self.loop_penalty,
            "action_error_penalty": self.action_error_penalty,
            "success_bonus": self.success_bonus,
        }

        train_ds = BrowserDataset(
            task_names=self.train_tasks,
            renderer=renderer,
            num_envs_per_group=self.num_envs_per_group,
            max_steps=self.max_steps,
            headless=self.headless,
            reward_config=reward_config,
            seed=self.seed,
        )

        test_ds = None
        if self.test_tasks:
            test_ds = BrowserDataset(
                task_names=self.test_tasks,
                renderer=renderer,
                num_envs_per_group=1,  # single rollout per task for eval
                max_steps=self.max_steps,
                headless=self.headless,
                reward_config=reward_config,
                seed=self.seed + 10000,
            )

        return train_ds, test_ds
