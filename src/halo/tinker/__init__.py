"""Tinker API integration for HALO-Agent online RL training.

Provides Tinker-compatible environment, group builder, and dataset
abstractions that plug into ``tinker_cookbook.rl.train.main()``.

Usage::

    from halo.tinker import BrowserEnv, BrowserGroupBuilder
    from halo.tinker import BrowserDataset, BrowserDatasetBuilder
"""

from .browser_env import BrowserEnv
from .group_builder import BrowserGroupBuilder
from .dataset import BrowserDataset, BrowserDatasetBuilder
from .action_parser import parse_action_from_text, validate_action

__all__ = [
    "BrowserEnv",
    "BrowserGroupBuilder",
    "BrowserDataset",
    "BrowserDatasetBuilder",
    "parse_action_from_text",
    "validate_action",
]
