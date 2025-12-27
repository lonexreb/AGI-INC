"""Trajectory logger for HALO Agent.

JSONL logging for trajectory collection and analysis.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Record for a single step."""
    task_id: str
    task_name: str
    step_idx: int
    timestamp: str
    url: str
    obs_hash: str
    action: str
    action_source: str  # 'worker', 'manager', 'cache', 'macro'
    last_action_error: str
    elapsed_ms: float
    manager_called: bool
    cache_hit: bool
    mode: str  # 'baseline', 'halo', 'halo_cache'
    obs_summary: str = ""
    actionable_elements: Optional[List[Dict[str, Any]]] = None
    valid_bids: Optional[List[str]] = None
    progress_score: float = 0.0
    milestones: Optional[List[str]] = None


@dataclass
class EpisodeRecord:
    """Record for episode summary."""
    task_id: str
    task_name: str
    success: bool
    total_steps: int
    wall_time_sec: float
    manager_calls: int
    cache_hits: int
    cache_misses: int
    final_url: str
    final_message: str
    mode: str
    error_count: int
    timestamp: str
    run_id: str = ""
    attempt_idx: int = 0
    reward: float = 0.0
    max_progress_score: float = 0.0
    final_progress_score: float = 0.0
    error: str = ""


class TrajectoryLogger:
    """JSONL trajectory logger."""

    def __init__(
        self,
        run_id: str,
        output_dir: str = "data/trajectories",
        mode: str = "baseline"
    ):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._episode_fh = None
        self._episode_path: Optional[Path] = None
        self._attempt_idx: int = 0
        self._last_url: str = ""

        # Current episode state
        self.current_task_id: Optional[str] = None
        self.current_task_name: Optional[str] = None
        self.episode_start_time: float = 0
        self.step_records: List[StepRecord] = []
        self.manager_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.error_count: int = 0

        self.run_metadata: Dict[str, Any] = {}

    def start_episode(
        self,
        task_id: str,
        task_name: str,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        attempt_idx: Optional[int] = None,
    ):
        """Start logging a new episode.

        Args:
            task_id: Task identifier
            task_name: Human-readable task name
        """
        # Close any previously open episode file handle.
        try:
            if self._episode_fh is not None:
                self._episode_fh.close()
        except Exception:
            pass

        self.current_task_id = task_id
        self.current_task_name = task_name
        self.episode_start_time = time.time()
        self.step_records = []
        self.manager_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0

        if run_id is not None and run_id != self.run_id:
            self.run_id = run_id
        if mode is not None and mode != self.mode:
            self.mode = mode

        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if attempt_idx is None:
            # Numeric and very unlikely to collide across processes.
            attempt_idx = int(time.time() * 1_000_000)
        self._attempt_idx = int(attempt_idx)

        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)

        while True:
            candidate = self.run_dir / f"{safe_task}__attempt_{self._attempt_idx}.jsonl"
            try:
                self._episode_fh = open(candidate, "x", encoding="utf-8")
                self._episode_path = candidate
                break
            except FileExistsError:
                self._attempt_idx += 1

        self._last_url = ""

        record: Dict[str, Any] = {
            "type": "episode_start",
            "task_id": self.current_task_id or "",
            "task_name": self.current_task_name or "",
            "run_id": self.run_id,
            "mode": self.mode,
            "attempt_idx": self._attempt_idx,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
        }

        if self.run_metadata:
            for k, v in self.run_metadata.items():
                if k not in record:
                    record[k] = v

        self._write_record(record)

    def log_step(
        self,
        step_idx: int,
        url: str,
        obs_hash: str,
        action: str,
        action_source: str,
        last_action_error: str = "",
        elapsed_ms: float = 0,
        manager_called: bool = False,
        cache_hit: bool = False,
        obs_summary: str = "",
        actionable_elements: Optional[List[Dict[str, Any]]] = None,
        valid_bids: Optional[List[str]] = None,
        progress_score: float = 0.0,
        milestones: Optional[List[str]] = None,
    ):
        """Log a single step.

        Args:
            step_idx: Step number
            url: Current URL
            obs_hash: Hash of observation
            action: Action taken
            action_source: Source of action decision
            last_action_error: Error from previous action
            elapsed_ms: Time for this step in milliseconds
            manager_called: Whether manager was consulted
            cache_hit: Whether action came from cache
        """
        record = StepRecord(
            task_id=self.current_task_id or "",
            task_name=self.current_task_name or "",
            step_idx=step_idx,
            timestamp=datetime.now().isoformat(),
            url=url,
            obs_hash=obs_hash,
            action=action,
            action_source=action_source,
            last_action_error=last_action_error,
            elapsed_ms=elapsed_ms,
            manager_called=manager_called,
            cache_hit=cache_hit,
            mode=self.mode,
            obs_summary=obs_summary,
            actionable_elements=actionable_elements,
            valid_bids=valid_bids,
            progress_score=progress_score,
            milestones=milestones,
        )

        self.step_records.append(record)

        if manager_called:
            self.manager_calls += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if last_action_error:
            self.error_count += 1

        self._last_url = url or self._last_url

        # Write step to file
        self._write_record({"type": "step", **asdict(record)})

    def end_episode(
        self,
        success: bool,
        final_url: str = "",
        final_message: str = "",
        reward: float = 0.0,
        error: str = "",
        max_progress_score: Optional[float] = None,
        final_progress_score: Optional[float] = None,
    ) -> EpisodeRecord:
        """End current episode and write summary.

        Args:
            success: Whether task was completed successfully
            final_url: Final URL at end of episode
            final_message: Final message sent to user

        Returns:
            EpisodeRecord summary
        """
        wall_time = time.time() - self.episode_start_time

        if max_progress_score is None:
            max_progress_score = max(
                (getattr(step, "progress_score", 0.0) for step in self.step_records),
                default=0.0,
            )
        if final_progress_score is None:
            final_progress_score = (
                getattr(self.step_records[-1], "progress_score", 0.0) if self.step_records else 0.0
            )

        record = EpisodeRecord(
            task_id=self.current_task_id or "",
            task_name=self.current_task_name or "",
            success=success,
            total_steps=len(self.step_records),
            wall_time_sec=wall_time,
            manager_calls=self.manager_calls,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            final_url=final_url or self._last_url,
            final_message=final_message,
            mode=self.mode,
            error_count=self.error_count,
            timestamp=datetime.now().isoformat(),
            run_id=self.run_id,
            attempt_idx=self._attempt_idx,
            reward=reward,
            max_progress_score=max_progress_score,
            final_progress_score=final_progress_score,
            error=error,
        )

        self._write_record({"type": "episode_end", **asdict(record)})

        try:
            if self._episode_fh is not None:
                self._episode_fh.close()
        finally:
            self._episode_fh = None
            self._episode_path = None

        return record

    def _write_record(self, record: Dict[str, Any]):
        """Write a record to the JSONL file."""
        try:
            if self._episode_fh is None:
                return
            self._episode_fh.write(json.dumps(record) + '\n')
            self._episode_fh.flush()
        except Exception as e:
            logger.error(f"Failed to write trajectory record: {e}")


def load_trajectories(file_path: str) -> List[Dict[str, Any]]:
    """Load trajectories from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of records
    """
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_episode_summaries(file_path: str) -> List[EpisodeRecord]:
    """Get only episode summaries from trajectory file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of EpisodeRecord objects
    """
    records = load_trajectories(file_path)
    episodes = []
    for r in records:
        if r.get('type') in {'episode', 'episode_end'}:
            # Remove 'type' key before creating dataclass
            data = {k: v for k, v in r.items() if k != 'type'}
            episodes.append(EpisodeRecord(**data))
    return episodes
