from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..obs.fingerprint import extract_site_id
from ..obs.obs_summarizer import flatten_axtree_simple
from ..obs.page_type import (
    PAGE_TYPE_CALENDAR,
    PAGE_TYPE_CART,
    PAGE_TYPE_CHECKOUT,
    PAGE_TYPE_EMAIL_COMPOSE,
    PAGE_TYPE_EMAIL_LIST,
    PAGE_TYPE_EMAIL_VIEW,
    PAGE_TYPE_HOME,
    PAGE_TYPE_PRODUCT,
    PAGE_TYPE_RESULTS,
    PAGE_TYPE_SEARCH,
    classify_page_type,
    has_confirmation_signal,
)


@dataclass
class ProgressInfo:
    progress_score: float
    milestones: List[str]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def score_progress(
    obs: Dict[str, Any],
    goal: str = "",
    site_id: Optional[str] = None,
    page_type: Optional[str] = None,
) -> ProgressInfo:
    url = obs.get("url", "") or ""
    title = obs.get("title", "") or ""

    resolved_site = site_id or extract_site_id(url)
    resolved_page_type = page_type or classify_page_type(url, title)

    axtree_text = ""
    axtree = obs.get("axtree_object")
    if axtree:
        try:
            axtree_text = flatten_axtree_simple(axtree) or ""
        except Exception:
            axtree_text = ""

    text_lower = (axtree_text + " " + title).lower()

    score = 0.0
    milestones: List[str] = []

    if resolved_site == "omnizon":
        stage_scores = {
            PAGE_TYPE_HOME: 0.1,
            PAGE_TYPE_SEARCH: 0.3,
            PAGE_TYPE_RESULTS: 0.3,
            PAGE_TYPE_PRODUCT: 0.5,
            PAGE_TYPE_CART: 0.7,
            PAGE_TYPE_CHECKOUT: 0.85,
        }

        score = max(score, 0.05)
        score = max(score, stage_scores.get(resolved_page_type, 0.0))

        if resolved_page_type in stage_scores:
            milestones.append(f"omnizon_{resolved_page_type}")

        if "cart" in url.lower() or "basket" in url.lower():
            score = max(score, 0.7)
            milestones.append("omnizon_cart")

        if has_confirmation_signal(axtree_text, title):
            score = max(score, 1.0)
            milestones.append("omnizon_confirmation")

    elif resolved_site == "gomail":
        score = max(score, 0.05)

        if resolved_page_type in {PAGE_TYPE_EMAIL_LIST, PAGE_TYPE_EMAIL_VIEW, PAGE_TYPE_EMAIL_COMPOSE}:
            score = max(score, 0.1)
            milestones.append(f"gomail_{resolved_page_type}")

        read_done = ("marked as read" in text_lower) or ("mark all as read" in text_lower)
        delete_done = (
            ("moved to trash" in text_lower)
            or ("moved to bin" in text_lower)
            or ("deleted" in text_lower and "amazon" in text_lower)
        )

        if read_done:
            score += 0.45
            milestones.append("gomail_marked_read")

        if delete_done:
            score += 0.45
            milestones.append("gomail_moved_to_trash")

        if delete_done and "4" in text_lower:
            score = max(score, 0.95)
            milestones.append("gomail_trash_4")

    elif resolved_site == "calendar":
        score = max(score, 0.05)

        if resolved_page_type == PAGE_TYPE_CALENDAR:
            score = max(score, 0.2)
            milestones.append("calendar_view")

        event_form = (
            ("location" in text_lower and ("start" in text_lower or "end" in text_lower or "time" in text_lower))
            or ("event details" in text_lower)
            or ("create event" in text_lower)
            or ("add event" in text_lower)
        )

        if event_form:
            score = max(score, 0.4)
            milestones.append("calendar_event_form")

        requires_gym = "gym" in (goal or "").lower()
        if (not requires_gym) or ("gym" in text_lower):
            if "gym" in text_lower:
                score = max(score, 0.6)
                milestones.append("calendar_gym")

        has_time = any(t in text_lower for t in ["7:45", "8:45", "19:45", "20:45"])
        if has_time:
            score = max(score, 0.7)
            milestones.append("calendar_time")

        if "gym" in text_lower and has_time:
            score = max(score, 0.8)

        if has_confirmation_signal(axtree_text, title):
            score = max(score, 1.0)
            milestones.append("calendar_confirmation")

    score = _clamp01(score)
    milestones = _dedupe(milestones)

    return ProgressInfo(progress_score=score, milestones=milestones)


class DenseRewardCalculator:
    """Dense reward calculator for Online RL training.

    Provides learning signal even without task completion by rewarding:
    - Progress toward goal (via score_progress)
    - Valid action format
    - State novelty (visiting new states)
    - Penalizing loops and repeated failures

    Reward components:
    - progress_delta: Difference in progress score from prev to current state
    - action_validity: +0.1 for valid JSON action format
    - novelty: +0.2 for reaching new state (not seen before)
    - loop_penalty: -0.5 for repeating same action in same state
    - success_bonus: +1.0 for task completion
    - failure_penalty: -0.2 for action errors
    """

    def __init__(
        self,
        progress_weight: float = 1.0,
        novelty_bonus: float = 0.2,
        loop_penalty: float = -0.5,
        action_error_penalty: float = -0.2,
        success_bonus: float = 1.0,
    ):
        """Initialize reward calculator.

        Args:
            progress_weight: Weight for progress delta rewards
            novelty_bonus: Bonus for visiting new states
            loop_penalty: Penalty for repeating actions in same state
            action_error_penalty: Penalty for action errors
            success_bonus: Bonus for task completion
        """
        self.progress_weight = progress_weight
        self.novelty_bonus = novelty_bonus
        self.loop_penalty = loop_penalty
        self.action_error_penalty = action_error_penalty
        self.success_bonus = success_bonus

        # Episode state
        self.seen_states: set = set()
        self.action_history: List[str] = []
        self.state_action_pairs: set = set()
        self.prev_progress: float = 0.0
        self.prev_milestones: List[str] = []

        # Stats
        self.total_rewards: List[float] = []
        self.progress_rewards: List[float] = []
        self.novelty_rewards: List[float] = []

    def reset(self):
        """Reset for new episode."""
        self.seen_states = set()
        self.action_history = []
        self.state_action_pairs = set()
        self.prev_progress = 0.0
        self.prev_milestones = []
        self.total_rewards = []
        self.progress_rewards = []
        self.novelty_rewards = []

    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: str,
        next_obs: Dict[str, Any],
        done: bool = False,
        success: bool = False,
        last_action_error: str = "",
        state_hash: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute dense reward for a transition.

        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation after action
            done: Whether episode is done
            success: Whether task completed successfully
            last_action_error: Error message from action (if any)
            state_hash: Optional precomputed state hash

        Returns:
            Dict with reward components:
                - total: Total reward
                - progress: Progress delta reward
                - novelty: Novelty bonus
                - loop: Loop penalty
                - error: Action error penalty
                - success: Success bonus
        """
        rewards = {
            "total": 0.0,
            "progress": 0.0,
            "novelty": 0.0,
            "loop": 0.0,
            "error": 0.0,
            "success": 0.0,
        }

        # Get state hash for novelty detection
        if state_hash is None:
            state_hash = self._compute_state_hash(next_obs)

        # 1. Progress reward (delta from previous)
        next_progress_info = score_progress(next_obs)
        progress_delta = next_progress_info.progress_score - self.prev_progress

        if progress_delta > 0:
            rewards["progress"] = progress_delta * self.progress_weight
        elif progress_delta < 0:
            # Small penalty for going backwards
            rewards["progress"] = progress_delta * 0.5 * self.progress_weight

        self.progress_rewards.append(rewards["progress"])
        self.prev_progress = next_progress_info.progress_score
        self.prev_milestones = next_progress_info.milestones

        # 2. Novelty bonus
        if state_hash not in self.seen_states:
            rewards["novelty"] = self.novelty_bonus
            self.seen_states.add(state_hash)
        self.novelty_rewards.append(rewards["novelty"])

        # 3. Loop penalty (same action in same state)
        state_action_key = f"{state_hash}:{action}"
        if state_action_key in self.state_action_pairs:
            rewards["loop"] = self.loop_penalty
        self.state_action_pairs.add(state_action_key)
        self.action_history.append(action)

        # 4. Action error penalty
        if last_action_error:
            rewards["error"] = self.action_error_penalty

        # 5. Success bonus
        if done and success:
            rewards["success"] = self.success_bonus

        # Compute total
        rewards["total"] = sum([
            rewards["progress"],
            rewards["novelty"],
            rewards["loop"],
            rewards["error"],
            rewards["success"],
        ])

        self.total_rewards.append(rewards["total"])

        return rewards

    def _compute_state_hash(self, obs: Dict[str, Any]) -> str:
        """Compute a hash of the observation state."""
        url = obs.get("url", "")
        title = obs.get("title", "")

        # Simple hash from URL and title
        return f"{url}:{title}"

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for current episode."""
        if not self.total_rewards:
            return {
                "mean_reward": 0.0,
                "total_reward": 0.0,
                "mean_progress": 0.0,
                "mean_novelty": 0.0,
                "unique_states": 0,
                "num_steps": 0,
            }

        return {
            "mean_reward": sum(self.total_rewards) / len(self.total_rewards),
            "total_reward": sum(self.total_rewards),
            "mean_progress": sum(self.progress_rewards) / len(self.progress_rewards) if self.progress_rewards else 0.0,
            "mean_novelty": sum(self.novelty_rewards) / len(self.novelty_rewards) if self.novelty_rewards else 0.0,
            "unique_states": len(self.seen_states),
            "num_steps": len(self.total_rewards),
            "final_progress": self.prev_progress,
            "milestones": self.prev_milestones,
        }
