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
