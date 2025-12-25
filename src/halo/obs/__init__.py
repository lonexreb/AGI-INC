"""Observation summarizer and fingerprinting for HALO Agent."""

from .obs_summarizer import summarize_observation, extract_actionable_nodes, get_obs_hash
from .fingerprint import StateKey, build_state_key, state_key_hash
from .page_type import (
    classify_page_type, is_high_stakes_page, is_form_page,
    has_confirmation_signal, extract_confirmation_text, CONFIRMATION_SIGNALS
)

__all__ = [
    'summarize_observation',
    'extract_actionable_nodes',
    'get_obs_hash',
    'StateKey',
    'build_state_key',
    'state_key_hash',
    'classify_page_type',
    'is_high_stakes_page',
    'is_form_page',
    'has_confirmation_signal',
    'extract_confirmation_text',
    'CONFIRMATION_SIGNALS',
]
