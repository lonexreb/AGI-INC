"""State fingerprinting for HALO Agent.

Builds stable StateKey hash for caching and loop detection.
"""

import hashlib
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass

from .probe import extract_signals, AXSignals


@dataclass(frozen=True)
class StateKey:
    """Immutable state key for caching."""
    site_id: str
    page_type: str
    url_normalized: str
    elements_signature: str
    has_error: bool

    def __hash__(self):
        return hash((self.site_id, self.page_type, self.url_normalized,
                     self.elements_signature, self.has_error))

    def to_string(self) -> str:
        """Convert to string for JSON serialization."""
        return f"{self.site_id}|{self.page_type}|{self.url_normalized}|{self.elements_signature}|{self.has_error}"

    @classmethod
    def from_string(cls, s: str) -> 'StateKey':
        """Create from string."""
        parts = s.split('|')
        return cls(
            site_id=parts[0],
            page_type=parts[1],
            url_normalized=parts[2],
            elements_signature=parts[3],
            has_error=parts[4] == 'True'
        )


def normalize_url(url: str) -> str:
    """Normalize URL for stable hashing.

    - Remove session IDs and tracking params
    - Sort query params
    - Normalize path
    """
    if not url:
        return ''

    parsed = urlparse(url)

    # Filter out common tracking/session params
    skip_params = {'sid', 'session', 'token', 'utm_source', 'utm_medium',
                   'utm_campaign', 'ref', 'fbclid', 'gclid', '_ga'}

    query = parse_qs(parsed.query)
    filtered_query = {k: v for k, v in query.items() if k.lower() not in skip_params}
    sorted_query = '&'.join(f"{k}={v[0]}" for k, v in sorted(filtered_query.items()))

    # Normalize path (remove trailing slash)
    path = parsed.path.rstrip('/')

    return f"{parsed.scheme}://{parsed.netloc}{path}?{sorted_query}".rstrip('?')


def extract_site_id(url: str) -> str:
    """Extract site identifier from URL."""
    if not url:
        return 'unknown'

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Common REAL benchmark sites
    site_patterns = {
        'omnizon': 'omnizon',
        'gomail': 'gomail',
        'dashdish': 'dashdish',
        'roundcube': 'roundcube',
        'gitlab': 'gitlab',
        'reddit': 'reddit',
        'shopping': 'shopping',
        'classifieds': 'classifieds',
        'maps': 'maps',
        'calendar': 'calendar',
    }

    for pattern, site_id in site_patterns.items():
        if pattern in host:
            return site_id

    return host.split('.')[0] if host else 'unknown'


def build_elements_signature(ax_signals: AXSignals) -> str:
    """Build signature from AX signals."""
    return f"{ax_signals.role_name_digest}:{ax_signals.visible_text_digest}"


def build_state_key(
    obs: Dict,
    task_info: Optional[Dict] = None,
    ax_signals: Optional[AXSignals] = None
) -> StateKey:
    """Build StateKey from observation.

    Args:
        obs: Observation dictionary
        task_info: Optional task metadata (site_id, task_type)
        ax_signals: Optional pre-computed AXSignals

    Returns:
        StateKey instance
    """
    url = obs.get('url', '')

    # Extract site ID
    site_id = 'unknown'
    if task_info and 'site_id' in task_info:
        site_id = task_info['site_id']
    else:
        site_id = extract_site_id(url)

    # Determine page type
    from .page_type import classify_page_type
    page_type = classify_page_type(url, obs.get('title', ''))

    # Normalize URL
    url_normalized = normalize_url(url)

    # Build elements signature
    if ax_signals is None:
        ax_signals = extract_signals(obs)
    elements_sig = build_elements_signature(ax_signals)

    # Check for error
    has_error = bool(obs.get('last_action_error', ''))

    return StateKey(
        site_id=site_id,
        page_type=page_type,
        url_normalized=url_normalized,
        elements_signature=elements_sig,
        has_error=has_error
    )


def state_key_hash(state_key: StateKey) -> str:
    """Get short hash of state key."""
    content = state_key.to_string()
    return hashlib.sha256(content.encode()).hexdigest()[:16]
