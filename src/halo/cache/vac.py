"""Verified Action Cache (VAC) for HALO Agent.

File-backed cache that stores state->action mappings after verification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry."""
    action: str
    postcondition: str
    success_count: int = 0
    fail_count: int = 0
    last_used: str = ""


class VerifiedActionCache:
    """Verified Action Cache with file persistence.

    Only caches actions after verifier confirms postcondition success.
    Evicts entries after 2 consecutive failures.
    """

    def __init__(self, cache_path: str = "data/cache/vac.json"):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CacheEntry] = {}
        self._load()

        # Stats
        self.hits = 0
        self.misses = 0

    def _load(self):
        """Load cache from file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    for key, entry in data.items():
                        self._cache[key] = CacheEntry(**entry)
                logger.info(f"Loaded {len(self._cache)} entries from VAC")
            except Exception as e:
                logger.error(f"Failed to load VAC: {e}")
                self._cache = {}

    def _save(self):
        """Save cache to file."""
        try:
            data = {k: asdict(v) for k, v in self._cache.items()}
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save VAC: {e}")

    def get(self, state_key: str) -> Optional[str]:
        """Get cached action for state key.

        Args:
            state_key: State key string

        Returns:
            Cached action or None if not found
        """
        if state_key in self._cache:
            entry = self._cache[state_key]
            # Don't return if too many failures
            if entry.fail_count >= 2:
                self.misses += 1
                return None
            self.hits += 1
            entry.last_used = datetime.now().isoformat()
            return entry.action

        self.misses += 1
        return None

    def put(
        self,
        state_key: str,
        action: str,
        postcondition: str,
        verified: bool = True
    ):
        """Store action in cache after verification.

        Args:
            state_key: State key string
            action: Action that was taken
            postcondition: Expected postcondition description
            verified: Whether the action was verified successful
        """
        if not verified:
            # Record failure
            if state_key in self._cache:
                self._cache[state_key].fail_count += 1
                # Evict if too many failures
                if self._cache[state_key].fail_count >= 2:
                    logger.info(f"Evicting cache entry due to failures: {state_key[:50]}")
                    del self._cache[state_key]
            return

        # Successful verification - add or update entry
        if state_key in self._cache:
            entry = self._cache[state_key]
            entry.success_count += 1
            entry.fail_count = 0  # Reset on success
            entry.last_used = datetime.now().isoformat()
        else:
            self._cache[state_key] = CacheEntry(
                action=action,
                postcondition=postcondition,
                success_count=1,
                fail_count=0,
                last_used=datetime.now().isoformat()
            )

        self._save()

    def record_failure(self, state_key: str):
        """Record a failure for a cached action."""
        if state_key in self._cache:
            self._cache[state_key].fail_count += 1
            if self._cache[state_key].fail_count >= 2:
                logger.info(f"Evicting cache entry: {state_key[:50]}")
                del self._cache[state_key]
            self._save()

    def clear(self):
        """Clear all cache entries."""
        self._cache = {}
        self._save()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

    def reset_stats(self):
        """Reset hit/miss counters."""
        self.hits = 0
        self.misses = 0
