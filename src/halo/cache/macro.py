"""Macro Replay Cache for HALO Agent.

Provides reusable action sequences (skills) for common patterns.
"""

import logging
from typing import Dict, List, Optional, Generator, Tuple, Any
from dataclasses import dataclass

from ..obs.page_type import (
    PAGE_TYPE_SEARCH, PAGE_TYPE_RESULTS, PAGE_TYPE_FORM,
    PAGE_TYPE_EMAIL_COMPOSE, PAGE_TYPE_CALENDAR
)

logger = logging.getLogger(__name__)


@dataclass
class MacroStep:
    """Single step in a macro."""
    action_template: str  # Action with placeholders like {bid}
    postcondition: str    # Expected result description
    element_role: str     # Required element role to find
    element_hint: str     # Hint for finding element (name contains, etc.)


@dataclass
class MacroDefinition:
    """Complete macro definition."""
    name: str
    description: str
    page_types: List[str]  # Page types where this macro applies
    steps: List[MacroStep]


# Predefined macros
MACROS: Dict[str, MacroDefinition] = {
    "search_and_open_first_result": MacroDefinition(
        name="search_and_open_first_result",
        description="Search for a term and click the first result",
        page_types=[PAGE_TYPE_SEARCH, PAGE_TYPE_RESULTS, "home", "unknown"],
        steps=[
            MacroStep(
                action_template='fill("{searchbox_bid}", "{search_term}")',
                postcondition="Search box filled",
                element_role="searchbox",
                element_hint="search"
            ),
            MacroStep(
                action_template='click("{submit_bid}")',
                postcondition="Search submitted",
                element_role="button",
                element_hint="search"
            ),
            MacroStep(
                action_template='click("{first_result_bid}")',
                postcondition="First result clicked",
                element_role="link",
                element_hint=""  # First link in results
            ),
        ]
    ),
    "fill_contact_form": MacroDefinition(
        name="fill_contact_form",
        description="Fill a contact form with standard fields",
        page_types=[PAGE_TYPE_FORM, "unknown"],
        steps=[
            MacroStep(
                action_template='fill("{email_bid}", "{email}")',
                postcondition="Email filled",
                element_role="textbox",
                element_hint="email"
            ),
            MacroStep(
                action_template='fill("{name_bid}", "{name}")',
                postcondition="Name filled",
                element_role="textbox",
                element_hint="name"
            ),
            MacroStep(
                action_template='fill("{phone_bid}", "{phone}")',
                postcondition="Phone filled",
                element_role="textbox",
                element_hint="phone"
            ),
            MacroStep(
                action_template='click("{submit_bid}")',
                postcondition="Form submitted",
                element_role="button",
                element_hint="submit"
            ),
        ]
    ),
    "select_date_range": MacroDefinition(
        name="select_date_range",
        description="Select start and end dates in a date picker",
        page_types=[PAGE_TYPE_CALENDAR, PAGE_TYPE_FORM, "unknown"],
        steps=[
            MacroStep(
                action_template='click("{start_date_bid}")',
                postcondition="Start date picker opened",
                element_role="textbox",
                element_hint="start"
            ),
            MacroStep(
                action_template='click("{start_day_bid}")',
                postcondition="Start date selected",
                element_role="button",
                element_hint=""
            ),
            MacroStep(
                action_template='click("{end_date_bid}")',
                postcondition="End date picker opened",
                element_role="textbox",
                element_hint="end"
            ),
            MacroStep(
                action_template='click("{end_day_bid}")',
                postcondition="End date selected",
                element_role="button",
                element_hint=""
            ),
        ]
    ),
}


class MacroReplayCache:
    """Manages macro replay for common action sequences."""

    def __init__(self):
        self.macros = MACROS
        self.active_macro: Optional[str] = None
        self.active_step: int = 0
        self.macro_params: Dict[str, str] = {}

        # Stats
        self.macro_attempts = 0
        self.macro_successes = 0

    def find_matching_macro(
        self,
        page_type: str,
        elements: List[Dict],
        goal_hints: List[str] = None
    ) -> Optional[str]:
        """Find a macro that matches current state.

        Args:
            page_type: Current page type
            elements: Available elements on page
            goal_hints: Keywords from goal

        Returns:
            Macro name or None
        """
        goal_hints = goal_hints or []

        for name, macro in self.macros.items():
            # Check page type match
            if page_type not in macro.page_types and "unknown" not in macro.page_types:
                continue

            # Check if first step elements exist
            first_step = macro.steps[0]
            if self._find_element(elements, first_step.element_role, first_step.element_hint):
                # Check goal hints
                if any(hint in name for hint in goal_hints):
                    return name
                # Also return if page type is specific enough
                if page_type in macro.page_types:
                    return name

        return None

    def _find_element(
        self,
        elements: List[Dict],
        role: str,
        hint: str
    ) -> Optional[Dict]:
        """Find element matching role and hint."""
        for elem in elements:
            if elem.get('role', '').lower() == role.lower():
                name = elem.get('name', '').lower()
                if not hint or hint.lower() in name:
                    return elem
        return None

    def start_macro(
        self,
        macro_name: str,
        params: Dict[str, str] = None
    ) -> bool:
        """Start executing a macro.

        Args:
            macro_name: Name of macro to start
            params: Parameters to fill in action templates

        Returns:
            True if macro started
        """
        if macro_name not in self.macros:
            return False

        self.active_macro = macro_name
        self.active_step = 0
        self.macro_params = params or {}
        self.macro_attempts += 1
        logger.info(f"Starting macro: {macro_name}")
        return True

    def get_next_action(
        self,
        elements: List[Dict]
    ) -> Optional[Tuple[str, str]]:
        """Get next action in active macro.

        Args:
            elements: Available elements on page

        Returns:
            Tuple of (action, postcondition) or None if macro complete/failed
        """
        if not self.active_macro:
            return None

        macro = self.macros[self.active_macro]

        if self.active_step >= len(macro.steps):
            self.macro_successes += 1
            self.active_macro = None
            return None

        step = macro.steps[self.active_step]

        # Find required element
        elem = self._find_element(elements, step.element_role, step.element_hint)
        if not elem:
            logger.warning(f"Macro step failed - element not found: {step.element_role} {step.element_hint}")
            self.active_macro = None
            return None

        # Build action from template
        params = {**self.macro_params, f"{step.element_role}_bid": elem['bid']}

        # Handle special placeholders
        action = step.action_template
        for key, value in params.items():
            action = action.replace(f"{{{key}}}", str(value))

        # Check for unfilled placeholders
        if '{' in action:
            logger.warning(f"Macro has unfilled placeholders: {action}")
            self.active_macro = None
            return None

        self.active_step += 1
        return (action, step.postcondition)

    def abort_macro(self):
        """Abort current macro."""
        if self.active_macro:
            logger.info(f"Aborting macro: {self.active_macro}")
            self.active_macro = None
            self.active_step = 0

    def is_active(self) -> bool:
        """Check if a macro is currently active."""
        return self.active_macro is not None

    def stats(self) -> Dict[str, Any]:
        """Get macro statistics."""
        return {
            "attempts": self.macro_attempts,
            "successes": self.macro_successes,
            "success_rate": self.macro_successes / self.macro_attempts if self.macro_attempts > 0 else 0,
            "active": self.active_macro
        }
