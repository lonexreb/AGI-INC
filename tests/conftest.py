"""Pytest fixtures for HALO-Agent tests."""

import pytest
from typing import Dict, Any


@pytest.fixture
async def browser():
    """Fixture for playwright browser instance."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser):
    """Fixture for playwright page instance."""
    page = await browser.new_page()
    yield page
    await page.close()


@pytest.fixture
def mock_state() -> Dict[str, Any]:
    """Fixture for mock browser state."""
    return {
        "url": "https://example.com",
        "title": "Example Domain",
        "dom_elements": [
            {"type": "button", "text": "Click me", "id": "btn1"},
            {"type": "input", "placeholder": "Enter text", "id": "input1"},
        ],
        "screenshot": None,
    }


@pytest.fixture
def mock_action() -> Dict[str, Any]:
    """Fixture for mock action."""
    return {
        "type": "click",
        "target": "btn1",
        "reason": "Test click action",
    }
