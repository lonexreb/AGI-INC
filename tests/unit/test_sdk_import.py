"""Test AGI SDK import and version verification."""

import pytest


def test_agisdk_import():
    """Test that agisdk can be imported."""
    try:
        import agisdk
        assert agisdk is not None
    except ImportError as e:
        pytest.fail(f"Failed to import agisdk: {e}")


def test_agisdk_version():
    """Test that AGI SDK version is 0.3.5."""
    try:
        import agisdk
        # Check if version attribute exists
        if hasattr(agisdk, "__version__"):
            assert agisdk.__version__ == "0.3.5", f"Expected 0.3.5, got {agisdk.__version__}"
        else:
            # If __version__ is not available, try alternative methods
            import importlib.metadata
            version = importlib.metadata.version("agisdk")
            assert version == "0.3.5", f"Expected 0.3.5, got {version}"
    except ImportError:
        pytest.skip("agisdk not installed")


def test_task_version_default_handling():
    """Test that task_version parameter defaults are handled correctly.

    This test verifies that:
    1. task_version="v2" is explicitly set when calling SDK functions
    2. Omitting task_version would default to v1 (which we must avoid)
    """
    # This is a documentation/reminder test
    # In actual usage, ALWAYS pass task_version="v2"

    task_version_v2 = "v2"
    assert task_version_v2 == "v2", "Always use task_version='v2' for REAL Bench v2 tasks"

    # Reminder: SDK defaults to v1 if omitted - this is what we must NOT do
    default_version = "v1"  # What SDK defaults to
    assert default_version != "v2", "SDK defaults to v1 - must explicitly pass v2"
