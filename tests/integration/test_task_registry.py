"""Integration tests for task registry and discovery.

Tests that list_v2_tasks correctly discovers available v2 tasks.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestTaskRegistry:
    """Tests for v2 task discovery."""

    def test_discover_v2_tasks_returns_non_empty_list(self):
        """Ensure list_v2_tasks returns a non-empty list of tasks."""
        from list_v2_tasks import discover_v2_tasks
        
        tasks, source = discover_v2_tasks()
        
        assert tasks is not None, "discover_v2_tasks returned None"
        assert isinstance(tasks, list), "discover_v2_tasks should return a list"
        assert len(tasks) > 0, "discover_v2_tasks returned empty list"
        assert source, "discover_v2_tasks should return a source description"

    def test_all_tasks_have_v2_prefix(self):
        """Ensure all discovered tasks have v2. prefix."""
        from list_v2_tasks import discover_v2_tasks
        
        tasks, _ = discover_v2_tasks()
        
        for task in tasks:
            assert task.startswith("v2."), f"Task '{task}' missing v2. prefix"

    def test_get_v2_tasks_api(self):
        """Test the get_v2_tasks API function."""
        from list_v2_tasks import get_v2_tasks
        
        tasks = get_v2_tasks()
        
        assert tasks is not None
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_get_v2_tasks_with_site_filter(self):
        """Test filtering tasks by site."""
        from list_v2_tasks import get_v2_tasks
        
        all_tasks = get_v2_tasks()
        omnizon_tasks = get_v2_tasks(sites=["omnizon"])
        
        assert len(omnizon_tasks) > 0, "Should find omnizon tasks"
        assert len(omnizon_tasks) < len(all_tasks), "Filtered list should be smaller"
        
        for task in omnizon_tasks:
            assert "omnizon" in task, f"Task '{task}' should contain 'omnizon'"

    def test_extract_site(self):
        """Test site extraction from task IDs."""
        from list_v2_tasks import extract_site
        
        assert extract_site("v2.omnizon-1") == "omnizon"
        assert extract_site("v2.omnizon-13") == "omnizon"
        assert extract_site("v2.gomail-1") == "gomail"
        assert extract_site("v2.dashdish-10") == "dashdish"
        assert extract_site("omnizon-1") == "omnizon"

    def test_group_tasks_by_site(self):
        """Test grouping tasks by site."""
        from list_v2_tasks import group_tasks_by_site
        
        tasks = ["v2.omnizon-1", "v2.omnizon-2", "v2.gomail-1"]
        grouped = group_tasks_by_site(tasks)
        
        assert "omnizon" in grouped
        assert "gomail" in grouped
        assert len(grouped["omnizon"]) == 2
        assert len(grouped["gomail"]) == 1

    def test_known_sites_exist(self):
        """Test that expected sites are present in discovered tasks."""
        from list_v2_tasks import discover_v2_tasks, group_tasks_by_site
        
        tasks, _ = discover_v2_tasks()
        grouped = group_tasks_by_site(tasks)
        
        expected_sites = ["omnizon", "gomail", "dashdish", "topwork"]
        
        for site in expected_sites:
            assert site in grouped, f"Expected site '{site}' not found in discovered tasks"

    def test_task_count_reasonable(self):
        """Test that we discover a reasonable number of tasks."""
        from list_v2_tasks import discover_v2_tasks
        
        tasks, _ = discover_v2_tasks()
        
        assert len(tasks) >= 50, f"Expected at least 50 v2 tasks, found {len(tasks)}"
        assert len(tasks) <= 500, f"Found unexpectedly many tasks: {len(tasks)}"
