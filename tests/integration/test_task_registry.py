"""Integration tests for task registry and discovery.

Tests that list_v2_tasks correctly discovers available v2 tasks.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestTaskRegistry:
    """Tests for v2 task discovery."""

    def test_discover_v2_tasks_returns_non_empty_list(self):
        """Ensure halo.sdk.task_registry returns a non-empty list of tasks."""
        from halo.sdk.task_registry import list_real_tasks

        tasks = list_real_tasks(task_version="v2")

        assert tasks is not None, "list_real_tasks returned None"
        assert isinstance(tasks, list), "list_real_tasks should return a list"
        assert len(tasks) > 0, "list_real_tasks returned empty list"

    def test_all_tasks_have_v2_prefix(self):
        """Ensure all discovered tasks have v2. prefix."""
        from halo.sdk.task_registry import list_real_tasks

        tasks = list_real_tasks(task_version="v2")

        for task in tasks:
            assert isinstance(task, dict)
            assert str(task.get("task_name", "")).startswith("v2."), f"Task '{task}' missing v2. prefix"

    def test_known_sites_exist(self):
        """Test that expected sites are present in discovered tasks."""
        from halo.sdk.task_registry import count_by_site, list_real_tasks

        tasks = list_real_tasks(task_version="v2")
        counts = count_by_site(tasks)

        expected_sites = ["omnizon", "gomail", "dashdish", "topwork"]

        for site in expected_sites:
            assert site in counts, f"Expected site '{site}' not found in discovered tasks"

    def test_task_count_reasonable(self):
        """Test that we discover a reasonable number of tasks."""
        from halo.sdk.task_registry import list_real_tasks

        tasks = list_real_tasks(task_version="v2")

        assert len(tasks) >= 100, f"Expected at least 100 v2 tasks, found {len(tasks)}"
        assert len(tasks) <= 500, f"Found unexpectedly many tasks: {len(tasks)}"

    def test_snapshot_schema_and_known_sites(self, tmp_path: Path):
        """Validate snapshot_real_tasks.py outputs the expected JSON schema."""

        repo_root = Path(__file__).parent.parent.parent
        script_path = repo_root / "scripts" / "snapshot_real_tasks.py"
        out_path = tmp_path / "real_v2_task_registry.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--task_version",
                "v2",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
        )

        assert result.returncode == 0, f"snapshot_real_tasks.py failed: {result.stderr}"
        assert out_path.exists(), "Snapshot file was not created"

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert data.get("task_version") == "v2"

        total_tasks = data.get("total_tasks")
        assert isinstance(total_tasks, int)
        assert total_tasks >= 100

        counts_by_site = data.get("counts_by_site")
        assert isinstance(counts_by_site, dict)

        for site in ["omnizon", "gomail", "dashdish", "topwork"]:
            assert site in counts_by_site, f"Expected site '{site}' not present in snapshot"

        tasks = data.get("tasks")
        assert isinstance(tasks, list)
        assert len(tasks) == total_tasks

        first = tasks[0]
        assert isinstance(first, dict)
        assert "task_name" in first
        assert str(first["task_name"]).startswith("v2.")
