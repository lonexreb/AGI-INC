"""Smoke tests for run_one_debug.py script.

Tests that the debug runner can be invoked in dry mode without LLM calls.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestRunOneDebugSmoke:
    """Smoke tests for the debug runner script."""

    @pytest.fixture
    def script_path(self):
        """Get path to run_one_debug.py script."""
        return Path(__file__).parent.parent.parent / "scripts" / "run_one_debug.py"

    @pytest.fixture
    def python_executable(self):
        """Get the Python executable path."""
        return sys.executable

    def test_script_exists(self, script_path):
        """Verify the script file exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_dry_run_exits_cleanly(self, script_path, python_executable):
        """Test that dry run mode exits cleanly without LLM calls."""
        result = subprocess.run(
            [
                python_executable,
                str(script_path),
                "--task", "v2.omnizon-13",
                "--mode", "baseline_worker",
                "--dry-run"
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert "Dry run completed successfully" in result.stdout or "DRY RUN" in result.stdout

    def test_dry_run_with_different_modes(self, script_path, python_executable):
        """Test dry run with different agent modes."""
        modes = [
            'baseline_worker',
            'hierarchy_mgr_gate',
            'hierarchy_vac',
            'hierarchy_vac_macros'
        ]
        
        for mode in modes:
            result = subprocess.run(
                [
                    python_executable,
                    str(script_path),
                    "--task", "v2.gomail-1",
                    "--mode", mode,
                    "--dry-run"
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(script_path.parent.parent)
            )
            
            assert result.returncode == 0, f"Mode {mode} failed: {result.stderr}"

    def test_help_flag(self, script_path, python_executable):
        """Test that --help works."""
        result = subprocess.run(
            [python_executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0
        assert "--task" in result.stdout
        assert "--mode" in result.stdout
        assert "--dry-run" in result.stdout

    def test_missing_task_argument_fails(self, script_path, python_executable):
        """Test that missing --task argument causes failure."""
        result = subprocess.run(
            [python_executable, str(script_path), "--mode", "baseline_worker"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode != 0, "Should fail without --task argument"

    def test_invalid_mode_fails(self, script_path, python_executable):
        """Test that invalid mode causes failure."""
        result = subprocess.run(
            [
                python_executable,
                str(script_path),
                "--task", "v2.omnizon-13",
                "--mode", "invalid_mode"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode != 0, "Should fail with invalid mode"


class TestListV2TasksSmoke:
    """Smoke tests for list_v2_tasks.py script."""

    @pytest.fixture
    def script_path(self):
        """Get path to list_v2_tasks.py script."""
        return Path(__file__).parent.parent.parent / "scripts" / "list_v2_tasks.py"

    @pytest.fixture
    def python_executable(self):
        """Get the Python executable path."""
        return sys.executable

    def test_script_runs_successfully(self, script_path, python_executable):
        """Test that list_v2_tasks.py runs and outputs task info."""
        result = subprocess.run(
            [python_executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Total v2 tasks:" in result.stdout
        assert "Per-site counts:" in result.stdout

    def test_ids_only_flag(self, script_path, python_executable):
        """Test --ids-only flag outputs just task IDs."""
        result = subprocess.run(
            [python_executable, str(script_path), "--ids-only"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) > 0
        assert all(line.startswith("v2.") for line in lines if line)

    def test_site_filter(self, script_path, python_executable):
        """Test --site filter."""
        result = subprocess.run(
            [python_executable, str(script_path), "--site", "omnizon", "--ids-only"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) > 0
        assert all("omnizon" in line for line in lines if line)

    def test_help_flag(self, script_path, python_executable):
        """Test --help flag."""
        result = subprocess.run(
            [python_executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(script_path.parent.parent)
        )
        
        assert result.returncode == 0
        assert "--json" in result.stdout
        assert "--site" in result.stdout
