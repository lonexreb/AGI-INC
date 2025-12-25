#!/usr/bin/env python3
"""
Smoke test for HALO-Agent setup.
Verifies dependencies, imports, and creates initial directories.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 50)
    print("HALO-Agent Smoke Test")
    print("=" * 50)
    print()
    
    all_passed = True
    version = "unknown"

    # Test 1: Import agisdk and print version
    print("1. Testing agisdk import...")
    try:
        import agisdk
        version = getattr(agisdk, "__version__", "unknown")
        print(f"   ✓ agisdk imported successfully (version: {version})\n")
    except ImportError as e:
        print(f"   ✗ Failed to import agisdk: {e}\n")
        all_passed = False

    # Test 2: Verify playwright is importable
    print("2. Testing playwright import...")
    try:
        import playwright
        print(f"   ✓ playwright imported successfully\n")
    except ImportError as e:
        print(f"   ✗ Failed to import playwright: {e}\n")
        all_passed = False

    # Test 3: Import HALO modules
    print("3. Testing HALO module imports...")
    try:
        from halo.sdk import HaloAgent, SUPPORTED_MODES, validate_action
        from halo.agent import Orchestrator, OrchestratorConfig
        from halo.policy import WorkerPolicy, ManagerPolicy, GatingController
        from halo.cache import VerifiedActionCache, MacroReplayCache
        from halo.verify import ActionVerifier, LoopDetector
        from halo.logging import TrajectoryLogger
        from halo.obs import summarize_observation, build_state_key
        print(f"   ✓ All HALO modules imported successfully")
        print(f"   ✓ Supported modes: {SUPPORTED_MODES}\n")
    except ImportError as e:
        print(f"   ✗ Failed to import HALO modules: {e}\n")
        all_passed = False

    # Test 4: Verify action validation
    print("4. Testing action validation...")
    try:
        from halo.sdk import validate_action, repair_action
        
        # Test valid actions
        valid_actions = [
            'click("123")',
            'fill("input", "text")',
            'scroll(0, 100)',
            'noop()',
            'go_back()',
            'send_msg_to_user("hello")',
        ]
        for action in valid_actions:
            assert validate_action(action), f"Should be valid: {action}"
        
        # Test invalid actions
        assert not validate_action(''), "Empty should be invalid"
        assert not validate_action('invalid'), "Invalid should be invalid"
        
        # Test repair
        assert repair_action('') == 'noop()', "Empty should repair to noop"
        
        print(f"   ✓ Action validation working correctly\n")
    except Exception as e:
        print(f"   ✗ Action validation failed: {e}\n")
        all_passed = False

    # Test 5: Create agent with each mode
    print("5. Testing agent creation for each mode...")
    try:
        from halo.sdk import HaloAgent, SUPPORTED_MODES
        
        for mode in ['baseline_worker', 'hierarchy_mgr_gate', 'hierarchy_vac', 'hierarchy_vac_macros']:
            agent = HaloAgent(mode=mode, max_steps=5)
            assert agent.mode == mode
            print(f"   ✓ Created agent with mode: {mode}")
        print()
    except Exception as e:
        print(f"   ✗ Agent creation failed: {e}\n")
        all_passed = False

    # Test 6: Create directories
    print("6. Creating required directories...")
    repo_root = Path(__file__).parent.parent
    dirs_to_create = [
        repo_root / "results",
        repo_root / "data" / "trajectories",
        repo_root / "data" / "datasets",
        repo_root / "data" / "cache",
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path.relative_to(repo_root)}")
    print()

    # Test 7: Write status.json
    print("7. Writing smoke test status...")
    results_dir = repo_root / "results" / "smoke_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "smoke_test": "passed" if all_passed else "failed",
        "agisdk_version": version,
        "supported_modes": ['baseline_worker', 'hierarchy_mgr_gate', 'hierarchy_vac', 'hierarchy_vac_macros'],
        "constraints": {
            "task_version": "v2",
            "browser_dimensions": [1280, 720],
            "use_html": False,
            "default_max_steps": 70
        }
    }

    status_file = results_dir / "status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    print(f"   ✓ Wrote {status_file.relative_to(repo_root)}\n")

    # Summary
    print("=" * 50)
    if all_passed:
        print("SUCCESS: All smoke tests passed!")
        print()
        print("Next steps:")
        print("  python scripts/eval_subset.py --mode baseline_worker --dry-run")
        print("  python scripts/eval_subset.py --mode baseline_worker --subset_size 5")
    else:
        print("FAILED: Some smoke tests failed!")
        print("Please check the errors above and fix before proceeding.")
    print("=" * 50)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
