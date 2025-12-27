import importlib.util
from pathlib import Path
from typing import Any, Dict, List


def _load_collect_traj_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "collect_traj.py"
    spec = importlib.util.spec_from_file_location("collect_traj", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load collect_traj module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_episode(
    task_id: str,
    max_progress_score: float,
    actions: List[str],
    action_source: str = "worker",
) -> Dict[str, Any]:
    steps = []
    for i, action in enumerate(actions, start=1):
        steps.append(
            {
                "type": "step",
                "task_id": task_id,
                "task_name": task_id,
                "step_idx": i,
                "url": f"https://example.com/{i}",
                "obs_hash": f"obs_{max_progress_score}_{i}",
                "action": action,
                "action_source": action_source,
                "last_action_error": "",
                "progress_score": max_progress_score,
                "obs_summary": f"obs_summary_{max_progress_score}_{i}",
            }
        )

    episode_end = {
        "type": "episode_end",
        "task_id": task_id,
        "task_name": task_id,
        "success": False,
        "total_steps": len(steps),
        "max_progress_score": max_progress_score,
    }

    episode_start = {
        "type": "episode_start",
        "task_id": task_id,
        "task_name": task_id,
    }

    return {
        "steps": steps,
        "episode": episode_end,
        "episode_start": episode_start,
    }


def test_progress_ranked_bc_and_dpo():
    mod = _load_collect_traj_module()

    task_id = "v2.omnizon-1"

    episodes = {
        "ep_best": _make_episode(task_id, 0.8, ['click("a1")', 'click("a2")']),
        "ep_mid": _make_episode(task_id, 0.5, ['click("b1")', 'click("b2")']),
        "ep_worst": _make_episode(task_id, 0.1, ['click("c1")', 'click("c2")']),
    }

    bc = mod.generate_bc_dataset_progress_ranked(episodes, top_percent=0.2)
    assert len(bc) == 2
    assert [ex.action for ex in bc] == ['click("a1")', 'click("a2")']
    assert all(ex.task_id == task_id for ex in bc)
    assert all(ex.site_id == "omnizon" for ex in bc)

    dpo = mod.generate_dpo_dataset_progress_ranked(episodes)
    assert len(dpo) == 2
    assert [ex.chosen for ex in dpo] == ['click("a1")', 'click("a2")']
    assert [ex.rejected for ex in dpo] == ['click("c1")', 'click("c2")']
    assert all(ex.task_id == task_id for ex in dpo)
    assert all(ex.site_id == "omnizon" for ex in dpo)
