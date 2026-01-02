from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_task_registry_snapshot_path(task_version: str = "v2") -> Path:
    return _repo_root() / "configs" / f"real_{task_version}_task_registry.json"


def get_agisdk_version() -> str:
    try:
        return importlib.metadata.version("agisdk")
    except Exception:
        return ""


def _split_task_type_and_id(task_name: str) -> Tuple[str, Optional[int]]:
    name = task_name
    if "." in name:
        _, name = name.split(".", 1)

    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])

    return name, None


def _discover_task_names_from_agisdk(task_version: str) -> Tuple[List[str], str]:
    try:
        from agisdk.REAL.browsergym.webclones.task_config import TASKS_BY_VERSION

        tasks = TASKS_BY_VERSION.get(task_version, [])
        if not tasks:
            return [], "agisdk.TASKS_BY_VERSION(empty)"

        return [f"{task_version}.{t}" for t in tasks], "agisdk.TASKS_BY_VERSION"
    except Exception:
        return [], "agisdk(import_error)"


def _discover_task_names_from_third_party(task_version: str) -> Tuple[List[str], str]:
    repo_root = _repo_root()
    tasks_dir = (
        repo_root
        / "third_party"
        / "agisdk"
        / "src"
        / "agisdk"
        / "REAL"
        / "browsergym"
        / "webclones"
        / task_version
        / "tasks"
    )

    if not tasks_dir.exists():
        return [], "third_party(missing)"

    tasks = sorted(p.stem for p in tasks_dir.glob("*.json"))
    return [f"{task_version}.{t}" for t in tasks], f"third_party({tasks_dir})"


def list_real_tasks(task_version: str = "v2") -> List[Dict[str, Any]]:
    tasks, source = _discover_task_names_from_agisdk(task_version)
    if not tasks:
        tasks, source = _discover_task_names_from_third_party(task_version)

    if not tasks:
        raise RuntimeError(f"No REAL tasks discovered for task_version={task_version} (source={source})")

    records: List[Dict[str, Any]] = []
    for task_name in sorted(tasks):
        task_type, task_id = _split_task_type_and_id(task_name)
        records.append(
            {
                "task_name": task_name,
                "task_type": task_type,
                "task_id": task_id,
            }
        )

    return records


def group_by_site(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        site = str(task.get("task_type", ""))
        grouped.setdefault(site, []).append(task)

    for site in list(grouped.keys()):
        grouped[site] = sorted(grouped[site], key=lambda t: str(t.get("task_name", "")))

    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def count_by_site(tasks: List[Dict[str, Any]]) -> Dict[str, int]:
    grouped = group_by_site(tasks)
    return {site: len(site_tasks) for site, site_tasks in grouped.items()}


def load_task_registry_snapshot(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Task registry snapshot must be a JSON object: {path}")

    if "tasks" not in data or not isinstance(data.get("tasks"), list):
        raise ValueError(f"Task registry snapshot missing 'tasks' list: {path}")

    return data


def task_names_from_snapshot(data: Dict[str, Any], task_version: str = "v2") -> List[str]:
    tasks = data.get("tasks", [])
    if not tasks:
        return []

    first = tasks[0]
    if isinstance(first, str):
        return [t if t.startswith(f"{task_version}.") else f"{task_version}.{t}" for t in tasks]

    names: List[str] = []
    for t in tasks:
        if not isinstance(t, dict) or "task_name" not in t:
            raise ValueError("Task registry snapshot tasks must be strings or dicts with 'task_name'")
        name = str(t["task_name"])
        if not name.startswith(f"{task_version}."):
            name = f"{task_version}.{name}"
        names.append(name)

    return sorted(names)


def load_real_task_names(
    task_version: str = "v2",
    task_registry: Optional[str] = None,
) -> Tuple[List[str], str]:
    if task_registry:
        path = Path(task_registry)
        data = load_task_registry_snapshot(path)
        return task_names_from_snapshot(data, task_version=task_version), f"snapshot:{path}"

    default_path = default_task_registry_snapshot_path(task_version)
    if default_path.exists():
        data = load_task_registry_snapshot(default_path)
        return task_names_from_snapshot(data, task_version=task_version), f"snapshot:{default_path}"

    tasks = list_real_tasks(task_version=task_version)
    return [t["task_name"] for t in tasks], "dynamic"
