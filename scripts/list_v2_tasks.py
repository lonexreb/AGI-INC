#!/usr/bin/env python3
"""
Task discovery utility for REAL Bench v2 tasks.

Lists all available v2 task IDs from the installed agisdk or third_party/agisdk.
Provides per-site counts and supports JSON export.

Usage:
    python scripts/list_v2_tasks.py
    python scripts/list_v2_tasks.py --json tasks.json
    python scripts/list_v2_tasks.py --site omnizon gomail
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def discover_v2_tasks_from_sdk() -> Tuple[List[str], str]:
    """Discover v2 tasks from the installed agisdk package.
    
    Returns:
        Tuple of (list of task IDs, source description)
    """
    try:
        from agisdk.REAL.browsergym.webclones.task_config import TASKS_BY_VERSION
        tasks = TASKS_BY_VERSION.get('v2', [])
        return [f"v2.{t}" for t in tasks], "agisdk package"
    except ImportError:
        return [], ""


def discover_v2_tasks_from_third_party() -> Tuple[List[str], str]:
    """Discover v2 tasks from third_party/agisdk directory.
    
    Returns:
        Tuple of (list of task IDs, source description)
    """
    repo_root = Path(__file__).parent.parent
    tasks_dir = repo_root / "third_party" / "agisdk" / "src" / "agisdk" / "REAL" / "browsergym" / "webclones" / "v2" / "tasks"
    
    if not tasks_dir.exists():
        return [], ""
    
    tasks = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        task_name = task_file.stem
        tasks.append(f"v2.{task_name}")
    
    return tasks, f"third_party/agisdk ({tasks_dir})"


def discover_v2_tasks() -> Tuple[List[str], str]:
    """Discover all available v2 tasks.
    
    Tries SDK first, falls back to third_party directory.
    
    Returns:
        Tuple of (list of task IDs with v2. prefix, source description)
    """
    # Try installed SDK first
    tasks, source = discover_v2_tasks_from_sdk()
    if tasks:
        return tasks, source
    
    # Fall back to third_party
    tasks, source = discover_v2_tasks_from_third_party()
    if tasks:
        return tasks, source
    
    return [], "No tasks found"


def extract_site(task_id: str) -> str:
    """Extract site name from task ID.
    
    Args:
        task_id: Task ID like 'v2.omnizon-1' or 'omnizon-1'
        
    Returns:
        Site name like 'omnizon'
    """
    # Remove v2. prefix if present
    name = task_id.replace("v2.", "")
    # Extract site (everything before the last dash and number)
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def group_tasks_by_site(tasks: List[str]) -> Dict[str, List[str]]:
    """Group tasks by their site.
    
    Args:
        tasks: List of task IDs
        
    Returns:
        Dict mapping site name to list of task IDs
    """
    by_site = defaultdict(list)
    for task in tasks:
        site = extract_site(task)
        by_site[site].append(task)
    return dict(by_site)


def print_task_summary(tasks: List[str], source: str, sites_filter: Optional[List[str]] = None):
    """Print a summary of discovered tasks.
    
    Args:
        tasks: List of task IDs
        source: Description of where tasks were found
        sites_filter: Optional list of sites to filter by
    """
    by_site = group_tasks_by_site(tasks)
    
    # Filter by sites if specified
    if sites_filter:
        by_site = {k: v for k, v in by_site.items() if k in sites_filter}
        tasks = [t for t in tasks if extract_site(t) in sites_filter]
    
    print("=" * 60)
    print("REAL Bench v2 Task Discovery")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Total v2 tasks: {len(tasks)}")
    print(f"Sites: {len(by_site)}")
    print()
    
    print("Per-site counts:")
    print("-" * 40)
    for site in sorted(by_site.keys()):
        site_tasks = by_site[site]
        print(f"  {site}: {len(site_tasks)} tasks")
    print()
    
    print("First 5 tasks per site:")
    print("-" * 40)
    for site in sorted(by_site.keys()):
        site_tasks = sorted(by_site[site])
        sample = site_tasks[:5]
        print(f"  {site}:")
        for task in sample:
            print(f"    - {task}")
        if len(site_tasks) > 5:
            print(f"    ... and {len(site_tasks) - 5} more")
    print()


def export_to_json(tasks: List[str], source: str, output_path: str, sites_filter: Optional[List[str]] = None):
    """Export task list to JSON file.
    
    Args:
        tasks: List of task IDs
        source: Description of where tasks were found
        output_path: Path to output JSON file
        sites_filter: Optional list of sites to filter by
    """
    by_site = group_tasks_by_site(tasks)
    
    # Filter by sites if specified
    if sites_filter:
        by_site = {k: v for k, v in by_site.items() if k in sites_filter}
        tasks = [t for t in tasks if extract_site(t) in sites_filter]
    
    data = {
        "source": source,
        "total_tasks": len(tasks),
        "sites": list(sorted(by_site.keys())),
        "per_site_counts": {site: len(tasks_list) for site, tasks_list in sorted(by_site.items())},
        "tasks": sorted(tasks),
        "tasks_by_site": {site: sorted(tasks_list) for site, tasks_list in sorted(by_site.items())}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(tasks)} tasks to {output_path}")


def get_v2_tasks(sites: Optional[List[str]] = None) -> List[str]:
    """Get list of v2 tasks, optionally filtered by site.
    
    This is the main API function for other scripts to use.
    
    Args:
        sites: Optional list of site names to filter by
        
    Returns:
        List of task IDs with v2. prefix
    """
    tasks, _ = discover_v2_tasks()
    
    if sites:
        tasks = [t for t in tasks if extract_site(t) in sites]
    
    return sorted(tasks)


def main():
    parser = argparse.ArgumentParser(
        description="Discover and list REAL Bench v2 tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all v2 tasks
    python scripts/list_v2_tasks.py
    
    # Export to JSON
    python scripts/list_v2_tasks.py --json tasks.json
    
    # Filter by site
    python scripts/list_v2_tasks.py --site omnizon gomail
    
    # List only task IDs (for scripting)
    python scripts/list_v2_tasks.py --ids-only
"""
    )
    parser.add_argument(
        "--json",
        type=str,
        metavar="PATH",
        help="Export task list to JSON file"
    )
    parser.add_argument(
        "--site",
        type=str,
        nargs='+',
        metavar="SITE",
        help="Filter by site name(s) (e.g., omnizon gomail)"
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Print only task IDs, one per line (for scripting)"
    )
    
    args = parser.parse_args()
    
    # Discover tasks
    tasks, source = discover_v2_tasks()
    
    if not tasks:
        print("ERROR: No v2 tasks found!", file=sys.stderr)
        print("Make sure agisdk is installed or third_party/agisdk exists.", file=sys.stderr)
        sys.exit(1)
    
    # Filter by sites if specified
    filtered_tasks = tasks
    if args.site:
        filtered_tasks = [t for t in tasks if extract_site(t) in args.site]
        if not filtered_tasks:
            print(f"ERROR: No tasks found for sites: {args.site}", file=sys.stderr)
            print(f"Available sites: {sorted(set(extract_site(t) for t in tasks))}", file=sys.stderr)
            sys.exit(1)
    
    # Output based on mode
    if args.ids_only:
        for task in sorted(filtered_tasks):
            print(task)
    elif args.json:
        export_to_json(tasks, source, args.json, args.site)
    else:
        print_task_summary(tasks, source, args.site)


if __name__ == "__main__":
    main()
