"""Observation summarizer for HALO Agent.

Converts observation dict -> compact text prompt for LLM policies.
"""

from typing import Dict, List, Any, Optional


def extract_actionable_nodes(axtree_object: Any, top_k: int = 30) -> List[Dict]:
    """Extract top-K actionable nodes from AXTree.

    Args:
        axtree_object: AXTree object from observation (can be nested dict or flat 'nodes' array)
        top_k: Maximum number of nodes to return

    Returns:
        List of actionable node dicts with bid, role, name
    """
    actionable_roles = {
        'button', 'link', 'textbox', 'combobox', 'menuitem',
        'checkbox', 'radio', 'searchbox', 'tab', 'menuitemcheckbox',
        'menuitemradio', 'option', 'spinbutton', 'slider', 'switch'
    }

    if top_k <= 0:
        return []

    recall_keywords = [
        "search",
        "checkout",
        "cart",
        "add to cart",
        "add to basket",
        "buy",
        "purchase",
        "place order",
        "order",
        "pay",
        "payment",
        "sign in",
        "log in",
        "login",
        "submit",
        "continue",
        "next",
        "send",
        "compose",
        "reply",
    ]

    def _merge_keyword_recall(top_nodes: List[Dict], recall_hits: Dict[str, Dict]) -> List[Dict]:
        if not recall_hits:
            return top_nodes[:top_k]

        recall_nodes: List[Dict] = []
        recall_bids = set()
        for kw in recall_keywords:
            cand = recall_hits.get(kw)
            if cand and cand.get("bid") not in recall_bids:
                recall_nodes.append(cand)
                recall_bids.add(cand.get("bid"))

        if not recall_nodes:
            return top_nodes[:top_k]

        merged = list(top_nodes)
        merged_bids = {n.get("bid") for n in merged}
        for cand in recall_nodes:
            if cand.get("bid") not in merged_bids:
                merged.append(cand)
                merged_bids.add(cand.get("bid"))

        if len(merged) > top_k:
            while len(merged) > top_k:
                removed = False
                for i in range(len(merged) - 1, -1, -1):
                    if merged[i].get("bid") not in recall_bids:
                        merged.pop(i)
                        removed = True
                        break
                if not removed:
                    merged = merged[:top_k]
                    break

        return merged[:top_k]

    nodes = []
    recall_hits = {}
    max_scan = max(top_k * 10, 300)
    scanned = 0

    # Handle flat 'nodes' array format from SDK
    if isinstance(axtree_object, dict) and 'nodes' in axtree_object:
        for node in axtree_object['nodes']:
            scanned += 1
            if scanned >= max_scan and len(nodes) >= top_k:
                break
            if isinstance(node, dict):
                # Role can be string or dict with 'value' key
                role_raw = node.get('role', '')
                if isinstance(role_raw, dict):
                    role = role_raw.get('value', '').lower()
                elif isinstance(role_raw, str):
                    role = role_raw.lower()
                else:
                    continue

                # SDK uses 'browsergym_id' for the element ID
                bid = node.get('browsergym_id', node.get('bid', ''))

                # Name can also be string or dict
                name_raw = node.get('name', '')
                if isinstance(name_raw, dict):
                    name = name_raw.get('value', '')
                elif isinstance(name_raw, str):
                    name = name_raw
                else:
                    name = ''

                if role in actionable_roles and bid:
                    elem = {
                        'bid': str(bid),
                        'role': role,
                        'name': name[:50] if name else '',
                        'depth': 0
                    }
                    if len(nodes) < top_k:
                        nodes.append(elem)
                    if name:
                        name_l = name.lower()
                        for kw in recall_keywords:
                            if kw not in recall_hits and kw in name_l:
                                recall_hits[kw] = elem
        return _merge_keyword_recall(nodes, recall_hits)

    # Handle nested dict format (fallback)
    def traverse(node: Any, depth: int = 0):
        nonlocal scanned
        if node is None:
            return

        if isinstance(node, dict):
            scanned += 1
            if scanned >= max_scan and len(nodes) >= top_k:
                return

            role_raw = node.get('role', '')
            if isinstance(role_raw, dict):
                role = role_raw.get('value', '').lower()
            elif isinstance(role_raw, str):
                role = role_raw.lower()
            else:
                role = ''
            bid = node.get('bid', node.get('browsergym_id', ''))
            name_raw = node.get('name', '')
            if isinstance(name_raw, dict):
                name = name_raw.get('value', '')
            elif isinstance(name_raw, str):
                name = name_raw
            else:
                name = ''

            if role in actionable_roles and bid:
                elem = {
                    'bid': str(bid),
                    'role': role,
                    'name': name[:50] if name else '',
                    'depth': depth
                }
                if len(nodes) < top_k:
                    nodes.append(elem)
                if name:
                    name_l = name.lower()
                    for kw in recall_keywords:
                        if kw not in recall_hits and kw in name_l:
                            recall_hits[kw] = elem

            # Traverse children
            children = node.get('children', [])
            if isinstance(children, list):
                for child in children:
                    traverse(child, depth + 1)

        elif isinstance(node, list):
            for item in node:
                traverse(item, depth)

    traverse(axtree_object)
    return _merge_keyword_recall(nodes, recall_hits)


def flatten_axtree_simple(axtree_object: Any) -> str:
    """Simple flattening of AXTree to text."""
    try:
        from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str
        return flatten_axtree_to_str(axtree_object)
    except ImportError:
        # Fallback if import fails
        nodes = extract_actionable_nodes(axtree_object, top_k=50)
        lines = []
        for n in nodes:
            lines.append(f"[{n['bid']}] {n['role']}: {n['name']}")
        return "\n".join(lines)


def summarize_observation(
    obs: Dict,
    goal: Optional[str] = None,
    top_k_nodes: int = 30
) -> str:
    """Summarize observation into compact text prompt.

    Args:
        obs: Observation dictionary from environment
        goal: Task goal text (if not in obs)
        top_k_nodes: Number of actionable nodes to include

    Returns:
        Compact text summary for LLM prompt
    """
    parts = []

    # Goal
    goal_text = goal
    if not goal_text and 'goal_object' in obs:
        goal_obj = obs['goal_object']
        if isinstance(goal_obj, list):
            goal_text = ' '.join(
                item.get('text', '') for item in goal_obj
                if isinstance(item, dict) and item.get('type') == 'text'
            )
        elif isinstance(goal_obj, str):
            goal_text = goal_obj

    if goal_text:
        parts.append(f"# Goal\n{goal_text}")

    # Current URL
    url = obs.get('url', '')
    if url:
        parts.append(f"# Current URL\n{url}")

    # Last action
    last_action = obs.get('last_action', '')
    if last_action:
        parts.append(f"# Last Action\n{last_action}")

    # Last action error
    last_error = obs.get('last_action_error', '')
    if last_error:
        parts.append(f"# Last Action Error\n{last_error}")

    # Actionable elements
    axtree = obs.get('axtree_object')
    if axtree:
        nodes = extract_actionable_nodes(axtree, top_k=top_k_nodes)
        if nodes:
            node_lines = []
            for n in nodes:
                name_part = f' "{n["name"]}"' if n['name'] else ''
                node_lines.append(f"[{n['bid']}] {n['role']}{name_part}")
            parts.append(f"# Actionable Elements (bid, role, name)\n" + "\n".join(node_lines))

    return "\n\n".join(parts)


def get_obs_hash(obs: Dict) -> str:
    """Get a hash of key observation components for caching."""
    import hashlib

    url = obs.get('url', '')
    last_action = obs.get('last_action', '')
    last_error = obs.get('last_action_error', '')

    # Include top actionable elements in hash
    axtree = obs.get('axtree_object')
    nodes_sig = ''
    if axtree:
        nodes = extract_actionable_nodes(axtree, top_k=10)
        nodes_sig = '|'.join(f"{n['bid']}:{n['role']}" for n in nodes)

    content = f"{url}|{last_action}|{last_error}|{nodes_sig}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
