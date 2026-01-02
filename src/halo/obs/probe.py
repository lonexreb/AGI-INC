"""AXTree probe for HALO Agent.

Ported from TypeScript ax_probe.ts.
Extracts semantic signals from AXTree for state hashing and fallback logic.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AXSignals:
    url: str
    title: str
    focused_summary: Optional[str]
    dialog_count: int
    alert_count: int
    enabled_button_count: int
    disabled_button_count: int
    required_field_count: int
    invalid_field_count: int
    visible_text_digest: str
    role_name_digest: str
    raw: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).replace("\n", " ").strip().lower()

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def extract_signals(obs: Dict[str, Any]) -> AXSignals:
    """Extract semantic signals from observation.
    
    Args:
        obs: Observation dictionary containing 'axtree_object', 'url', 'title'
        
    Returns:
        AXSignals object
    """
    url = obs.get('url', '')
    title = obs.get('title', '')
    
    # Initialize buckets
    role_names: List[str] = []
    texts: List[str] = []
    
    raw_dialogs: List[Dict] = []
    raw_alerts: List[Dict] = []
    raw_buttons: List[Dict] = []
    raw_invalid_fields: List[Dict] = []
    raw_required_fields: List[Dict] = []
    
    focused_node_summary = None

    def _process_node(node: Dict[str, Any]):
        nonlocal focused_node_summary
        
        if not isinstance(node, dict):
            return

        # Extract basic props handling both direct values and dicts (SDK format)
        role = ""
        role_raw = node.get('role')
        if isinstance(role_raw, dict):
            role = _norm(role_raw.get('value', ''))
        else:
            role = _norm(role_raw)
            
        name = ""
        name_raw = node.get('name')
        if isinstance(name_raw, dict):
            name = _norm(name_raw.get('value', ''))
        else:
            name = _norm(name_raw)
            
        value = ""
        value_raw = node.get('value')
        if isinstance(value_raw, dict):
            value = _norm(value_raw.get('value', ''))
        else:
            value = _norm(value_raw)

        # Focused state
        is_focused = node.get('focused', False)
        if is_focused:
            bid = node.get('browsergym_id', node.get('bid', ''))
            focused_node_summary = f"{role}#{bid}:{name}"[:200]

        # Digest material
        if role:
            role_names.append(f"{role}:{name}")
        if name:
            texts.append(name)
        if value:
            texts.append(value)

        # Dialogs / Alerts
        if role in ('dialog', 'alertdialog'):
            raw_dialogs.append({'role': role, 'name': name, 'bid': node.get('browsergym_id')})
        if role in ('alert', 'status'):
            raw_alerts.append({'role': role, 'name': name, 'value': value})

        # Buttons
        if role == 'button':
            disabled = (
                node.get('disabled') is True or
                str(node.get('aria-disabled')).lower() == 'true'
            )
            raw_buttons.append({'name': name, 'disabled': disabled, 'bid': node.get('browsergym_id')})

        # Fields
        is_field = role in ('textbox', 'combobox', 'searchbox', 'spinbutton')
        
        # Required
        # SDK might put 'required' as a property or attribute
        required = (
            node.get('required') is True or
            str(node.get('aria-required')).lower() == 'true'
        )
        if is_field and required:
            raw_required_fields.append({'role': role, 'name': name, 'value': value, 'bid': node.get('browsergym_id')})

        # Invalid
        invalid = (
            node.get('invalid') is True or 
            str(node.get('aria-invalid')).lower() == 'true'
        )
        if is_field and invalid:
            raw_invalid_fields.append({'role': role, 'name': name, 'value': value, 'bid': node.get('browsergym_id')})

    def _traverse(obj: Any):
        if isinstance(obj, dict):
            # If it has a role, process it as a node
            if 'role' in obj:
                _process_node(obj)
            
            # Recurse into children
            # Handle standard 'children' list
            if 'children' in obj and isinstance(obj['children'], list):
                for child in obj['children']:
                    _traverse(child)
            
        elif isinstance(obj, list):
            for item in obj:
                _traverse(item)

    # Start traversal
    axtree = obs.get('axtree_object')
    if axtree:
        if isinstance(axtree, dict) and 'nodes' in axtree:
            # Flat list format
            for node in axtree['nodes']:
                _process_node(node)
        else:
            # Nested or other format
            _traverse(axtree)

    # Compute digests
    role_names.sort()
    texts.sort()
    
    role_name_digest = _sha256("|".join(role_names))[:16]
    visible_text_digest = _sha256("|".join(texts))[:16]
    
    enabled_btn_count = sum(1 for b in raw_buttons if not b['disabled'])
    disabled_btn_count = len(raw_buttons) - enabled_btn_count

    return AXSignals(
        url=url,
        title=title,
        focused_summary=focused_node_summary,
        dialog_count=len(raw_dialogs),
        alert_count=len(raw_alerts),
        enabled_button_count=enabled_btn_count,
        disabled_button_count=disabled_btn_count,
        required_field_count=len(raw_required_fields),
        invalid_field_count=len(raw_invalid_fields),
        visible_text_digest=visible_text_digest,
        role_name_digest=role_name_digest,
        raw={
            'dialogs': raw_dialogs,
            'alerts': raw_alerts,
            'invalidFields': raw_invalid_fields,
            'requiredFields': raw_required_fields,
            'buttons': raw_buttons
        }
    )

def compute_state_hash(sig: AXSignals) -> str:
    """Compute stable state hash from signals."""
    payload = [
        sig.url,
        sig.title,
        sig.focused_summary or "",
        f"dlg:{sig.dialog_count}",
        f"al:{sig.alert_count}",
        f"btnE:{sig.enabled_button_count}",
        f"btnD:{sig.disabled_button_count}",
        f"req:{sig.required_field_count}",
        f"inv:{sig.invalid_field_count}",
        sig.visible_text_digest,
        sig.role_name_digest
    ]
    return _sha256("||".join(payload))
