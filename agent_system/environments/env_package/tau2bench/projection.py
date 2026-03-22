"""
Projection functions for tau2-bench environments.

Solver projection: parse agent text into tool calls or text responses.
Challenger projection: parse challenger output into task specifications.
"""

import json
import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Solver tag patterns
# ---------------------------------------------------------------------------
RE_TOOL_CALL = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
RE_RESPONSE = re.compile(
    r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Challenger tag patterns — TOD-Zero: instructions + actions
# ---------------------------------------------------------------------------
RE_INSTRUCTIONS = re.compile(r"<instructions>(.*?)</instructions>", re.DOTALL | re.IGNORECASE)
RE_ACTIONS_TAG = re.compile(r"<actions>(.*?)</actions>", re.DOTALL | re.IGNORECASE)


def _safe_json_loads(s: str):
    """Best-effort JSON parse."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def solver_projection(actions: List[str]) -> Tuple[List[Dict], List[int]]:
    """Parse solver agent text outputs into structured actions.

    Each action is classified as one of:
      - tool_call: agent wants to execute tools
      - response: agent sends text to user
      - stop: agent wants to end conversation
      - invalid: cannot be parsed

    Returns:
        results: list of action dicts
        valids: list of 0/1 validity flags
    """
    results: List[Dict] = []
    valids: List[int] = []

    for action in actions:
        tool_call_matches = RE_TOOL_CALL.findall(action)
        response_matches = RE_RESPONSE.findall(action)

        # Both tool_call and response present → invalid
        if tool_call_matches and response_matches:
            results.append({"type": "invalid", "raw": action})
            valids.append(0)
            continue

        if tool_call_matches:
            raw_tc = tool_call_matches[-1].strip()
            parsed = _safe_json_loads(raw_tc)
            if parsed is not None:
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if isinstance(parsed, list):
                    # Normalize each call to {name, arguments}
                    calls = []
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            if isinstance(args, str):
                                args = _safe_json_loads(args) or {}
                            calls.append({
                                "name": item["name"],
                                "arguments": args if isinstance(args, dict) else {},
                            })
                    if calls:
                        results.append({"type": "tool_call", "calls": calls})
                        valids.append(1)
                        continue
            # Failed to parse tool call
            results.append({"type": "invalid", "raw": action})
            valids.append(0)
            continue

        if response_matches:
            text = response_matches[-1].strip()
            # Check for stop/transfer signals
            if any(sig in text for sig in ("[STOP]", "[TRANSFER]", "[OUT_OF_SCOPE]")):
                results.append({"type": "stop", "content": text})
                valids.append(1)
            else:
                results.append({"type": "response", "content": text})
                valids.append(1)
            continue

        # No recognized tags → treat as raw response (invalid format)
        results.append({"type": "response", "content": action})
        valids.append(0)

    return results, valids


def challenger_projection(actions: List[str]) -> Tuple[List[Dict], List[int]]:
    """Parse challenger text outputs into instructions + expected actions (TOD-Zero).

    Expected format:
        <instructions>Natural language briefing for the user simulator.</instructions>
        <actions>[{"name": "tool_name", "arguments": {...}}, ...]</actions>

    Returns:
        results: list of parsed challenger dicts or invalid markers
        valids: list of 0/1 validity flags
    """
    results: List[Dict] = []
    valids: List[int] = []

    for action in actions:
        instr_matches = RE_INSTRUCTIONS.findall(action)
        act_matches = RE_ACTIONS_TAG.findall(action)

        if instr_matches and act_matches:
            instructions = instr_matches[-1].strip()
            actions_str = act_matches[-1].strip()
            parsed_actions = _safe_json_loads(actions_str)

            if (
                len(instructions) >= 20
                and isinstance(parsed_actions, list)
                and len(parsed_actions) > 0
                and all(isinstance(a, dict) and "name" in a for a in parsed_actions)
            ):
                results.append({
                    "type": "challenger_output",
                    "instructions": instructions,
                    "actions": parsed_actions,
                })
                valids.append(1)
                continue

        results.append({"type": "invalid", "raw": action})
        valids.append(0)

    return results, valids
