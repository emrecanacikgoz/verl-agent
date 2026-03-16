"""
Reward functions for tau2-bench environments.

Solver rewards (sparse, at episode end):
  - tool-call accuracy: name match + argument key F1 + value match (Tool-R0 style)
  - task success: binary based on tau-bench evaluation criteria

Challenger rewards:
  - format: are all required XML tags present and parseable
  - validity: do tool calls reference valid tools, are args structurally valid
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

def _safe_json_loads(s: str):
    """Best-effort JSON parse."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Constants (aligned with Tool-R0 solver rewards)
# ---------------------------------------------------------------------------
LAMBDA_NAME = 0.2
LAMBDA_PARAM_NAMES = 0.3
LAMBDA_PARAM_VALUES = 0.5
EXTRA_CALL_PENALTY_ALPHA = 0.25

# Challenger reward weights
W_FORMAT = 0.5
W_VALIDITY = 0.5


# ===================================================================
# Shared utilities
# ===================================================================

def normalize_tool_call(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize a raw object into {name: str, arguments: dict}."""
    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    # OpenAI function-call wrapper
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _safe_json_loads(args)
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args}

    name = obj.get("name") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        return None

    args = obj.get("arguments")
    if isinstance(args, str):
        args = _safe_json_loads(args)
    if isinstance(args, dict):
        return {"name": name, "arguments": args}

    # Flat dict: all non-name keys are arguments
    flat = {k: v for k, v in obj.items() if k not in ("name", "tool_name")}
    return {"name": name, "arguments": flat}


_NUM_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def _coerce_number(x: Any) -> Optional[float]:
    """Safely coerce numeric-like values to float."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return float(x)
        except (OverflowError, ValueError):
            return None
    if isinstance(x, str) and _NUM_RE.match(x):
        s = x.strip()
        if len(s.lstrip("+-").replace(".", "")) > 15:
            return None
        try:
            return float(s)
        except (OverflowError, ValueError):
            return None
    return None


def _canonical_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def robust_value_match(v1: Any, v2: Any) -> bool:
    """Compare two values using semantics-aware heuristic."""
    if v1 == v2:
        return True
    # int/string numeric comparison
    if isinstance(v1, (int, str)) and isinstance(v2, (int, str)):
        s1, s2 = str(v1).strip(), str(v2).strip()
        if _NUM_RE.match(s1) and _NUM_RE.match(s2):
            return s1.lstrip("+") == s2.lstrip("+")
    # Numeric coercion
    n1, n2 = _coerce_number(v1), _coerce_number(v2)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) < 1e-9
    # String whitespace normalization
    if isinstance(v1, str) and isinstance(v2, str):
        return " ".join(v1.strip().split()) == " ".join(v2.strip().split())
    return _canonical_json(v1) == _canonical_json(v2)


def f1_keys(pred_keys: set, gt_keys: set) -> float:
    """F1 score between two sets of argument keys."""
    if not pred_keys and not gt_keys:
        return 1.0
    if not pred_keys or not gt_keys:
        return 0.0
    inter = len(pred_keys & gt_keys)
    if inter == 0:
        return 0.0
    prec = inter / len(pred_keys)
    rec = inter / len(gt_keys)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


# ===================================================================
# Solver reward functions
# ===================================================================

def score_tool_call(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Score a single predicted tool call against ground truth.

    Returns (name_score, key_score, value_score) each in [0, 1].
    """
    pred_name = (predicted.get("name") or "").strip()
    gt_name = (ground_truth.get("name") or "").strip()
    name_score = 1.0 if pred_name == gt_name and pred_name != "" else 0.0

    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gt_args, dict):
        gt_args = {}

    pred_keys = set(pred_args.keys())
    gt_keys = set(gt_args.keys())
    key_score = f1_keys(pred_keys, gt_keys)

    inter = pred_keys & gt_keys
    if not inter:
        value_score = 1.0 if (not pred_keys and not gt_keys) else 0.0
    else:
        matches = sum(
            1 for k in inter if robust_value_match(pred_args.get(k), gt_args.get(k))
        )
        value_score = matches / len(inter)

    return name_score, key_score, value_score


def compute_solver_accuracy(
    predicted_calls: List[Dict[str, Any]],
    ground_truth_calls: List[Dict[str, Any]],
) -> Tuple[float, Dict]:
    """Compute accuracy reward for solver's tool calls vs ground truth.

    Uses greedy matching: for each GT call, find the best-matching predicted call.
    Returns (final_reward, diagnostics).
    """
    diagnostics = {
        "mean_name_score": 0.0,
        "mean_key_score": 0.0,
        "mean_value_score": 0.0,
        "num_pred_calls": len(predicted_calls),
        "num_gt_calls": len(ground_truth_calls),
        "extra_calls": 0,
        "extra_call_penalty": 1.0,
        "base_score": 0.0,
    }

    if not ground_truth_calls:
        final = 1.0 if not predicted_calls else 0.0
        diagnostics["base_score"] = final
        return final, diagnostics

    if not predicted_calls:
        return 0.0, diagnostics

    used_pred = set()
    total = 0.0
    name_scores, key_scores, value_scores = [], [], []

    for gt in ground_truth_calls:
        best_r, best_i, best_tuple = 0.0, -1, (0.0, 0.0, 0.0)
        for i, pred in enumerate(predicted_calls):
            if i in used_pred:
                continue
            n_s, k_s, v_s = score_tool_call(pred, gt)
            r = LAMBDA_NAME * n_s + LAMBDA_PARAM_NAMES * k_s + LAMBDA_PARAM_VALUES * v_s
            if r > best_r:
                best_r, best_i, best_tuple = r, i, (n_s, k_s, v_s)
        if best_i != -1:
            used_pred.add(best_i)
            total += best_r
            name_scores.append(best_tuple[0])
            key_scores.append(best_tuple[1])
            value_scores.append(best_tuple[2])

    base = total / len(ground_truth_calls)
    extra = max(0, len(predicted_calls) - len(ground_truth_calls))
    penalty = 1.0 / (1.0 + EXTRA_CALL_PENALTY_ALPHA * extra) if extra > 0 else 1.0
    final = base * penalty

    diagnostics.update({
        "mean_name_score": float(sum(name_scores) / max(1, len(name_scores))),
        "mean_key_score": float(sum(key_scores) / max(1, len(key_scores))),
        "mean_value_score": float(sum(value_scores) / max(1, len(value_scores))),
        "base_score": float(base),
        "extra_calls": int(extra),
        "extra_call_penalty": float(penalty),
    })

    return float(max(0.0, min(1.0, final))), diagnostics


def compute_solver_reward(
    tool_calls_made: List[Dict[str, Any]],
    ground_truth_actions: List,
) -> Tuple[float, Dict]:
    """Compute the full solver episode reward.

    Compares all tool calls the agent made during the episode against
    the ground truth actions from tau-bench evaluation criteria.

    Args:
        tool_calls_made: list of {name, arguments} dicts from agent trajectory
        ground_truth_actions: list of tau2 Action objects (or dicts with name, arguments, compare_args)

    Returns:
        (reward, diagnostics) where reward ∈ [0, 1]
    """
    # Convert ground truth actions to normalized dicts
    gt_calls = []
    for action in ground_truth_actions:
        if hasattr(action, "name"):
            # tau2 Action object
            gt_dict = {"name": action.name, "arguments": dict(action.arguments)}
            # If compare_args specified, only keep those args
            if hasattr(action, "compare_args") and action.compare_args is not None:
                gt_dict["arguments"] = {
                    k: v
                    for k, v in gt_dict["arguments"].items()
                    if k in action.compare_args
                }
        elif isinstance(action, dict):
            gt_dict = {
                "name": action.get("name", ""),
                "arguments": action.get("arguments", {}),
            }
        else:
            continue
        gt_calls.append(gt_dict)

    return compute_solver_accuracy(tool_calls_made, gt_calls)


# ===================================================================
# Challenger reward functions
# ===================================================================

def compute_challenger_format_reward(action: Dict) -> float:
    """Format reward for challenger output.

    Graded:
      +0.33 if <question> tag present and non-empty
      +0.33 if <available_tools> parses as valid JSON list
      +0.34 if <tool_call_answer> parses and normalizes
    """
    if action.get("type") != "task_spec":
        return 0.0

    reward = 0.0

    # Question present
    question = action.get("question", "")
    if question and len(question.strip()) > 5:
        reward += 0.33

    # Available tools parse as JSON list
    tools_text = action.get("available_tools", "")
    tools = _safe_json_loads(tools_text)
    if isinstance(tools, list) and len(tools) > 0:
        # Check each tool has a name
        if all(isinstance(t, dict) and "name" in t for t in tools):
            reward += 0.33

    # Tool call answer parses and normalizes
    answer_text = action.get("tool_call_answer", "")
    answer = _safe_json_loads(answer_text)
    if answer is not None:
        norm = normalize_tool_call(answer)
        if norm is not None:
            reward += 0.34

    return min(1.0, reward)


def compute_challenger_validity_reward(
    action: Dict,
    domain_tool_names: Optional[set] = None,
) -> float:
    """Validity reward for challenger output.

    Checks:
      +0.4 if gold tool name exists in available_tools
      +0.4 if gold arguments match tool schema (required params present)
      +0.2 if argument values are grounded (non-empty, non-placeholder)
    """
    if action.get("type") != "task_spec":
        return 0.0

    tools_text = action.get("available_tools", "")
    answer_text = action.get("tool_call_answer", "")

    tools = _safe_json_loads(tools_text)
    if not isinstance(tools, list) or len(tools) == 0:
        return 0.0

    answer_obj = _safe_json_loads(answer_text)
    gold = normalize_tool_call(answer_obj)
    if gold is None:
        return 0.0

    tool_index = {t["name"]: t for t in tools if isinstance(t, dict) and "name" in t}
    reward = 0.0

    # Gold tool name exists in available_tools
    gold_name = gold["name"]
    if gold_name in tool_index:
        reward += 0.4

        # Check arguments against schema
        tool_spec = tool_index[gold_name]
        schema = tool_spec.get("parameters")
        if schema is not None and isinstance(schema, dict):
            required = schema.get("required", [])
            if isinstance(required, list):
                gold_args = gold.get("arguments", {})
                if all(k in gold_args for k in required):
                    reward += 0.4
                else:
                    # Partial credit
                    if required:
                        reward += 0.4 * sum(1 for k in required if k in gold_args) / len(required)
            else:
                reward += 0.4  # No required fields specified
        else:
            reward += 0.4  # No schema to validate against

        # Check argument values are non-trivial
        gold_args = gold.get("arguments", {})
        if gold_args:
            non_empty = sum(
                1 for v in gold_args.values()
                if v is not None and str(v).strip() != ""
            )
            reward += 0.2 * (non_empty / len(gold_args))
        else:
            reward += 0.2

    # Optionally check against domain tools
    if domain_tool_names is not None and gold_name not in domain_tool_names:
        reward *= 0.5  # Penalty for using tools not in domain

    return min(1.0, reward)


def compute_challenger_reward(
    action: Dict,
    domain_tool_names: Optional[set] = None,
) -> float:
    """Combined challenger reward (format + validity)."""
    fmt = compute_challenger_format_reward(action)
    val = compute_challenger_validity_reward(action, domain_tool_names)
    return W_FORMAT * fmt + W_VALIDITY * val
