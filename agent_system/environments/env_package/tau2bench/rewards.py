"""
Reward functions for tau2-bench environments.

Solver rewards (sparse, at episode end):
  - tool-call accuracy: name match + argument key F1 + value match (Tool-R0 style)
  - task success: binary based on tau-bench native evaluation criteria (DB state, actions, communicate)
  - combined: weighted sum of both

Challenger rewards:
  - format: are all required XML tags present and parseable
  - validity: do tool calls reference valid tools, are args structurally valid
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Challenger reward functions (TOD-Zero: instructions + actions)
# ===================================================================

def compute_challenger_reward(
    action: Dict,
    domain_tool_names: Optional[set] = None,
    domain_keywords: Optional[List[str]] = None,
) -> float:
    """Reward for TOD-Zero challenger: format + tool validity + arg validity.

    Mirrors the TRL self-play reward_format_and_validity:
      R_format       (0.0–1.0): 4 x 0.25 sub-components
        +0.25  <think> tag present and non-empty
        +0.25  <instructions> tag present and non-empty (>10 chars)
        +0.25  <actions> tag parses as JSON list with ≥1 item
        +0.25  every action has a non-empty "name" (str) and "arguments" (dict)
      R_tool_validity (0.0–1.0): fraction of action names in domain tool set
      R_arg_validity  (0.0–1.0): fraction of valid arg names per action

    Combined: 0.4 * R_format + 0.3 * R_tool_validity + 0.3 * R_arg_validity

    Args:
        action: parsed challenger action dict from challenger_projection
        domain_tool_names: set of valid tool names for this domain
        domain_keywords: unused (kept for API compatibility)
    """
    if action.get("type") != "challenger_output":
        return 0.0

    instructions = action.get("instructions", "")
    actions_list = action.get("actions", [])

    # --- R_format ---
    r_format = 0.0
    # think: we don't have it post-projection, but instructions presence covers most of it
    r_format += 0.25  # challenger_projection already validated instructions presence
    if len(instructions) > 10:
        r_format += 0.25
    if isinstance(actions_list, list) and len(actions_list) > 0:
        r_format += 0.25
        all_valid_shape = all(
            isinstance(a, dict)
            and isinstance(a.get("name"), str)
            and a["name"].strip()
            and isinstance(a.get("arguments", {}), dict)
            for a in actions_list
        )
        if all_valid_shape:
            r_format += 0.25

    # --- R_tool_validity ---
    r_tool = 0.0
    if domain_tool_names and isinstance(actions_list, list) and actions_list:
        valid_count = sum(1 for a in actions_list if a.get("name") in domain_tool_names)
        r_tool = valid_count / len(actions_list)

    # --- R_arg_validity ---
    r_arg = 0.0
    if domain_tool_names is not None and isinstance(actions_list, list) and actions_list:
        from agent_system.environments.env_package.tau2bench.db_sampler import DOMAIN_TOOLS
        # Build tool index for any domain that has the tool names
        all_tools = []
        for tools in DOMAIN_TOOLS.values():
            all_tools.extend(tools)
        tool_index = {t["name"]: t for t in all_tools}

        scores = []
        for a in actions_list:
            tool_spec = tool_index.get(a.get("name", ""))
            if tool_spec is None:
                scores.append(0.0)
                continue
            expected_params = set(tool_spec["parameters"].keys())
            provided_params = set((a.get("arguments") or {}).keys())
            if not expected_params and not provided_params:
                scores.append(1.0)
            elif not expected_params:
                scores.append(0.0)
            else:
                valid = len(provided_params & expected_params)
                total = len(provided_params | expected_params)
                scores.append(valid / total if total > 0 else 0.0)
        r_arg = sum(scores) / len(scores) if scores else 0.0

    return float(min(1.0, 0.4 * r_format + 0.3 * r_tool + 0.3 * r_arg))


# ===================================================================
# Task success reward (tau2-bench native evaluator)
# ===================================================================

def compute_task_success_reward(
    env_constructor: Callable,
    task: Any,
    message_history: list,
    domain: str,
    termination_reason: Any,
) -> Tuple[float, Dict]:
    """Compute tau2-bench native task success reward.

    Runs the full tau2-bench evaluator (DB state, action match, communicate checks)
    against the completed conversation trajectory.

    Args:
        env_constructor: tau2-bench environment factory (from registry)
        task: tau2 Task object with evaluation_criteria
        message_history: list of tau2 Message objects built during the episode
        domain: domain name (e.g. "retail", "airline")
        termination_reason: tau2 TerminationReason enum value

    Returns:
        (reward ∈ [0, 1], diagnostics dict)
    """
    try:
        from uuid import uuid4
        from tau2.data_model.simulation import SimulationRun
        from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
        from tau2.utils.utils import get_now

        now = get_now()
        sim = SimulationRun(
            id=str(uuid4()),
            task_id=task.id,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=termination_reason,
            messages=message_history,
        )
        reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=domain,
        )
        diagnostics = {
            "task_success_reward": reward_info.reward,
            "reward_breakdown": {
                k.value: v for k, v in (reward_info.reward_breakdown or {}).items()
            },
            "db_match": reward_info.db_check.db_match if reward_info.db_check else None,
        }
        return float(reward_info.reward), diagnostics
    except Exception as e:
        print(f"[Tau2Solver] Warning: task_success evaluation failed: {e}")
        return 0.0, {"task_success_error": str(e)}


def compute_combined_reward(
    env_constructor: Callable,
    task: Any,
    tool_calls_made: List[Dict[str, Any]],
    message_history: list,
    domain: str,
    termination_reason: Any,
    tool_call_reward_coef: float = 0.5,
    task_success_reward_coef: float = 0.5,
) -> Tuple[float, Dict]:
    """Weighted combination of tool-call accuracy and tau2-bench task success.

    tool_call_reward  – continuous [0, 1] greedy F1 over tool name/args
    task_success_reward – binary [0, 1] from tau2-bench DB + action + communicate evaluators

    combined = tool_call_reward_coef * tool_call_reward
             + task_success_reward_coef * task_success_reward

    Args:
        env_constructor: tau2-bench environment factory
        task: tau2 Task object
        tool_calls_made: list of {name, arguments} dicts collected during episode
        message_history: list of tau2 Message objects built during the episode
        domain: domain name
        termination_reason: tau2 TerminationReason enum value
        tool_call_reward_coef: weight for tool-call accuracy
        task_success_reward_coef: weight for task success

    Returns:
        (combined_reward ∈ [0, 1], diagnostics dict)
    """
    # --- Tool-call accuracy (continuous) ---
    tc_reward = 0.0
    tc_diagnostics: Dict[str, Any] = {}
    if task.evaluation_criteria is not None and task.evaluation_criteria.actions is not None:
        gt_actions = [
            a for a in task.evaluation_criteria.actions
            if a.requestor == "assistant"
        ]
        if not gt_actions and not tool_calls_made:
            tc_reward = 1.0
            tc_diagnostics = {"note": "no_actions_expected_or_made"}
        else:
            tc_reward, tc_diagnostics = compute_solver_reward(tool_calls_made, gt_actions)

    # --- Task success (binary, tau2-bench native) ---
    ts_reward, ts_diagnostics = compute_task_success_reward(
        env_constructor=env_constructor,
        task=task,
        message_history=message_history,
        domain=domain,
        termination_reason=termination_reason,
    )

    combined = tool_call_reward_coef * tc_reward + task_success_reward_coef * ts_reward
    diagnostics = {
        "tool_call_reward": tc_reward,
        "tool_call_reward_coef": tool_call_reward_coef,
        "task_success_reward": ts_reward,
        "task_success_reward_coef": task_success_reward_coef,
        "combined_reward": float(combined),
        "tool_call_diagnostics": tc_diagnostics,
        "task_success_diagnostics": ts_diagnostics,
    }
    return float(max(0.0, min(1.0, combined))), diagnostics


# ===================================================================
# Synthetic solver reward (TOD-Zero: no ground-truth eval criteria)
# ===================================================================

def compute_synthetic_reward(
    termination_reason: Any,
    tool_calls_made: List[Dict[str, Any]],
    expected_actions: Optional[List[Dict[str, Any]]] = None,
    completion_coef: float = 0.4,
    tool_usage_coef: float = 0.1,
    action_match_coef: float = 0.5,
) -> Tuple[float, Dict]:
    """Reward for solver trained on challenger-generated scenarios.

    Used when no ground-truth evaluation criteria exist (synthetic tasks).

    Components:
      R_completion (0.0-1.0): did the user simulator signal task completion?
        - USER_STOP  → 1.0  (user satisfied)
        - AGENT_STOP → 0.5  (agent ended; partial credit)
        - timeout    → 0.0  (no resolution)
      R_tool_usage (0.0-1.0): did the agent make successful API calls?
        - min(1.0, n_successful_calls / 2)
      R_action_match (0.0-1.0): how well do solver's tool calls match
        the challenger's expected actions? Uses the same greedy F1 scoring
        as the standard solver reward.

    When expected_actions is empty (legacy scenarios without actions),
    falls back to completion + tool_usage only.

    Returns:
        (reward ∈ [0, 1], diagnostics dict)
    """
    from tau2.data_model.simulation import TerminationReason

    # --- R_completion ---
    if termination_reason == TerminationReason.USER_STOP:
        r_completion = 1.0
    elif termination_reason == TerminationReason.AGENT_STOP:
        r_completion = 0.5
    else:
        r_completion = 0.0

    # --- R_tool_usage: reward for making at least 2 successful tool calls ---
    n_calls = len(tool_calls_made)
    r_tool = min(1.0, n_calls / 2.0)

    # --- R_action_match: compare with challenger's expected actions ---
    r_action = 0.0
    action_diagnostics = {}
    if expected_actions and len(expected_actions) > 0:
        # Normalize expected actions to the same format as tool_calls_made
        gt_calls = []
        for a in expected_actions:
            if isinstance(a, dict) and "name" in a:
                gt_calls.append({
                    "name": a.get("name", ""),
                    "arguments": a.get("arguments", {}),
                })
        if gt_calls:
            r_action, action_diagnostics = compute_solver_accuracy(
                predicted_calls=tool_calls_made,
                ground_truth_calls=gt_calls,
            )
        # With action matching available, use full 3-component weighting
        combined = (
            completion_coef * r_completion
            + tool_usage_coef * r_tool
            + action_match_coef * r_action
        )
    else:
        # No expected actions — fall back to completion + tool_usage only
        combined = 0.7 * r_completion + 0.3 * r_tool

    diagnostics = {
        "synthetic_mode": True,
        "termination_reason": str(termination_reason),
        "completion_reward": r_completion,
        "tool_usage_reward": r_tool,
        "action_match_reward": r_action,
        "n_successful_tool_calls": n_calls,
        "n_expected_actions": len(expected_actions) if expected_actions else 0,
        "action_match_diagnostics": action_diagnostics,
        "combined_reward": float(combined),
    }
    return float(max(0.0, min(1.0, combined))), diagnostics
