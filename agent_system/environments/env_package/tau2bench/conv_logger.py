"""
Conversation logger for tau2-bench solver episodes.

Stores complete episode traces (agent inputs, outputs, user sim exchanges,
tool calls, rewards) as JSON files for post-training debugging.

Usage:
    logger = ConversationLogger(log_dir="./conv_logs", max_episodes=500)

    # In _SolverWorker.reset():
    logger.start_episode(episode_id, metadata)

    # In _SolverWorker.step():
    logger.log_turn(...)

    # At episode end:
    logger.end_episode(reward, diagnostics)

Output format: one JSON file per episode at {log_dir}/ep_{episode_id}.json
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConversationLogger:
    """Logs complete solver episode conversations to JSON files."""

    def __init__(
        self,
        log_dir: str = "./conv_logs",
        max_episodes: int = 999999,
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """
        Args:
            log_dir: Directory to write JSON files.
            max_episodes: Stop logging after this many episodes (disk safety).
            enabled: Master switch. Set False to disable all logging.
            sample_rate: Fraction of episodes to log (1.0 = all, 0.1 = 10%).
        """
        self.log_dir = Path(log_dir)
        self.max_episodes = max_episodes
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._episode_count = 0
        self._current: Optional[Dict] = None
        self._rng = __import__("random").Random(42)

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(
        self,
        episode_id: str,
        task_id: str,
        domain: str,
        synthetic_mode: bool,
        user_instructions: str,
        expected_actions: List[Dict],
        policy_snippet: str = "",
        tool_names: Optional[List[str]] = None,
    ):
        """Call at the start of each episode (in _SolverWorker.reset)."""
        if not self.enabled:
            return
        if self._episode_count >= self.max_episodes:
            return
        if self._rng.random() > self.sample_rate:
            self._current = None
            return

        self._current = {
            "episode_id": episode_id,
            "task_id": task_id,
            "domain": domain,
            "synthetic_mode": synthetic_mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),

            # --- INPUTS ---
            "user_simulator_input": {
                "instructions": user_instructions,
                "description": "This is the scenario text given to the user simulator's system prompt. It defines who the customer is, what they want, and what they know.",
            },
            "agent_input": {
                "policy_snippet": policy_snippet[:500] if policy_snippet else "",
                "tool_names": tool_names or [],
                "description": "The agent receives the full policy and tool schemas in its system prompt. Only a snippet is logged here.",
            },
            "expected_actions": expected_actions,

            # --- CONVERSATION ---
            "turns": [],

            # --- OUTCOME (filled at end) ---
            "outcome": None,
        }

    def log_turn(
        self,
        turn_number: int,
        agent_input_observation: str,
        agent_raw_output: str,
        parsed_action_type: str,
        parsed_action_detail: Any = None,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        user_sim_reply: Optional[str] = None,
        is_done: bool = False,
        step_reward: float = 0.0,
    ):
        """Call after each step (in _SolverWorker.step handlers)."""
        if self._current is None:
            return

        turn = {
            "turn": turn_number,

            # What the agent saw as input this turn
            "agent_saw": {
                "observation": agent_input_observation[:2000],
                "description": "The text observation the agent received (user message or tool result).",
            },

            # What the agent produced
            "agent_produced": {
                "raw_output": agent_raw_output[:2000],
                "parsed_type": parsed_action_type,
                "description": "The agent's raw text generation and what it was parsed as.",
            },

            # Tool call details (if applicable)
            "tool_execution": None,

            # User simulator exchange (if applicable)
            "user_sim_exchange": None,

            "step_reward": step_reward,
            "is_done": is_done,
        }

        if parsed_action_type == "tool_call" and tool_calls is not None:
            turn["tool_execution"] = {
                "calls": [
                    {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                    for tc in tool_calls
                ],
                "results": [
                    {
                        "name": tr.get("name", ""),
                        "result": str(tr.get("result", ""))[:1000],
                        "error": tr.get("error", False),
                    }
                    for tr in (tool_results or [])
                ],
                "description": "Tool calls made by the agent and their execution results from the tau2-bench environment.",
            }

        if parsed_action_type in ("response", "stop") and user_sim_reply is not None:
            turn["user_sim_exchange"] = {
                "agent_said_to_user": parsed_action_detail[:1000] if parsed_action_detail else "",
                "user_replied": user_sim_reply[:1000],
                "description": "The agent's text message sent to the user simulator and the user's reply.",
            }

        self._current["turns"].append(turn)

    def end_episode(
        self,
        final_reward: float,
        termination_reason: str,
        diagnostics: Optional[Dict] = None,
        tool_calls_made: Optional[List[Dict]] = None,
    ):
        """Call at episode end (after reward computation)."""
        if self._current is None:
            return

        self._current["outcome"] = {
            "final_reward": final_reward,
            "termination_reason": termination_reason,
            "num_turns": len(self._current["turns"]),
            "num_tool_calls": len(tool_calls_made) if tool_calls_made else 0,
            "tool_calls_made": [
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                for tc in (tool_calls_made or [])
            ],
            "reward_diagnostics": _safe_serialize(diagnostics),
        }

        # Write to file
        ep_id = self._current["episode_id"]
        filepath = self.log_dir / f"ep_{ep_id}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(self._current, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[ConvLogger] Warning: failed to write {filepath}: {e}")

        self._episode_count += 1
        self._current = None


def _safe_serialize(obj):
    """Convert non-serializable objects to strings."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)):
        return obj
    return str(obj)
