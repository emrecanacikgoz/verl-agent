"""
Parallel environment wrappers for tau2-bench integration with verl-agent.

Provides:
  - Tau2BenchSolverEnvs: multi-turn solver environment (agent interacts with user sim + tools)
  - Tau2BenchChallengerEnvs: single-step challenger environment (generates task specs)
"""

import json
import concurrent.futures
import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

try:
    import gym
except ImportError:
    import gymnasium as gym
import numpy as np


# ===================================================================
# Single solver worker
# ===================================================================

class _SolverWorker:
    """Manages a single solver environment instance.

    Wraps a tau2-bench Environment + UserSimulator to provide
    a simple reset/step interface.
    """

    def __init__(self, domain: str, user_sim_url: str, user_sim_model: str, seed: int = 0):
        self.domain = domain
        self.user_sim_url = user_sim_url
        self.user_sim_model = user_sim_model
        self.rng = np.random.RandomState(seed)

        # Load domain from tau2-bench registry
        from tau2.registry import registry
        self.env_constructor = registry.get_env_constructor(domain)
        self.tasks = registry.get_tasks_loader(domain)()

        # Runtime state
        self.env = None
        self.task = None
        self.user_sim = None
        self.tool_calls_made: List[Dict] = []
        self.step_count = 0
        self.policy = ""
        self.tool_schemas: List[Dict] = []

    def reset(self, kwargs: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Reset with a (possibly random) task."""
        # Select task
        if kwargs and "task_idx" in kwargs:
            task_idx = kwargs["task_idx"]
        else:
            task_idx = self.rng.randint(0, len(self.tasks))
        self.task = self.tasks[task_idx % len(self.tasks)]

        # Create fresh tau2-bench environment
        self.env = self.env_constructor()
        if self.task.initial_state is not None:
            try:
                self.env.set_state(
                    initialization_data=self.task.initial_state.initialization_data,
                    initialization_actions=self.task.initial_state.initialization_actions,
                    message_history=self.task.initial_state.message_history or [],
                )
            except Exception as e:
                print(f"[Tau2Solver] Warning: set_state failed for task {self.task.id}: {e}")

        # Get domain info
        tools = self.env.get_tools()
        self.tool_schemas = [t.openai_schema for t in tools]
        self.policy = self.env.get_policy()

        # Reset state
        self.tool_calls_made = []
        self.step_count = 0

        # Initialize user simulator
        from agent_system.environments.env_package.tau2bench.user_sim import LightUserSimulator
        user_instructions = str(self.task.user_scenario.instructions)
        self.user_sim = LightUserSimulator(
            api_url=self.user_sim_url,
            model=self.user_sim_model,
            user_instructions=user_instructions,
        )

        # Generate first user message
        first_msg = self.user_sim.generate_first_message()

        info = {
            "task_id": self.task.id,
            "domain": self.domain,
            "data_source": f"tau2bench_{self.domain}",
        }
        return first_msg, info

    def step(self, parsed_action: Dict) -> Tuple[str, float, bool, Dict]:
        """Process one parsed action from the agent.

        Args:
            parsed_action: dict from solver_projection with type + data

        Returns:
            (observation, reward, done, info)
        """
        self.step_count += 1
        action_type = parsed_action.get("type", "invalid")

        if action_type == "tool_call":
            return self._handle_tool_calls(parsed_action["calls"])
        elif action_type == "response":
            return self._handle_response(parsed_action["content"])
        elif action_type == "stop":
            return self._handle_stop(parsed_action.get("content", ""))
        else:
            # Invalid action
            return (
                "Error: Could not parse your response. Use <tool_call> or <response> tags.",
                0.0,
                False,
                {"is_action_valid": 0, "action_type": "invalid"},
            )

    def _handle_tool_calls(self, calls: List[Dict]) -> Tuple[str, float, bool, Dict]:
        """Execute tool calls via tau2-bench environment."""
        tool_results = []
        for tc in calls:
            self.tool_calls_made.append(tc)
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            try:
                result = self.env.make_tool_call(name, requestor="assistant", **args)
                result_str = self.env.to_json_str(result)
            except Exception as e:
                result_str = f"Error: {str(e)}"
            tool_results.append({"name": name, "result": result_str})

        # Sync tools after execution
        self.env.sync_tools()

        # Format tool results as observation
        obs_parts = []
        for tr in tool_results:
            obs_parts.append(f"Tool '{tr['name']}' returned: {tr['result']}")
        obs = "\n".join(obs_parts)

        return obs, 0.0, False, {"action_type": "tool_call", "is_action_valid": 1}

    def _handle_response(self, content: str) -> Tuple[str, float, bool, Dict]:
        """Send text response to user simulator, get reply."""
        user_reply = self.user_sim.respond(content)

        if self.user_sim.is_stop(user_reply):
            reward, diagnostics = self._compute_final_reward()
            return user_reply, reward, True, {
                "won": reward >= 0.99,
                "action_type": "user_stop",
                "reward_diagnostics": diagnostics,
                "is_action_valid": 1,
            }

        return user_reply, 0.0, False, {"action_type": "response", "is_action_valid": 1}

    def _handle_stop(self, content: str) -> Tuple[str, float, bool, Dict]:
        """Agent signals end of conversation."""
        reward, diagnostics = self._compute_final_reward()
        return "", reward, True, {
            "won": reward >= 0.99,
            "action_type": "stop",
            "reward_diagnostics": diagnostics,
            "is_action_valid": 1,
        }

    def _compute_final_reward(self) -> Tuple[float, Dict]:
        """Compute sparse episode reward based on tool-call accuracy."""
        if self.task.evaluation_criteria is None:
            return 0.0, {}
        if self.task.evaluation_criteria.actions is None:
            return 0.0, {}

        # Filter to assistant actions
        gt_actions = [
            a for a in self.task.evaluation_criteria.actions
            if a.requestor == "assistant"
        ]

        if not gt_actions and not self.tool_calls_made:
            return 1.0, {"note": "no_actions_expected_or_made"}

        from agent_system.environments.env_package.tau2bench.rewards import compute_solver_reward
        reward, diagnostics = compute_solver_reward(
            self.tool_calls_made, gt_actions
        )
        return reward, diagnostics


# ===================================================================
# Single challenger worker
# ===================================================================

class _ChallengerWorker:
    """Manages a single challenger environment instance."""

    def __init__(self, domain: str, seed: int = 0):
        self.domain = domain
        self.rng = np.random.RandomState(seed)

        # Load domain info from tau2-bench
        from tau2.registry import registry
        env = registry.get_env_constructor(domain)()
        self.policy = env.get_policy()
        tools = env.get_tools()
        self.tool_schemas = [t.openai_schema for t in tools]
        self.tool_names = {t.name for t in tools}

    def reset(self, kwargs: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Return domain info as observation."""
        obs = self._format_domain_info()
        info = {
            "domain": self.domain,
            "data_source": f"tau2bench_challenger_{self.domain}",
        }
        return obs, info

    def step(self, parsed_action: Dict) -> Tuple[str, float, bool, Dict]:
        """Validate challenger output and compute reward."""
        from agent_system.environments.env_package.tau2bench.rewards import compute_challenger_reward
        reward = compute_challenger_reward(parsed_action, self.tool_names)
        return "", reward, True, {
            "won": reward >= 0.8,
            "reward": reward,
            "is_action_valid": 1 if parsed_action.get("type") == "task_spec" else 0,
        }

    def _format_domain_info(self) -> str:
        """Format domain policy and tool schemas for the challenger."""
        tools_json = json.dumps(self.tool_schemas, indent=2)
        return (
            f"Domain: {self.domain}\n\n"
            f"Policy:\n{self.policy}\n\n"
            f"Available Tools:\n{tools_json}"
        )


# ===================================================================
# Parallel environment wrappers
# ===================================================================

class Tau2BenchSolverEnvs(gym.Env):
    """Parallelized solver environments using ThreadPoolExecutor."""

    def __init__(
        self,
        domain: str,
        user_sim_url: str,
        user_sim_model: str,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
    ):
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.workers = [
            _SolverWorker(
                domain=domain,
                user_sim_url=user_sim_url,
                user_sim_model=user_sim_model,
                seed=seed + i,
            )
            for i in range(self.batch_size)
        ]

        max_workers = min(self.batch_size, 64)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()

    def reset(self, kwargs=None):
        """Reset all environments.

        Args:
            kwargs: optional list of dicts (one per env) with task_idx etc.
        """
        if kwargs is None:
            kwargs = [None] * self.batch_size
        elif len(kwargs) < self.batch_size:
            kwargs = list(kwargs) + [None] * (self.batch_size - len(kwargs))

        tasks = [
            self._loop.run_in_executor(self._executor, w.reset, kw)
            for w, kw in zip(self.workers, kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, parsed_actions: List[Dict]):
        """Step all environments with parsed actions.

        Args:
            parsed_actions: list of action dicts from solver_projection
        """
        if len(parsed_actions) < self.batch_size:
            parsed_actions = list(parsed_actions) + [{"type": "invalid"}] * (
                self.batch_size - len(parsed_actions)
            )

        tasks = [
            self._loop.run_in_executor(self._executor, w.step, a)
            for w, a in zip(self.workers, parsed_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


class Tau2BenchChallengerEnvs(gym.Env):
    """Parallelized challenger environments."""

    def __init__(
        self,
        domain: str,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
    ):
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.workers = [
            _ChallengerWorker(domain=domain, seed=seed + i)
            for i in range(self.batch_size)
        ]

        max_workers = min(self.batch_size, 64)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()

    def reset(self, kwargs=None):
        if kwargs is None:
            kwargs = [None] * self.batch_size
        elif len(kwargs) < self.batch_size:
            kwargs = list(kwargs) + [None] * (self.batch_size - len(kwargs))

        tasks = [
            self._loop.run_in_executor(self._executor, w.reset, kw)
            for w, kw in zip(self.workers, kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, parsed_actions: List[Dict]):
        if len(parsed_actions) < self.batch_size:
            parsed_actions = list(parsed_actions) + [{"type": "invalid"}] * (
                self.batch_size - len(parsed_actions)
            )

        tasks = [
            self._loop.run_in_executor(self._executor, w.step, a)
            for w, a in zip(self.workers, parsed_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


# ===================================================================
# Builder functions
# ===================================================================

def build_tau2bench_solver_envs(
    domain: str,
    user_sim_url: str,
    user_sim_model: str,
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    **kwargs,
) -> Tau2BenchSolverEnvs:
    return Tau2BenchSolverEnvs(
        domain=domain,
        user_sim_url=user_sim_url,
        user_sim_model=user_sim_model,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )


def build_tau2bench_challenger_envs(
    domain: str,
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    **kwargs,
) -> Tau2BenchChallengerEnvs:
    return Tau2BenchChallengerEnvs(
        domain=domain,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )
