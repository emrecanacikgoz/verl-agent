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

    TOD-Zero mode: when external_scenarios is provided, uses challenger-generated
    user goals instead of tau2-bench registry task scenarios.  The tau2-bench
    environment is still used for tool execution (provides a real DB state).
    Reward uses user simulator satisfaction rather than ground-truth eval criteria.
    """

    def __init__(
        self,
        domain: str,
        user_sim_url: str,
        user_sim_model: str,
        seed: int = 0,
        tool_call_reward_coef: float = 0.5,
        task_success_reward_coef: float = 0.5,
        external_scenarios: Optional[List[Dict]] = None,
        conv_log_dir: Optional[str] = None,
        task_split: Optional[str] = "base",
        max_steps: int = 30,
    ):
        self.domain = domain
        self.user_sim_url = user_sim_url
        self.user_sim_model = user_sim_model
        self.tool_call_reward_coef = tool_call_reward_coef
        self.task_success_reward_coef = task_success_reward_coef
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        # TOD-Zero: challenger-generated user goals (None = use registry tasks)
        self.external_scenarios = external_scenarios
        self.synthetic_mode = external_scenarios is not None and len(external_scenarios) > 0

        # Conversation logger for post-training debugging
        from agent_system.environments.env_package.tau2bench.conv_logger import ConversationLogger
        self.conv_logger = ConversationLogger(
            log_dir=conv_log_dir or "./conv_logs",
            enabled=conv_log_dir is not None,
            sample_rate=0.05,  # log 5% of episodes to limit I/O
        )

        # Load domain from tau2-bench registry
        from tau2.registry import registry
        self.env_constructor = registry.get_env_constructor(domain)
        self.tasks = registry.get_tasks_loader(domain)(task_split)

        # Runtime state
        self.env = None
        self.task = None
        self.tool_calls_made: List[Dict] = []
        self.step_count = 0
        self.policy = ""
        self.tool_schemas: List[Dict] = []
        # Full tau2-bench message history for native evaluation
        self.message_history: list = []
        self._tool_call_counter: int = 0
        self._termination_reason = None
        # Challenger expected actions for synthetic reward (set on reset)
        self._expected_actions: List[Dict] = []

        # Verify user sim server is reachable before starting any rollouts
        from agent_system.environments.env_package.tau2bench.user_sim import (
            LightUserSimulator, check_user_sim_connection,
        )
        if not check_user_sim_connection(self.user_sim_url):
            raise RuntimeError(
                f"[Tau2Solver] FATAL: User simulator server is not reachable at "
                f"{self.user_sim_url}.\n"
                f"Start it first:\n"
                f"  python -m vllm.entrypoints.openai.api_server \\\n"
                f"      --model <user_sim_model> --port 8000 &\n"
                f"Then wait for it to be healthy before launching training."
            )

        # Create user simulator once; reuse across resets to avoid repeated HTTP discovery calls
        self.user_sim = LightUserSimulator(
            api_url=self.user_sim_url,
            model=self.user_sim_model,
            user_instructions="",  # will be set properly on first reset
        )

    def reset(self, kwargs: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Reset with a (possibly random) task.

        In synthetic mode (TOD-Zero), uses a challenger-generated user goal
        as the user scenario. The tau2-bench env is still initialized from
        a registry task so the DB has realistic data for tool calls.
        """
        from tau2.data_model.message import UserMessage as TauUserMessage

        # Select task from registry (provides DB state and tools)
        if kwargs and "task_idx" in kwargs:
            task_idx = kwargs["task_idx"]
        else:
            task_idx = self.rng.randint(0, len(self.tasks))
        self.task = self.tasks[task_idx % len(self.tasks)]

        # Create fresh tau2-bench environment (loads base db.json state)
        self.env = self.env_constructor()

        if not self.synthetic_mode and self.task.initial_state is not None:
            # Standard mode: apply task-specific initialization to set up DB state.
            # In synthetic mode we skip this — initialization_actions from registry
            # tasks would overwrite entities (cancel reservations, modify orders)
            # that the challenger scenario was grounded in, breaking the episode.
            # The base db.json already contains all users/reservations the challenger
            # sampled from, so no initialization is needed.
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

        # Reset episode state
        self.tool_calls_made = []
        self.step_count = 0
        self._tool_call_counter = 0
        self._termination_reason = None

        # Seed message_history with the task's initial state messages (required by evaluator)
        initial_msgs = []
        if self.task.initial_state is not None and self.task.initial_state.message_history:
            initial_msgs = list(self.task.initial_state.message_history)
        self.message_history = initial_msgs

        # --- User scenario: challenger-generated goal OR registry task scenario ---
        if self.synthetic_mode:
            # TOD-Zero: sample a challenger-generated user goal
            goal_idx = self.rng.randint(0, len(self.external_scenarios))
            scenario = self.external_scenarios[goal_idx]
            if isinstance(scenario, dict):
                user_instructions = scenario.get("instructions", "")
                self._expected_actions = scenario.get("actions", [])
            else:
                user_instructions = str(scenario)
                self._expected_actions = []
        else:
            user_instructions = str(self.task.user_scenario.instructions)
            self._expected_actions = []

        # Reset user simulator with the scenario instructions
        self.user_sim.reset(user_instructions)

        # Generate first user message and record it
        first_msg = self.user_sim.generate_first_message()
        self.message_history.append(TauUserMessage(role="user", content=first_msg))

        # Start conversation logging
        import uuid as _uuid
        self._episode_id = str(_uuid.uuid4())[:8]
        self._user_instructions = user_instructions
        self._last_obs = first_msg  # track what the agent will see next
        self.conv_logger.start_episode(
            episode_id=self._episode_id,
            task_id=self.task.id,
            domain=self.domain,
            synthetic_mode=self.synthetic_mode,
            user_instructions=user_instructions,
            expected_actions=self._expected_actions,
            policy_snippet=self.policy[:500],
            tool_names=[t.get("name", "") for t in self.tool_schemas[:20]],
        )

        info = {
            "task_id": self.task.id,
            "domain": self.domain,
            "data_source": f"tau2bench_{self.domain}",
            "synthetic_mode": self.synthetic_mode,
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
        agent_obs = getattr(self, '_last_obs', '')  # what the agent saw this turn

        if action_type == "tool_call":
            obs, reward, done, info = self._handle_tool_calls(parsed_action["calls"])
            # Log turn
            self.conv_logger.log_turn(
                turn_number=self.step_count,
                agent_input_observation=agent_obs,
                agent_raw_output=f"[tool_call] {json.dumps(parsed_action['calls'], default=str)[:500]}",
                parsed_action_type="tool_call",
                tool_calls=parsed_action["calls"],
                tool_results=[
                    {"name": tc.get("name", ""), "result": r, "error": False}
                    for tc, r in zip(parsed_action["calls"], obs.split("\n"))
                ] if obs else [],
                is_done=done,
                step_reward=reward,
            )
            self._last_obs = obs

        elif action_type == "response":
            obs, reward, done, info = self._handle_response(parsed_action["content"])
            self.conv_logger.log_turn(
                turn_number=self.step_count,
                agent_input_observation=agent_obs,
                agent_raw_output=parsed_action["content"],
                parsed_action_type="response",
                parsed_action_detail=parsed_action["content"],
                user_sim_reply=obs,
                is_done=done,
                step_reward=reward,
            )
            self._last_obs = obs

        elif action_type == "stop":
            obs, reward, done, info = self._handle_stop(parsed_action.get("content", ""))
            self.conv_logger.log_turn(
                turn_number=self.step_count,
                agent_input_observation=agent_obs,
                agent_raw_output=parsed_action.get("content", "[STOP]"),
                parsed_action_type="stop",
                parsed_action_detail=parsed_action.get("content", ""),
                is_done=True,
                step_reward=reward,
            )
            self._last_obs = ""

        else:
            obs = "Error: Could not parse your response. Use <tool_call> or <response> tags."
            self.conv_logger.log_turn(
                turn_number=self.step_count,
                agent_input_observation=agent_obs,
                agent_raw_output=parsed_action.get("raw", "[unparseable]")[:500],
                parsed_action_type="invalid",
                is_done=False,
                step_reward=0.0,
            )
            self._last_obs = obs
            obs, reward, done, info = (obs, 0.0, False, {"is_action_valid": 0, "action_type": "invalid"})

        # H3 fix: Force termination with final reward if max_steps reached
        if not done and self.step_count >= self.max_steps:
            from tau2.data_model.simulation import TerminationReason
            self._termination_reason = TerminationReason.MAX_STEPS
            timeout_reward, timeout_diag = self._compute_final_reward()
            self.conv_logger.end_episode(
                final_reward=timeout_reward,
                termination_reason="max_steps",
                diagnostics=timeout_diag,
                tool_calls_made=self.tool_calls_made,
            )
            return obs, timeout_reward, True, {
                "won": timeout_reward >= 0.99,
                "action_type": "max_steps_timeout",
                "reward_diagnostics": timeout_diag,
                "is_action_valid": info.get("is_action_valid", 1),
            }

        return obs, reward, done, info

    def _handle_tool_calls(self, calls: List[Dict]) -> Tuple[str, float, bool, Dict]:
        """Execute tool calls via tau2-bench environment."""
        from tau2.data_model.message import (
            AssistantMessage as TauAssistantMessage,
            ToolCall as TauToolCall,
            ToolMessage as TauToolMessage,
        )

        # Build tau2 ToolCall objects with stable IDs for the evaluator
        tau_tool_calls = []
        for i, tc in enumerate(calls):
            tc_id = f"tc_{self._tool_call_counter + i}"
            tau_tool_calls.append(TauToolCall(
                id=tc_id,
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
                requestor="assistant",
            ))

        # Execute each call, only track successful ones for tool-call reward
        tool_results = []
        for tc, tau_tc in zip(calls, tau_tool_calls):
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            error = False
            try:
                result = self.env.make_tool_call(name, requestor="assistant", **args)
                result_str = self.env.to_json_str(result)
                self.tool_calls_made.append(tc)
            except Exception as e:
                result_str = f"Error: {str(e)}"
                error = True
            tool_results.append({"tau_tc": tau_tc, "result": result_str, "error": error})

        self._tool_call_counter += len(calls)

        # Record to message history: one AssistantMessage with all tool_calls,
        # followed by one ToolMessage per call (required format for tau2 evaluator)
        self.message_history.append(TauAssistantMessage(
            role="assistant",
            tool_calls=tau_tool_calls,
        ))
        for tr in tool_results:
            self.message_history.append(TauToolMessage(
                id=tr["tau_tc"].id,
                role="tool",
                content=tr["result"],
                requestor="assistant",
                error=tr["error"],
            ))

        # Sync tools after execution
        try:
            self.env.sync_tools()
        except Exception as e:
            print(f"[Tau2Solver] Warning: sync_tools failed: {e}")

        # Format tool results as observation
        obs_parts = []
        for tr in tool_results:
            obs_parts.append(f"Tool '{tr['tau_tc'].name}' returned: {tr['result']}")
        obs = "\n".join(obs_parts)

        return obs, 0.0, False, {"action_type": "tool_call", "is_action_valid": 1}

    def _handle_response(self, content: str) -> Tuple[str, float, bool, Dict]:
        """Send text response to user simulator, get reply."""
        from tau2.data_model.message import (
            AssistantMessage as TauAssistantMessage,
            UserMessage as TauUserMessage,
        )
        from tau2.data_model.simulation import TerminationReason

        # Record agent's text response
        self.message_history.append(TauAssistantMessage(role="assistant", content=content))

        user_reply = self.user_sim.respond(content)

        # Record user's reply
        self.message_history.append(TauUserMessage(role="user", content=user_reply))

        if self.user_sim.is_stop(user_reply):
            if self.user_sim.is_api_fail(user_reply):
                # API failure forced the stop — don't credit as user satisfaction.
                # Use TOO_MANY_ERRORS so the tau2 evaluator gives 0 reward
                # (TerminationReason is a required enum; None would crash SimulationRun).
                self._termination_reason = TerminationReason.TOO_MANY_ERRORS
                self.conv_logger.end_episode(
                    final_reward=0.0,
                    termination_reason="api_fail",
                    diagnostics={"api_fail": True},
                    tool_calls_made=self.tool_calls_made,
                )
                return user_reply, 0.0, True, {
                    "won": False,
                    "action_type": "api_fail",
                    "reward_diagnostics": {"api_fail": True},
                    "is_action_valid": 1,
                }
            else:
                self._termination_reason = TerminationReason.USER_STOP
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
        from tau2.data_model.simulation import TerminationReason

        self._termination_reason = TerminationReason.AGENT_STOP
        reward, diagnostics = self._compute_final_reward()
        return "", reward, True, {
            "won": reward >= 0.99,
            "action_type": "stop",
            "reward_diagnostics": diagnostics,
            "is_action_valid": 1,
        }

    def _compute_final_reward(self) -> Tuple[float, Dict]:
        """Compute reward at episode end.

        TOD-Zero (synthetic mode): uses user satisfaction + tool usage signal
        + pseudo-ground-truth action matching from challenger's expected actions.
        Standard mode: uses tool-call accuracy + tau2-bench task success.
        """
        if self.synthetic_mode:
            from agent_system.environments.env_package.tau2bench.rewards import compute_synthetic_reward
            reward, diagnostics = compute_synthetic_reward(
                termination_reason=self._termination_reason,
                tool_calls_made=self.tool_calls_made,
                expected_actions=getattr(self, '_expected_actions', []),
            )
        else:
            from agent_system.environments.env_package.tau2bench.rewards import compute_combined_reward
            reward, diagnostics = compute_combined_reward(
                env_constructor=self.env_constructor,
                task=self.task,
                tool_calls_made=self.tool_calls_made,
                message_history=self.message_history,
                domain=self.domain,
                termination_reason=self._termination_reason,
                tool_call_reward_coef=self.tool_call_reward_coef,
                task_success_reward_coef=self.task_success_reward_coef,
            )

        # Finalize conversation log
        self.conv_logger.end_episode(
            final_reward=reward,
            termination_reason=str(self._termination_reason) if self._termination_reason else "unknown",
            diagnostics=diagnostics,
            tool_calls_made=self.tool_calls_made,
        )

        return reward, diagnostics


# ===================================================================
# Single challenger worker
# ===================================================================

class _ChallengerWorker:
    """Manages a single challenger environment instance.

    TOD-Zero challenger: generates realistic customer scenarios grounded in
    real DB data. Outputs <instructions> + <actions> used to drive multi-turn
    conversations between the solver agent and user simulator.
    """

    def __init__(self, domain: str, seed: int = 0):
        self.domain = domain
        self.rng = np.random.RandomState(seed)

        from agent_system.environments.env_package.tau2bench.db_sampler import (
            get_tools, get_policy, format_tools_for_prompt,
            format_context_for_prompt, sample_context, DOMAIN_TOOLS,
        )
        self._get_tools = get_tools
        self._get_policy = get_policy
        self._format_tools = format_tools_for_prompt
        self._format_context = format_context_for_prompt
        self._sample_context = lambda: sample_context(domain)

        self.policy = get_policy(domain)
        self.tools = get_tools(domain)
        self.tool_names = {t["name"] for t in self.tools}
        self._tools_text = format_tools_for_prompt(self.tools)

    def reset(self, kwargs: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Sample a fresh DB context and return the formatted challenger prompt."""
        from agent_system.environments.prompts.tau2bench import (
            TAU2BENCH_CHALLENGER_SYSTEM,
            TAU2BENCH_CHALLENGER_USER,
            TAU2BENCH_CHALLENGER_TEMPLATE,
        )
        context = self._sample_context()
        context_text = self._format_context(context)

        system_prompt = TAU2BENCH_CHALLENGER_SYSTEM.format(
            domain=self.domain,
            policy=self.policy,
            tools_text=self._tools_text,
            context_text=context_text,
        )
        obs = TAU2BENCH_CHALLENGER_TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=TAU2BENCH_CHALLENGER_USER,
        )
        info = {
            "domain": self.domain,
            "data_source": f"tau2bench_challenger_{self.domain}",
            "context": context,
        }
        return obs, info

    def step(self, parsed_action: Dict) -> Tuple[str, float, bool, Dict]:
        """Evaluate challenger output and compute reward."""
        from agent_system.environments.env_package.tau2bench.rewards import compute_challenger_reward
        reward = compute_challenger_reward(
            parsed_action,
            domain_tool_names=self.tool_names,
        )
        is_valid = 1 if parsed_action.get("type") == "challenger_output" else 0
        return "", reward, True, {
            "won": reward >= 0.7,
            "reward": reward,
            "is_action_valid": is_valid,
            "generated_instructions": parsed_action.get("instructions", "") if is_valid else "",
            "generated_actions": parsed_action.get("actions", []) if is_valid else [],
        }


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
        tool_call_reward_coef: float = 0.5,
        task_success_reward_coef: float = 0.5,
        external_scenarios: Optional[List[Dict]] = None,
        conv_log_dir: Optional[str] = None,
        task_split: Optional[str] = "base",
        max_steps: int = 30,
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
                tool_call_reward_coef=tool_call_reward_coef,
                task_success_reward_coef=task_success_reward_coef,
                external_scenarios=external_scenarios,
                conv_log_dir=conv_log_dir,
                task_split=task_split,
                max_steps=max_steps,
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
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        if hasattr(self, "_loop"):
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
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        if hasattr(self, "_loop"):
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
    tool_call_reward_coef: float = 0.5,
    task_success_reward_coef: float = 0.5,
    challenger_scenarios_path: Optional[str] = None,
    conv_log_dir: Optional[str] = None,
    task_split: Optional[str] = "base",
    max_steps: int = 30,
    **kwargs,
) -> Tau2BenchSolverEnvs:
    """Build tau2-bench solver environments.

    Args:
        challenger_scenarios_path: path to JSON file with challenger-generated
            user goals. If provided, activates TOD-Zero synthetic training mode.
        conv_log_dir: if set, logs sampled episode conversations to this directory
            as JSON files for post-training debugging.
        task_split: which task split to load from registry ("train", "test", "base").
            Training envs should use "train" or None (synthetic mode ignores this).
            Val envs should use "test".
        max_steps: maximum conversation turns before forced termination.
    """
    external_scenarios = None
    if challenger_scenarios_path:
        import os
        if os.path.exists(challenger_scenarios_path):
            with open(challenger_scenarios_path, "r") as f:
                data = json.load(f)
            # Keep full scenario dicts so solver can use expected actions as
            # pseudo-ground-truth for richer reward in synthetic mode.
            # Format: [{"instructions": str, "actions": [{"name":..., "arguments":...}]}, ...]
            if isinstance(data, list):
                external_scenarios = []
                for item in data:
                    if isinstance(item, str) and item.strip():
                        external_scenarios.append({"instructions": item, "actions": []})
                    elif isinstance(item, dict):
                        instr = item.get("instructions", "") or item.get("goal", "")
                        actions = item.get("actions", [])
                        if instr.strip():
                            external_scenarios.append({"instructions": instr, "actions": actions})
            print(f"[Tau2Solver] Loaded {len(external_scenarios)} challenger scenarios from {challenger_scenarios_path}")
        else:
            print(f"[Tau2Solver] Warning: challenger_scenarios_path not found: {challenger_scenarios_path}")

    return Tau2BenchSolverEnvs(
        domain=domain,
        user_sim_url=user_sim_url,
        user_sim_model=user_sim_model,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        tool_call_reward_coef=tool_call_reward_coef,
        task_success_reward_coef=task_success_reward_coef,
        external_scenarios=external_scenarios,
        conv_log_dir=conv_log_dir,
        task_split=task_split,
        max_steps=max_steps,
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
