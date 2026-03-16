"""
Tests for tau2-bench integration with verl-agent.

Tests cover:
  1. Projection functions (solver + challenger parsing)
  2. Reward functions (solver accuracy + challenger format/validity)
  3. Environment wrappers (solver + challenger reset/step)
  4. User simulator (basic functionality)
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the repo root to path
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

# Stub out parent __init__.py to avoid torch/omegaconf dependency chain
import types
for mod_name in [
    "agent_system",
    "agent_system.environments",
    "agent_system.environments.env_package",
    "agent_system.environments.env_package.tau2bench",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules[mod_name].__path__ = [
            os.path.join(REPO_ROOT, mod_name.replace(".", "/"))
        ]

# Now import the tau2bench submodules directly
from agent_system.environments.env_package.tau2bench.projection import (
    solver_projection,
    challenger_projection,
)
from agent_system.environments.env_package.tau2bench.rewards import (
    compute_solver_accuracy,
    compute_solver_reward,
    compute_solver_format_reward,
    compute_challenger_format_reward,
    compute_challenger_validity_reward,
    compute_challenger_reward,
    score_tool_call,
    f1_keys,
    robust_value_match,
)
from agent_system.environments.env_package.tau2bench.user_sim import LightUserSimulator
from agent_system.environments.env_package.tau2bench.envs import (
    _SolverWorker,
    _ChallengerWorker,
)


# ===================================================================
# 1. Projection tests
# ===================================================================

class TestSolverProjection:
    def test_tool_call_single(self):
        actions = [
            '<think>I need to look up the order.</think>\n'
            '<tool_call>{"name": "get_order_details", "arguments": {"order_id": "12345"}}</tool_call>'
        ]
        results, valids = solver_projection(actions)
        assert len(results) == 1
        assert results[0]["type"] == "tool_call"
        assert results[0]["calls"][0]["name"] == "get_order_details"
        assert results[0]["calls"][0]["arguments"]["order_id"] == "12345"
        assert valids[0] == 1

    def test_tool_call_multiple(self):
        actions = [
            '<think>Need both.</think>\n'
            '<tool_call>[{"name": "func_a", "arguments": {"x": 1}}, '
            '{"name": "func_b", "arguments": {"y": 2}}]</tool_call>'
        ]
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "tool_call"
        assert len(results[0]["calls"]) == 2
        assert valids[0] == 1

    def test_response(self):
        actions = [
            '<think>The user needs help.</think>\n'
            '<response>Hello! How can I help you today?</response>'
        ]
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "response"
        assert "Hello" in results[0]["content"]
        assert valids[0] == 1

    def test_stop_signal(self):
        actions = [
            '<think>Issue resolved.</think>\n'
            '<response>[STOP] Your issue has been resolved.</response>'
        ]
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "stop"
        assert valids[0] == 1

    def test_both_tags_invalid(self):
        actions = [
            '<tool_call>{"name": "foo", "arguments": {}}</tool_call>\n'
            '<response>Also responding</response>'
        ]
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "invalid"
        assert valids[0] == 0

    def test_no_tags_invalid(self):
        actions = ["Just some random text without any tags"]
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "response"
        assert valids[0] == 0

    def test_bad_json_in_tool_call(self):
        actions = ['<tool_call>not valid json{</tool_call>']
        results, valids = solver_projection(actions)
        assert results[0]["type"] == "invalid"
        assert valids[0] == 0

    def test_batch_processing(self):
        actions = [
            '<response>Hello</response>',
            '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
            '<response>[STOP] Done</response>',
        ]
        results, valids = solver_projection(actions)
        assert len(results) == 3
        assert results[0]["type"] == "response"
        assert results[1]["type"] == "tool_call"
        assert results[2]["type"] == "stop"


class TestChallengerProjection:
    def test_valid_task_spec(self):
        actions = [
            '<think>Create a booking task.</think>\n'
            '<question>I need to cancel my flight booking ABC123.</question>\n'
            '<available_tools>[{"name": "cancel_booking", "parameters": {"type": "object"}}]</available_tools>\n'
            '<tool_call_answer>{"name": "cancel_booking", "arguments": {"booking_id": "ABC123"}}</tool_call_answer>'
        ]
        results, valids = challenger_projection(actions)
        assert results[0]["type"] == "task_spec"
        assert "cancel" in results[0]["question"].lower()
        assert valids[0] == 1

    def test_missing_tags(self):
        actions = [
            '<question>Some question</question>\n'
            '<available_tools>[{"name": "foo"}]</available_tools>'
        ]
        results, valids = challenger_projection(actions)
        assert results[0]["type"] == "invalid"
        assert valids[0] == 0


# ===================================================================
# 2. Reward tests
# ===================================================================

class TestSolverRewards:
    def test_exact_match(self):
        pred = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert reward == 1.0
        assert diag["mean_name_score"] == 1.0

    def test_wrong_name(self):
        pred = [{"name": "wrong_func", "arguments": {"order_id": "123"}}]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert reward < 1.0
        assert diag["mean_name_score"] == 0.0

    def test_correct_name_wrong_args(self):
        pred = [{"name": "get_order", "arguments": {"order_id": "999"}}]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert diag["mean_name_score"] == 1.0
        assert diag["mean_value_score"] == 0.0

    def test_extra_calls_penalty(self):
        pred = [
            {"name": "get_order", "arguments": {"order_id": "123"}},
            {"name": "extra_call", "arguments": {}},
            {"name": "another_extra", "arguments": {}},
        ]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert reward < 1.0
        assert diag["extra_calls"] == 2

    def test_no_predictions_no_gt(self):
        reward, _ = compute_solver_accuracy([], [])
        assert reward == 1.0

    def test_no_predictions_with_gt(self):
        gt = [{"name": "func", "arguments": {}}]
        reward, _ = compute_solver_accuracy([], gt)
        assert reward == 0.0

    def test_multiple_calls_matching(self):
        pred = [
            {"name": "func_a", "arguments": {"x": 1}},
            {"name": "func_b", "arguments": {"y": 2}},
        ]
        gt = [
            {"name": "func_a", "arguments": {"x": 1}},
            {"name": "func_b", "arguments": {"y": 2}},
        ]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert reward == 1.0

    def test_partial_arg_match(self):
        pred = [{"name": "func", "arguments": {"a": 1, "b": 2, "c": 3}}]
        gt = [{"name": "func", "arguments": {"a": 1, "b": 99}}]
        reward, diag = compute_solver_accuracy(pred, gt)
        assert 0.0 < reward < 1.0

    def test_robust_value_match_numbers(self):
        assert robust_value_match(42, "42")
        assert robust_value_match("3.14", 3.14)
        assert not robust_value_match(42, "43")

    def test_robust_value_match_strings(self):
        assert robust_value_match("hello world", "hello  world")
        assert not robust_value_match("hello", "world")

    def test_solver_reward_with_action_objects(self):
        """Test compute_solver_reward with mock Action-like objects."""
        class MockAction:
            def __init__(self, name, arguments, requestor="assistant", compare_args=None):
                self.name = name
                self.arguments = arguments
                self.requestor = requestor
                self.compare_args = compare_args

        pred = [{"name": "get_user", "arguments": {"user_id": "abc"}}]
        gt = [MockAction("get_user", {"user_id": "abc", "extra": "ignored"}, compare_args=["user_id"])]
        reward, diag = compute_solver_reward(pred, gt)
        assert reward == 1.0

    def test_solver_format_reward(self):
        """Test format reward based on action validity flags."""
        # All valid actions
        assert compute_solver_format_reward([1, 1, 1, 1]) == 1.0
        # Half valid
        assert compute_solver_format_reward([1, 0, 1, 0]) == 0.5
        # No valid actions
        assert compute_solver_format_reward([0, 0, 0]) == 0.0
        # Empty
        assert compute_solver_format_reward([]) == 0.0

    def test_solver_combined_reward_with_valids(self):
        """Test compute_solver_reward with action_valids for format component."""
        pred = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]

        # Perfect match with all valid actions
        reward, diag = compute_solver_reward(pred, gt, action_valids=[1, 1, 1])
        assert reward == 1.0
        assert diag["format_reward"] == 1.0
        assert diag["tool_call_reward"] == 1.0
        assert diag["task_success"] == 1.0  # fallback: tool_call >= 0.99

        # Perfect tool-call match but poor format
        reward2, diag2 = compute_solver_reward(pred, gt, action_valids=[1, 0, 0])
        assert reward2 < 1.0
        assert diag2["format_reward"] == pytest.approx(1.0 / 3.0)
        assert diag2["tool_call_reward"] == 1.0

    def test_solver_combined_reward_no_match(self):
        """Test combined reward when tool calls don't match."""
        pred = [{"name": "wrong", "arguments": {}}]
        gt = [{"name": "get_order", "arguments": {"order_id": "123"}}]
        reward, diag = compute_solver_reward(pred, gt, action_valids=[1])
        # format=1.0, tool_call<1.0, task_success=0.0 (fallback)
        assert diag["task_success"] == 0.0
        assert diag["format_reward"] == 1.0
        assert reward < 1.0


class TestChallengerRewards:
    def test_format_reward_valid(self):
        action = {
            "type": "task_spec",
            "question": "I need to cancel my flight booking.",
            "available_tools": json.dumps([{"name": "cancel_booking", "parameters": {"type": "object"}}]),
            "tool_call_answer": json.dumps({"name": "cancel_booking", "arguments": {"booking_id": "ABC"}}),
        }
        reward = compute_challenger_format_reward(action)
        assert reward > 0.9

    def test_format_reward_invalid(self):
        action = {"type": "invalid", "raw": "garbage"}
        reward = compute_challenger_format_reward(action)
        assert reward == 0.0

    def test_format_reward_partial(self):
        action = {
            "type": "task_spec",
            "question": "A valid question here.",
            "available_tools": "not valid json",
            "tool_call_answer": json.dumps({"name": "foo", "arguments": {"x": 1}}),
        }
        reward = compute_challenger_format_reward(action)
        assert 0.3 < reward < 0.8

    def test_validity_reward_valid(self):
        action = {
            "type": "task_spec",
            "available_tools": json.dumps([{
                "name": "get_booking",
                "parameters": {
                    "type": "object",
                    "properties": {"booking_id": {"type": "string"}},
                    "required": ["booking_id"],
                },
            }]),
            "tool_call_answer": json.dumps({
                "name": "get_booking",
                "arguments": {"booking_id": "ABC123"},
            }),
        }
        reward = compute_challenger_validity_reward(action)
        assert reward == 1.0

    def test_validity_reward_tool_not_found(self):
        action = {
            "type": "task_spec",
            "available_tools": json.dumps([{"name": "other_tool"}]),
            "tool_call_answer": json.dumps({"name": "nonexistent", "arguments": {}}),
        }
        reward = compute_challenger_validity_reward(action)
        assert reward == 0.0

    def test_combined_challenger_reward(self):
        action = {
            "type": "task_spec",
            "question": "I need to update my shipping address for order 12345.",
            "available_tools": json.dumps([{
                "name": "update_address",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}},
                    "required": ["order_id", "address"],
                },
            }]),
            "tool_call_answer": json.dumps({
                "name": "update_address",
                "arguments": {"order_id": "12345", "address": "123 Main St"},
            }),
        }
        reward = compute_challenger_reward(action)
        assert reward > 0.9


# ===================================================================
# 3. Environment wrapper tests
# ===================================================================

class TestSolverWorker:
    def _make_mock_worker(self):
        with patch.object(_SolverWorker, "__init__", return_value=None):
            worker = _SolverWorker.__new__(_SolverWorker)

        worker.domain = "mock"
        worker.rng = np.random.RandomState(42)
        worker.tool_calls_made = []
        worker.action_valids = []
        worker.step_count = 0
        worker.policy = "Be helpful."
        worker.tool_schemas = [{"name": "get_info", "parameters": {}}]

        # Mock task
        worker.task = MagicMock()
        action_mock = MagicMock()
        action_mock.name = "get_info"
        action_mock.arguments = {"id": "123"}
        action_mock.requestor = "assistant"
        action_mock.compare_args = None
        worker.task.evaluation_criteria.actions = [action_mock]

        # Mock environment
        worker.env = MagicMock()
        worker.env.make_tool_call.return_value = {"status": "success"}
        worker.env.to_json_str.return_value = '{"status": "success"}'
        worker.env.sync_tools.return_value = None

        # Mock evaluate for task success reward
        eval_result = MagicMock()
        eval_result.success = True
        worker.env.evaluate.return_value = eval_result

        # Mock user sim
        worker.user_sim = MagicMock()
        worker.user_sim.respond.return_value = "Thanks!"
        worker.user_sim.is_stop.return_value = False

        return worker

    def test_handle_tool_calls(self):
        worker = self._make_mock_worker()
        obs, reward, done, info = worker._handle_tool_calls([
            {"name": "get_info", "arguments": {"id": "123"}}
        ])
        assert "get_info" in obs
        assert reward == 0.0
        assert done is False
        assert len(worker.tool_calls_made) == 1

    def test_handle_response(self):
        worker = self._make_mock_worker()
        obs, reward, done, info = worker._handle_response("Hello!")
        assert obs == "Thanks!"
        assert done is False

    def test_handle_response_user_stop(self):
        worker = self._make_mock_worker()
        worker.user_sim.respond.return_value = "[STOP] Thanks!"
        worker.user_sim.is_stop.return_value = True
        obs, reward, done, info = worker._handle_response("Done.")
        assert done is True
        assert "won" in info

    def test_handle_stop(self):
        worker = self._make_mock_worker()
        worker.tool_calls_made = [{"name": "get_info", "arguments": {"id": "123"}}]
        obs, reward, done, info = worker._handle_stop("Bye!")
        assert done is True
        assert reward > 0

    def test_compute_final_reward_correct(self):
        worker = self._make_mock_worker()
        worker.tool_calls_made = [{"name": "get_info", "arguments": {"id": "123"}}]
        worker.action_valids = [1, 1]  # all valid format actions
        reward, diag = worker._compute_final_reward()
        # format=1.0, tool_call=1.0, task_success=1.0 → 0.1+0.4+0.5=1.0
        assert reward == 1.0
        assert diag["format_reward"] == 1.0
        assert diag["task_success"] == 1.0

    def test_compute_final_reward_incorrect(self):
        worker = self._make_mock_worker()
        worker.tool_calls_made = [{"name": "wrong_func", "arguments": {"x": "y"}}]
        worker.action_valids = [1, 0]  # partial format validity
        worker.env.evaluate.return_value = MagicMock(success=False)
        reward, diag = worker._compute_final_reward()
        assert reward < 1.0
        assert diag["task_success"] == 0.0


class TestChallengerWorker:
    def test_step_valid_spec(self):
        with patch.object(_ChallengerWorker, "__init__", return_value=None):
            worker = _ChallengerWorker.__new__(_ChallengerWorker)

        worker.domain = "mock"
        worker.rng = np.random.RandomState(42)
        worker.policy = "Be helpful."
        worker.tool_schemas = [{"name": "get_info", "parameters": {"type": "object"}}]
        worker.tool_names = {"get_info"}

        action = {
            "type": "task_spec",
            "question": "I need information about item 456.",
            "available_tools": json.dumps([{
                "name": "get_info",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            }]),
            "tool_call_answer": json.dumps({"name": "get_info", "arguments": {"id": "456"}}),
        }
        obs, reward, done, info = worker.step(action)
        assert done is True
        assert reward > 0.8

    def test_step_invalid_spec(self):
        with patch.object(_ChallengerWorker, "__init__", return_value=None):
            worker = _ChallengerWorker.__new__(_ChallengerWorker)

        worker.domain = "mock"
        worker.tool_names = {"get_info"}

        action = {"type": "invalid", "raw": "garbage"}
        obs, reward, done, info = worker.step(action)
        assert done is True
        assert reward == 0.0


# ===================================================================
# 4. User simulator tests
# ===================================================================

class TestUserSimulator:
    def test_init(self):
        sim = LightUserSimulator(
            api_url="http://localhost:8000/v1",
            model="test-model",
            user_instructions="You are a customer who needs to cancel a flight.",
        )
        assert "cancel" in sim.system_prompt.lower()
        assert sim.history == []

    def test_is_stop(self):
        sim = LightUserSimulator(
            api_url="http://localhost:8000/v1",
            model="test-model",
            user_instructions="Test",
        )
        assert sim.is_stop("[STOP] Thanks!")
        assert sim.is_stop("[TRANSFER] Transferring...")
        assert not sim.is_stop("I still need help.")

    def test_generate_first_message(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi, I need to cancel my flight."
        mock_client.chat.completions.create.return_value = mock_response

        sim = LightUserSimulator(
            api_url="http://localhost:8000/v1",
            model="test-model",
            user_instructions="Cancel flight ABC123",
        )
        sim._client = mock_client

        msg = sim.generate_first_message()
        assert msg == "Hi, I need to cancel my flight."
        assert len(sim.history) == 1

    def test_respond(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Yes, the booking ID is ABC123."
        mock_client.chat.completions.create.return_value = mock_response

        sim = LightUserSimulator(
            api_url="http://localhost:8000/v1",
            model="test-model",
            user_instructions="Cancel flight ABC123",
        )
        sim._client = mock_client

        reply = sim.respond("Can you provide your booking ID?")
        assert reply == "Yes, the booking ID is ABC123."
        assert len(sim.history) == 2


# ===================================================================
# 5. Score tool call unit tests
# ===================================================================

class TestScoreToolCall:
    def test_perfect_match(self):
        pred = {"name": "func", "arguments": {"a": 1, "b": "hello"}}
        gt = {"name": "func", "arguments": {"a": 1, "b": "hello"}}
        n, k, v = score_tool_call(pred, gt)
        assert n == 1.0
        assert k == 1.0
        assert v == 1.0

    def test_name_mismatch(self):
        pred = {"name": "wrong", "arguments": {"a": 1}}
        gt = {"name": "func", "arguments": {"a": 1}}
        n, k, v = score_tool_call(pred, gt)
        assert n == 0.0
        assert k == 1.0
        assert v == 1.0

    def test_extra_keys(self):
        pred = {"name": "func", "arguments": {"a": 1, "b": 2, "c": 3}}
        gt = {"name": "func", "arguments": {"a": 1}}
        n, k, v = score_tool_call(pred, gt)
        assert n == 1.0
        assert k < 1.0

    def test_missing_keys(self):
        pred = {"name": "func", "arguments": {"a": 1}}
        gt = {"name": "func", "arguments": {"a": 1, "b": 2, "c": 3}}
        n, k, v = score_tool_call(pred, gt)
        assert n == 1.0
        assert k < 1.0

    def test_empty_args(self):
        pred = {"name": "func", "arguments": {}}
        gt = {"name": "func", "arguments": {}}
        n, k, v = score_tool_call(pred, gt)
        assert n == 1.0
        assert k == 1.0
        assert v == 1.0


# ===================================================================
# 6. F1 keys tests
# ===================================================================

class TestF1Keys:
    def test_identical(self):
        assert f1_keys({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert f1_keys({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        score = f1_keys({"a", "b"}, {"b", "c"})
        assert abs(score - 0.5) < 1e-9

    def test_both_empty(self):
        assert f1_keys(set(), set()) == 1.0

    def test_one_empty(self):
        assert f1_keys(set(), {"a"}) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
