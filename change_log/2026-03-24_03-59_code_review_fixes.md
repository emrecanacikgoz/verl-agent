# Code Review Fixes — 2026-03-24

## Overview

Full code review of the verl-agent tau2-bench integration for the TOD-Zero self-play pipeline. 
Focused on airline and retail domains only (telecom deferred). 
6 files modified, 12 issues fixed, 1 deferred (H1: difficulty-calibrated challenger reward).

---

## Files Modified

| # | File | What Changed |
|---|------|-------------|
| 1 | `agent_system/environments/prompts/tau2bench.py` | Prompt template updates |
| 2 | `agent_system/environments/env_manager.py` | Rewrote `Tau2BenchSolverEnvironmentManager`, added split/null handling |
| 3 | `agent_system/environments/env_package/tau2bench/envs.py` | Added `task_split`, `max_steps` params; max-step termination; reduced logging |
| 4 | `agent_system/environments/env_package/tau2bench/rewards.py` | Added trajectory sanitizer (M1); reweighted synthetic reward (H2) |
| 5 | `agent_system/environments/env_package/tau2bench/user_sim.py` | Load tau2-bench official user sim guidelines (H4) |
| 6 | `examples/gigpo_trainer/run_tod_zero.sh` | Fixed final evaluation to use vLLM servers (N1) |

---

## Issue-by-Issue Details

### C1: Train/Test Split Leakage — FIXED

**Problem:** `_SolverWorker.__init__()` called `registry.get_tasks_loader(domain)()` with default `task_split_name="base"`, loading ALL tasks (train+test). Val envs also used all tasks, so validation metrics included test tasks.

**Fix:**
- Added `task_split` parameter to `_SolverWorker`, `Tau2BenchSolverEnvs`, and `build_tau2bench_solver_envs`
- `env_manager.py`: training envs pass `task_split="train"`, val envs pass `task_split="test"`

**Files:** `envs.py` (lines 48, 73), `env_manager.py` (lines 942, 957)

---

### C2: Policy Truncation — FIXED

**Problem:** `build_text_obs()` had `policy=workers[i].policy[:2000]`. Airline policy is ~7.6KB, retail ~6.7KB. Cutting >60% of policy meant the agent couldn't see most rules. The `CommunicateEvaluator` checks if the agent communicated specific info defined in the policy — if that section was truncated, the agent could never learn it.

**Fix:** Full policy text passed to system prompt. No truncation.

**Files:** `env_manager.py` (line 657: `policy=w.policy` replaces `policy=workers[i].policy[:2000]`)

---

### C3: System Prompt Rebuilt Every Turn — FIXED

**Problem:** Every call to `build_text_obs()` reconstructed the entire system prompt (~5KB: policy + full JSON tool schemas). With `history_length=4` and raw LLM outputs (including `<think>` reasoning) stored in history, prompts easily exceeded 4096 tokens. Left-truncation then cut the system prompt, so the agent lost its instructions mid-episode.

**Fix (3 sub-changes):**
1. **Cached system prompts:** Built once in `_cache_system_prompts()` on `reset()`, reused every turn.
2. **Compact tool signatures:** Changed from full OpenAI JSON schemas (~2000+ tokens) to `- name(params) — description` format (~430 tokens). Template placeholder changed from `{tool_schemas}` to `{tool_signatures}`.
3. **Clean history entries:** Agent outputs stripped of `<think>...</think>` blocks before storage. Both actions and observations truncated to 600 chars max.

**Token budget after fix:** System ~2430 + History(4×150) ~600 + Current ~100 = ~3130 tokens (well within 4096 limit).

**Files:** `prompts/tau2bench.py` (line 15: `{tool_signatures}`; line 48: removed `{history_length}`), `env_manager.py` (entire `Tau2BenchSolverEnvironmentManager` class rewritten)

---

### C4: Evaluation Split Mismatch — FIXED

**Problem:** Val envs during training used all tasks (base split). Final eval used test split. Inconsistent.

**Fix:** Val envs now use `task_split="test"` (same as final eval).

**Files:** `env_manager.py` (line 957)

---

### H2: Synthetic Reward Weights — FIXED

**Problem:** Original weights were `completion=0.4, tool_usage=0.1, action_match=0.5`. The action_match component compared solver's tool calls against challenger's expected actions using full greedy matching (name + arg keys + arg values). With 50% weight, half the solver's reward came from matching the challenger's output.

**Fix:** Reweighted to `completion=0.5, tool_usage=0.2, action_match=0.3`. Completion (user satisfaction) is now the primary signal. Action match is retained with full matching (including arguments) because the challenger sees real DB context — argument values are grounded in actual entities, not random guesses. Also added `MAX_STEPS → 0.1` partial credit (was 0.0).

**What was NOT changed:** The matching function itself (`compute_solver_accuracy`) is unchanged. Full name + arg key F1 + arg value matching is preserved.

**Files:** `rewards.py` (lines 596-598: new default coefficients; line 635-636: MAX_STEPS handling)

---

### H3: Max-Step Termination Without Reward — FIXED

**Problem:** When conversation exceeded `max_steps=30`, the rollout loop exited but `_compute_final_reward()` was never called. The episode got total reward 0.0 silently. `_termination_reason` stayed `None`, which would crash `SimulationRun` construction.

**Fix:** Added max-step check at the end of `_SolverWorker.step()`. If `step_count >= max_steps` and episode isn't done, forces termination with `TerminationReason.MAX_STEPS`, calls `_compute_final_reward()`, logs the episode, and returns `done=True` with proper reward. Also added `max_steps` parameter (propagated from config through to worker).

**Files:** `envs.py` (lines 48, 57, 277-297)

---

### H4: User Simulator Missing tau2-bench Guidelines — FIXED

**Problem:** `LightUserSimulator` used a hardcoded simplified system prompt (14 lines). tau2-bench's native `UserSimulator` loads detailed `simulation_guidelines.md` files (~30+ lines) with specific behavioral rules: progressive disclosure, no hallucination, grounding responses in tool call results, one-action-at-a-time instruction. Training with simplified rules meant the user sim behaved differently during training vs. `tau2 run` evaluation.

**Fix:** `user_sim.py` now loads tau2-bench's official `simulation_guidelines.md` from `tau2-bench/data/tau2/user_simulator/`. Guidelines cached at class level (loaded once). Falls back to built-in minimal version if file not found. Added `use_tools` parameter for future telecom support (loads `simulation_guidelines_tools.md` instead).

**Files:** `user_sim.py` (entire top section rewritten: lines 10-55 new; `__init__` and `reset` updated to use `{guidelines}` template)

---

### M1: Evaluator Replay Failures on Error Tool Calls — FIXED

**Problem:** tau2-bench's `EnvironmentEvaluator` replays ALL tool calls from the trajectory via `set_state()` and compares results exactly. When a tool call errored during the episode (e.g., `"Error: user_id 'xyz' not found"`), the error string might differ on replay (different exception formatting, state-dependent messages), causing `set_state` to raise `ValueError`. This was caught by the try/except in `compute_task_success_reward`, silently returning reward=0.0 — hiding correct agent behavior.

**Fix:** Added `_sanitize_trajectory_for_evaluator()` that strips error tool calls from message history before evaluation. Logic: for each `AssistantMessage` with tool_calls, keep only successful calls (where `ToolMessage.error=False`). If all calls failed, skip the entire block. Error calls never changed DB state, so removing them is safe for environment state reconstruction. Also added traceback logging to the except block.

**Files:** `rewards.py` (lines 374-452: new function; lines 484-486: called before evaluation; lines 515-518: better error logging)

---

### M2: Conversation Logger I/O Overhead — FIXED

**Problem:** `ConversationLogger` was set to `sample_rate=1.0` (log every episode). During training with thousands of episodes, this creates one JSON file per episode.

**Fix:** Reduced to `sample_rate=0.05` (log 5% of episodes).

**Files:** `envs.py` (line 67)

---

### M3: "null" String From Hydra Config — FIXED

**Problem:** `run_tau2bench_solver.sh` defaults `CHALLENGER_SCENARIOS_PATH` to the string `"null"`. Hydra may pass this as the literal string `"null"` to Python instead of `None`. The code would try to open a file named "null".

**Fix:** Added explicit check: `if challenger_scenarios_path in (None, "null", "None", ""): challenger_scenarios_path = None`

**Files:** `env_manager.py` (lines 924-926)

---

### N1: Final Evaluation Script Broken — FIXED

**Problem:** `run_tod_zero.sh` ran `tau2 run --agent-llm "$SOLVER_PREV"` where `$SOLVER_PREV` is a local HF model path. But tau2-bench uses `litellm` which needs an OpenAI-compatible API endpoint, not a local path.

**Fix:** Rewrote the final evaluation section to:
1. Start a vLLM server with the final solver model on port 8001
2. Start a vLLM server for the user simulator on port 8002
3. Call `tau2 run` with `--agent-llm "openai/solver_eval"` and `--agent-llm-args` pointing to the local server
4. Same for user simulator
5. Clean up servers after evaluation

**Files:** `examples/gigpo_trainer/run_tod_zero.sh` (lines 253-318 rewritten)

---

## Deferred

### H1: Difficulty-Calibrated Challenger Reward

The challenger reward currently only checks format validity (well-formed XML, valid tool names, correct arg names). There is no signal for scenario quality, diversity, or difficulty calibration to the agent's competence frontier. This is the core research contribution and was intentionally deferred.

---

## Remaining Notes (not bugs, informational)

- **N2 (Conversation flow):** Training starts with user speaking first (`generate_first_message`). tau2-bench native evaluation starts with agent greeting ("Hi! How can I help?"). Minor distribution shift at turn 1 only.
- **N3 (No initial_state):** Both airline and retail tasks have `initial_state=None`. DB stays at base state in both standard and synthetic mode. No behavioral difference.
- **M4 (Vestigial data prep):** `prepare.py` loads geometry3k to create dummy parquet files. The actual task data comes from the environment. Harmless placeholder.
