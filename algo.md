# ALGORITHM.md — TOD-Zero Self-Play: Complete Data Flow Specification

This document traces **every input and output** through the TOD-Zero self-play pipeline. Its purpose is to serve as a debugging reference — if any data is missing, misrouted, or has the wrong format, this document specifies what the correct state should be.

---

## Table of Contents

1. [Self-Play Loop Overview](#1-self-play-loop-overview)
2. [Step 1: Challenger Training](#2-step-1-challenger-training)
3. [Step 2: Scenario Generation (Offline)](#3-step-2-scenario-generation-offline)
4. [Step 3: Solver Training](#4-step-3-solver-training)
5. [Reward Functions Reference](#5-reward-functions-reference)
6. [Data Format Specifications](#6-data-format-specifications)
7. [Prompt Templates Reference](#7-prompt-templates-reference)
8. [Critical Invariants (Bug Checklist)](#8-critical-invariants-bug-checklist)

---

## 1. Self-Play Loop Overview

Each iteration performs four sequential steps. The output of each step feeds into the next.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ITERATION N                                          │
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │ Step 1            │    │ Step 2            │    │ Step 3               │  │
│  │ TRAIN CHALLENGER  │───→│ GENERATE SCENARIOS│───→│ TRAIN SOLVER         │  │
│  │ (GRPO, 4 GPUs)    │    │ (vLLM offline)    │    │ (GRPO, 3 GPUs)       │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘  │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│  Challenger ckpt          scenarios.json            Solver ckpt             │
│  (HF model)               (K scenarios)             (HF model)             │
│                                                                             │
│  Inputs:                  Inputs:                  Inputs:                  │
│  • prev challenger ckpt   • trained challenger     • prev solver ckpt       │
│  • tau2 DB + policy       • tau2 DB + policy       • scenarios.json         │
│  • tool schemas           • tool schemas           • user sim (vLLM, GPU 3) │
│                                                    • tau2 environment       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Feed ckpts into Iteration N+1
```

**Orchestration script**: `examples/gigpo_trainer/run_tod_zero.sh`

---

## 2. Step 1: Challenger Training

**Goal**: Train $C_\theta$ to generate well-formed, grounded customer service scenarios.

**Script**: `examples/gigpo_trainer/run_tau2bench_challenger.sh`

**Environment**: `Tau2BenchChallengerEnvironmentManager` → `_ChallengerWorker`

### 2.1 Challenger Input Construction

On each `_ChallengerWorker.reset()`, a prompt is built from four data sources:

```
┌───────────────────────────────────────────────────────────────┐
│              CHALLENGER PROMPT (input to LLM)                 │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  DOMAIN: airline                                              │ ← config: env.tau2bench.domain
│                                                               │
│  POLICY:                                                      │ ← db_sampler.get_policy(domain)
│  <full policy.md text from tau2-bench>                        │   Source: tau2-bench/data/tau2/domains/{domain}/policy.md
│                                                               │
│  AVAILABLE TOOLS:                                             │ ← db_sampler.get_tools(domain) → format_tools_for_prompt()
│  1. get_user_details (READ)                                   │   Source: DOMAIN_TOOLS dict in db_sampler.py
│     Parameters: user_id (str)                                 │
│  2. cancel_reservation (WRITE)                                │
│     ...                                                       │
│                                                               │
│  DATABASE CONTEXT (real data you MUST reference):             │ ← db_sampler.sample_context(domain) → format_context_for_prompt()
│  User: {"user_id": "john_doe_123", "name": "John Doe", ...}  │   Source: tau2-bench/data/tau2/domains/{domain}/db.json
│  Reservation: {"reservation_id": "RES456", ...}              │   (random user + their reservations/orders sampled each reset)
│                                                               │
│  RULES: [7 rules about grounding, valid tools, etc.]         │
│                                                               │
│  OUTPUT FORMAT: <think>...</think>                            │
│                 <instructions>...</instructions>              │
│                 <actions>[...]</actions>                      │
│                                                               │
│  ---                                                          │
│  Generate a realistic customer service scenario now.          │ ← TAU2BENCH_CHALLENGER_USER
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Code path**:
- `_ChallengerWorker.reset()` → `db_sampler.sample_context()` → `format_context_for_prompt()` in `db_sampler.py`
- Prompt assembled using `TAU2BENCH_CHALLENGER_SYSTEM` + `TAU2BENCH_CHALLENGER_TEMPLATE` in `prompts/tau2bench.py`
- `Tau2BenchChallengerEnvironmentManager.build_text_obs()` passes the prompt through unchanged (line 759 of `env_manager.py`)

### 2.2 Challenger Output (LLM Generation)

The challenger LLM generates:

```xml
<think>
I'll create a scenario where a customer wants to change their flight.
The user john_doe_123 has reservation RES456 from LAX to JFK.
</think>

<instructions>
You are John Doe, a frequent flyer. You have a reservation (RES456)
for a flight from LAX to JFK on March 15. You want to change your
flight to March 20 instead. You know your name and email
(john@example.com) but not your user ID. You should be polite but
firm about needing the change.
</instructions>

<actions>
[
  {"name": "get_user_details", "arguments": {"user_id": "john_doe_123"}},
  {"name": "update_reservation_flights", "arguments": {"reservation_id": "RES456", ...}}
]
</actions>
```

### 2.3 Challenger Output Parsing

**Function**: `challenger_projection()` in `projection.py`

```
Raw LLM text
    │
    ▼
challenger_projection()                    ← projection.py:110
    │
    ├── Extract <instructions>...</instructions>  via RE_INSTRUCTIONS regex
    ├── Extract <actions>...</actions>             via RE_ACTIONS_TAG regex
    ├── JSON-parse actions string
    │
    ├── Validation:
    │   ├── instructions length ≥ 20 chars?
    │   ├── actions is non-empty list?
    │   └── every action has "name" key?
    │
    ▼
    Result: {"type": "challenger_output",
             "instructions": str,            ← user goal for user simulator
             "actions": [{"name":..., "arguments":...}, ...]}  ← pseudo-GT for solver reward
    Valid: 1

    OR

    Result: {"type": "invalid", "raw": ...}
    Valid: 0
```

### 2.4 Challenger Reward

**Function**: `_ChallengerWorker.step()` → `compute_challenger_reward()` in `rewards.py`

```
parsed_action (from challenger_projection)
    │
    ▼
compute_challenger_reward()                ← rewards.py (compute_challenger_reward function)
    │
    ├── compute_challenger_format_reward()  ← rewards.py
    │   ├── +0.25 if <instructions> present and ≥ 20 chars
    │   ├── +0.25 if <actions> present and parseable JSON list
    │   ├── +0.25 if all actions have "name" key
    │   └── +0.25 if all actions have dict "arguments"
    │   → r_format ∈ [0, 1]
    │
    ├── compute_challenger_validity_reward()  ← rewards.py
    │   ├── r_tool: fraction of actions with valid tool names (exist in domain)
    │   └── r_arg: fraction of actions with valid argument keys (match tool schema)
    │   → r_tool ∈ [0, 1], r_arg ∈ [0, 1]
    │
    ▼
    reward = 0.4 * r_format + 0.3 * r_tool + 0.3 * r_arg
    reward ∈ [0, 1]
```

**NOTE**: This reward does NOT include learnability. The challenger is rewarded for producing well-formed scenarios, not for producing scenarios that are difficult for the solver.

---

## 3. Step 2: Scenario Generation (Offline)

**Goal**: Use the trained challenger to produce $K$ scenarios saved as JSON.

**Script**: `examples/gigpo_trainer/generate_challenger_scenarios.py`

### 3.1 Generation Process

```
Trained Challenger (HF model)
    │
    ▼
Load into vLLM (TP=2, 4 GPUs)
    │
    ▼
For each batch:
    ├── Sample fresh DB context per item         ← db_sampler.sample_context()
    ├── Build prompt (same format as training)   ← TAU2BENCH_CHALLENGER_SYSTEM + TEMPLATE
    ├── vLLM generate (temperature=0.9)
    ├── Parse output                             ← parse_challenger_output()
    ├── Validate tool names                      ← validate_actions()
    ├── Dedup by action fingerprint              ← fingerprint()
    └── Accept or reject
    │
    ▼
scenarios.json
```

### 3.2 Scenario JSON Format

```json
[
  {
    "domain": "airline",
    "iter": 1,
    "instructions": "You are John Doe, a frequent flyer...",     ← GOES TO: user simulator
    "actions": [                                                   ← GOES TO: solver reward (pseudo-GT)
      {"name": "get_user_details", "arguments": {"user_id": "john_doe_123"}},
      {"name": "update_reservation_flights", "arguments": {"reservation_id": "RES456", ...}}
    ],
    "context": {                                                   ← METADATA ONLY (not used in training)
      "user": {"user_id": "john_doe_123", "name": "John Doe", ...},
      "reservation": {"reservation_id": "RES456", ...}
    }
  },
  ...
]
```

### 3.3 Where Each Field Goes

```
scenarios.json
    │
    ├── "instructions" ──────→ User Simulator system prompt
    │                          (drives the multi-turn conversation)
    │
    ├── "actions" ───────────→ Solver reward computation
    │                          (pseudo-ground-truth for action matching)
    │
    ├── "domain" ────────────→ Metadata (logged)
    ├── "iter" ──────────────→ Metadata (logged)
    └── "context" ───────────→ Metadata (not used at runtime)
```

---

## 4. Step 3: Solver Training

**Goal**: Train $A_\phi$ to resolve customer requests through multi-turn conversation.

**Script**: `examples/gigpo_trainer/run_tau2bench_solver.sh`

**Environment**: `Tau2BenchSolverEnvironmentManager` → `_SolverWorker`

### 4.1 Scenario Loading

```
scenarios.json
    │
    ▼
build_tau2bench_solver_envs()              ← envs.py:584
    │
    ├── Load JSON file
    ├── For each item:
    │   ├── Extract "instructions" (str)
    │   └── Extract "actions" (list of dicts)
    │   └── Store as {"instructions": str, "actions": list}
    │
    ▼
external_scenarios: List[Dict]             ← passed to each _SolverWorker
```

### 4.2 Solver Episode: reset()

```
_SolverWorker.reset()                      ← envs.py:98
    │
    ├── Select task from tau2 registry     ← provides DB state + tools
    │   task = self.tasks[task_idx]
    │
    ├── Create fresh tau2 Environment      ← self.env = self.env_constructor()
    │   (loads db.json, creates tool instances)
    │
    ├── Get tools and policy               ← self.env.get_tools() → openai_schema
    │   self.tool_schemas = [t.openai_schema for t in tools]  ← GOES TO: solver prompt
    │   self.policy = self.env.get_policy()                    ← GOES TO: solver prompt
    │
    ├── [SYNTHETIC MODE] Sample scenario
    │   scenario = self.external_scenarios[random_idx]
    │   user_instructions = scenario["instructions"]           ← GOES TO: user sim
    │   self._expected_actions = scenario["actions"]           ← STORED FOR: reward computation
    │
    ├── [STANDARD MODE] Use registry task
    │   user_instructions = task.user_scenario.instructions    ← GOES TO: user sim
    │   self._expected_actions = []                            ← no pseudo-GT needed
    │
    ├── Reset user simulator
    │   self.user_sim.reset(user_instructions)                 ← user_sim.py:112
    │   first_msg = self.user_sim.generate_first_message()     ← user_sim.py:152
    │
    ├── Record first message in tau2 format
    │   self.message_history.append(TauUserMessage(role="user", content=first_msg))
    │
    ▼
    Returns: (first_msg: str, info: dict)
             first_msg = "Hi, I'm John Doe. I need to change my flight reservation."
```

### 4.3 Solver Prompt Construction

The `Tau2BenchSolverEnvironmentManager.build_text_obs()` constructs the full prompt the solver LLM sees:

```
┌───────────────────────────────────────────────────────────────┐
│                SOLVER PROMPT (input to LLM)                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  You are a helpful customer service agent for the airline     │ ← TAU2BENCH_SOLVER_SYSTEM
│  domain.                                                      │
│                                                               │
│  ## Policy                                                    │ ← worker.policy (from env.get_policy())
│  <full policy.md text>                                        │   Source: tau2-bench/data/tau2/domains/{domain}/policy.md
│                                                               │
│  ## Available Tools                                           │ ← compact tool signatures (from env.get_tools())
│  - get_user_details(user_id) — Get user details by user id.   │   Built in _cache_system_prompts()
│  - cancel_reservation(reservation_id) — Cancel reservation.   │   ~430 tokens (vs ~2000 for full JSON schemas)
│   ...]                                                        │
│                                                               │
│  ## Output Format                                             │
│  <tool_call>{"name":..., "arguments":...}</tool_call>         │
│  <response>Your message</response>                            │
│  <response>[STOP] Final message</response>                    │
│                                                               │
│  ---                                                          │
│  [If step > 0: history of last N interactions]                │ ← SimpleMemory.fetch(history_length)
│                                                               │
│  Current message:                                             │
│  Hi, I'm John Doe. I need to change my flight reservation.   │ ← observation (user msg or tool result)
│                                                               │
│  Now respond with your reasoning and action.                  │
└───────────────────────────────────────────────────────────────┘
```

**Data source for each component**:

| Prompt Component | Source | Code Location |
|---|---|---|
| `{domain}` | `env.tau2bench.domain` config | `env_manager.py` |
| `{policy}` | `worker.policy` = `env.get_policy()` — **full text, no truncation** | `envs.py`, `env_manager.py:_cache_system_prompts()` |
| `{tool_signatures}` | Compact `- name(params) — desc` format from `worker.tool_schemas` | `env_manager.py:_cache_system_prompts()` |
| History | `SimpleMemory.fetch(history_length)` — `<think>` stripped, 600 char limit | `env_manager.py` |
| Current observation | `text_obs[i]` from `envs.step()` return | `env_manager.py:698-700` |

### 4.4 Solver Output Parsing

**Function**: `solver_projection()` in `projection.py`

```
Raw LLM text (e.g., "<think>Need to look up user.</think>\n<tool_call>{...}</tool_call>")
    │
    ▼
solver_projection()                        ← projection.py:39
    │
    ├── Search for <tool_call>...</tool_call>    via RE_TOOL_CALL regex
    ├── Search for <response>...</response>      via RE_RESPONSE regex
    │
    ├── CASE: tool_call found (no response)
    │   ├── JSON-parse content
    │   ├── Normalize to [{"name": str, "arguments": dict}, ...]
    │   └── Result: {"type": "tool_call", "calls": [...]}  valid=1
    │
    ├── CASE: response found (no tool_call)
    │   ├── Check for stop signals: ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE###
    │   │                           [STOP] / [TRANSFER] / [OUT_OF_SCOPE]
    │   ├── If stop signal: {"type": "stop", "content": text}  valid=1
    │   └── If no stop:     {"type": "response", "content": text}  valid=1
    │
    ├── CASE: both found → {"type": "invalid"}  valid=0
    └── CASE: neither found → {"type": "response", "content": raw}  valid=0
```

### 4.5 Solver Step Dispatch

```
parsed_action (from solver_projection)
    │
    ▼
_SolverWorker.step()                       ← envs.py:180
    │
    ├── type == "tool_call"
    │   └── _handle_tool_calls(calls)      ← envs.py:207
    │       │
    │       ├── For each call:
    │       │   ├── Build TauToolCall(id=f"tc_{counter}", name=..., arguments=..., requestor="assistant")
    │       │   ├── Execute: result = self.env.make_tool_call(name, requestor="assistant", **args)
    │       │   ├── Convert: result_str = self.env.to_json_str(result)
    │       │   ├── On success: append to self.tool_calls_made      ← USED IN: reward computation
    │       │   └── On error: result_str = "Error: {e}"
    │       │
    │       ├── Record in message_history:
    │       │   ├── AssistantMessage(role="assistant", tool_calls=[TauToolCall, ...])
    │       │   └── ToolMessage(id=tc_id, role="tool", content=result_str, requestor="assistant")
    │       │       (one per call, IDs match)
    │       │
    │       ├── self.env.sync_tools()                               ← syncs DB state
    │       │
    │       ▼
    │       Returns: (observation=tool_results_text, reward=0.0, done=False, info)
    │
    ├── type == "response"
    │   └── _handle_response(content)      ← envs.py:272
    │       │
    │       ├── Record: message_history.append(AssistantMessage(content=content))
    │       │
    │       ├── Call user sim: user_reply = self.user_sim.respond(content)
    │       │
    │       ├── Record: message_history.append(UserMessage(content=user_reply))
    │       │
    │       ├── Check: self.user_sim.is_stop(user_reply)?
    │       │   │
    │       │   ├── YES + is_api_fail → TerminationReason.TOO_MANY_ERRORS, reward=0, done=True
    │       │   ├── YES (normal)     → TerminationReason.USER_STOP, compute reward, done=True
    │       │   └── NO               → reward=0.0, done=False
    │       │
    │       ▼
    │       Returns: (observation=user_reply, reward, done, info)
    │
    ├── type == "stop"
    │   └── _handle_stop(content)          ← envs.py:312
    │       ├── TerminationReason.AGENT_STOP
    │       ├── Compute final reward
    │       ▼
    │       Returns: ("", reward, True, info)
    │
    └── type == "invalid"
        └── Returns: ("Error: Could not parse...", 0.0, False, info)
```

### 4.6 User Simulator Data Flow

```
                        SCENARIO INSTRUCTIONS
                        (from challenger JSON or tau2 task)
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│                   LightUserSimulator                      │
│                                                          │
│  System Prompt:                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ {tau2-bench simulation_guidelines.md}              │  │ ← loaded from tau2-bench/data/tau2/user_simulator/
│  │ (Core Principles, Task Completion rules,           │  │
│  │  ###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###)   │  │
│  │                                                    │  │
│  │ <scenario>                                         │  │
│  │ {instructions from challenger/task}                │  │ ← THIS IS THE KEY INPUT
│  │ </scenario>                                        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Conversation History (role-flipped):                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │ user_sim messages → role: "assistant"              │  │ ← matches tau2 UserState.flip_roles()
│  │ agent messages    → role: "user"                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  API: OpenAI-compatible (vLLM server on GPU 3)           │
│  Model: env.tau2bench.user_sim_model                     │
│  Temperature: 0.7                                        │
│  Max tokens: 256                                         │
│                                                          │
│  Output: natural language customer reply                  │
│          OR stop signal (###STOP### / ###TRANSFER###)    │
└──────────────────────────────────────────────────────────┘
```

### 4.7 Message History Format

The `_SolverWorker` builds a `message_history: list[Message]` that exactly matches what the tau2-bench evaluator expects. This is critical for correct evaluation.

```
message_history = [
    # --- From initial state (if standard mode) ---
    # ... (task.initial_state.message_history)

    # --- First user message ---
    UserMessage(role="user", content="Hi, I'm John Doe..."),

    # --- Agent makes tool call ---
    AssistantMessage(role="assistant", tool_calls=[
        ToolCall(id="tc_0", name="get_user_details",
                 arguments={"user_id": "john_doe_123"}, requestor="assistant")
    ]),
    ToolMessage(id="tc_0", role="tool", content='{"user_id":"john_doe_123",...}',
                requestor="assistant", error=False),

    # --- Agent responds to user ---
    AssistantMessage(role="assistant", content="I found your account. Let me update your flight."),

    # --- User replies ---
    UserMessage(role="user", content="Great, please change it to March 20."),

    # --- Agent makes another tool call ---
    AssistantMessage(role="assistant", tool_calls=[
        ToolCall(id="tc_1", name="update_reservation_flights", ...)
    ]),
    ToolMessage(id="tc_1", role="tool", content='...', requestor="assistant"),

    # --- Agent confirms ---
    AssistantMessage(role="assistant", content="Done! Your flight has been changed."),

    # --- User satisfied ---
    UserMessage(role="user", content="Thank you! ###STOP###"),
]
```

**Invariants** (violations cause evaluator crashes):
- Every `AssistantMessage` with `tool_calls` must be followed by exactly `len(tool_calls)` `ToolMessage` objects
- `ToolCall.id` must match the corresponding `ToolMessage.id`
- `ToolMessage.requestor` must match the message sender ("assistant" for agent tool calls)

---

## 5. Reward Functions Reference

### 5.1 Solver Reward — Standard Mode

**When**: `synthetic_mode = False` (using tau2 registry tasks with ground-truth evaluation criteria)

**Function**: `compute_combined_reward()` in `rewards.py:432`

```
reward = tool_call_reward_coef * r_tool_accuracy
       + task_success_reward_coef * r_task_success

where:
  r_tool_accuracy  ← compute_solver_accuracy(tool_calls_made, gt_actions)     [0, 1] continuous
  r_task_success   ← compute_task_success_reward(env, task, message_history)   {0, 1} binary
```

| Component | Function | Location | Input | Output |
|---|---|---|---|---|
| Tool-call F1 | `compute_solver_accuracy()` | `rewards.py:176` | `predicted_calls`, `ground_truth_calls` (from `task.evaluation_criteria.actions`) | reward ∈ [0, 1] |
| DB state check | `compute_task_success_reward()` | `rewards.py:374` | `env_constructor`, `task`, `message_history`, `domain`, `termination_reason` | reward ∈ {0, 1} |

**DB state check details**: Sanitizes trajectory (strips error tool calls), then replays all remaining tool calls on a fresh environment, compares DB hash with gold environment that executed the expected actions. Binary match.

### 5.2 Solver Reward — Synthetic Mode (TOD-Zero)

**When**: `synthetic_mode = True` (using challenger-generated scenarios)

**Function**: `compute_synthetic_reward()` in `rewards.py`

```
IF expected_actions available (from challenger):
    reward = 0.5 * r_completion + 0.2 * r_tool_usage + 0.3 * r_action_match

ELSE (legacy, no expected actions):
    reward = 0.7 * r_completion + 0.3 * r_tool_usage
```

| Component | Weight | Computation | Values |
|---|---|---|---|
| `r_completion` | 0.5 | `USER_STOP` → 1.0, `AGENT_STOP` → 0.5, `MAX_STEPS` → 0.1, else → 0.0 | {0.0, 0.1, 0.5, 1.0} |
| `r_tool_usage` | 0.2 | `min(1.0, len(tool_calls_made) / 2.0)` | [0, 1] |
| `r_action_match` | 0.3 | `compute_solver_accuracy(tool_calls_made, expected_actions)` — full greedy matching (name + arg keys + arg values) | [0, 1] |

**Data flow for `expected_actions`**:
```
scenarios.json["actions"]
    → build_tau2bench_solver_envs() preserves as dict
    → _SolverWorker.external_scenarios[i]["actions"]
    → _SolverWorker.reset() stores as self._expected_actions
    → _SolverWorker._compute_final_reward() passes to compute_synthetic_reward()
    → compute_synthetic_reward() calls compute_solver_accuracy(tool_calls_made, expected_actions)
```

### 5.3 Challenger Reward

**Function**: `compute_challenger_reward()` in `rewards.py`

```
reward = 0.4 * r_format + 0.3 * r_tool_validity + 0.3 * r_arg_validity
```

| Component | Weight | Function | Location |
|---|---|---|---|
| `r_format` | 0.4 | `compute_challenger_format_reward()` | `rewards.py` |
| `r_tool_validity` | 0.3 | fraction of actions with valid tool names | inside `compute_challenger_validity_reward()` |
| `r_arg_validity` | 0.3 | fraction of actions with valid arg keys | inside `compute_challenger_validity_reward()` |

### 5.4 Reward Assignment in RL Training

Both solver and challenger rewards are **sparse** (assigned at episode end). The `EpisodeRewardManager` in `reward_manager/episode.py` places the scalar episode reward on the last valid response token:

```python
reward_tensor[i, valid_response_length - 1] = score
```

This is compatible with GRPO's group-level advantage computation.

---

## 6. Data Format Specifications

### 6.1 Scenario JSON (Challenger → Solver)

```typescript
interface Scenario {
    domain: string;                    // "airline" | "retail" | "telecom"
    iter: number;                      // self-play iteration number
    instructions: string;              // natural language user goal (≥ 20 chars)
    actions: Action[];                 // expected tool call sequence
    context: {                         // DB entities scenario is grounded in (metadata only)
        user?: object;
        reservation?: object;
        order?: object;
    };
}

interface Action {
    name: string;                      // must be a valid domain tool name
    arguments: Record<string, any>;    // tool arguments
}
```

### 6.2 Solver Parsed Action (from projection)

```typescript
type SolverAction =
    | { type: "tool_call"; calls: Array<{name: string; arguments: Record<string, any>}> }
    | { type: "response"; content: string }
    | { type: "stop"; content: string }
    | { type: "invalid"; raw?: string }
```

### 6.3 Challenger Parsed Action (from projection)

```typescript
type ChallengerAction =
    | { type: "challenger_output"; instructions: string; actions: Action[] }
    | { type: "invalid"; raw?: string }
```

### 6.4 tau2-bench Message Types (for evaluator)

```
UserMessage(role="user", content=str)
AssistantMessage(role="assistant", content=str | None, tool_calls=list[ToolCall] | None)
ToolCall(id=str, name=str, arguments=dict, requestor="assistant"|"user")
ToolMessage(id=str, role="tool", content=str, requestor="assistant"|"user", error=bool)
```

---

## 7. Prompt Templates Reference

All templates are in `agent_system/environments/prompts/tau2bench.py`.

### 7.1 Solver Templates

| Template | Placeholders | Used When |
|---|---|---|
| `TAU2BENCH_SOLVER_SYSTEM` | `{domain}`, `{policy}`, `{tool_signatures}` | Always (system context) |
| `TAU2BENCH_SOLVER_TEMPLATE_NO_HIS` | `{system_prompt}`, `{current_observation}` | First step (no history) |
| `TAU2BENCH_SOLVER_TEMPLATE` | `{system_prompt}`, `{step_count}`, `{action_history}`, `{current_observation}` | Subsequent steps |

### 7.2 Challenger Templates

| Template | Placeholders | Used When |
|---|---|---|
| `TAU2BENCH_CHALLENGER_SYSTEM` | `{domain}`, `{policy}`, `{tools_text}`, `{context_text}` | Always |
| `TAU2BENCH_CHALLENGER_USER` | (none) | Always (static instruction) |
| `TAU2BENCH_CHALLENGER_TEMPLATE` | `{system_prompt}`, `{user_prompt}` | Combines system + user |

### 7.3 User Simulator Template

In `user_sim.py`:

| Template | Placeholders | Used When |
|---|---|---|
| `USER_SIM_SYSTEM_PROMPT` | `{guidelines}`, `{instructions}` | Every reset (guidelines from tau2-bench, instructions from scenario or task) |

---

## 8. Critical Invariants (Bug Checklist)

Use this list to verify correctness when debugging:

### Stop Signals
- [ ] User sim prompt uses `###STOP###` (not `[STOP]`)
- [ ] `LightUserSimulator.is_stop()` accepts both `###STOP###` and `[STOP]` formats
- [ ] `solver_projection` recognizes both formats for `type: "stop"` classification
- [ ] `TerminationReason` is never `None` when constructing `SimulationRun`

### Message History
- [ ] Every `AssistantMessage` with `tool_calls` is followed by matching `ToolMessage`(s)
- [ ] `ToolCall.id` == `ToolMessage.id` for each pair
- [ ] `ToolMessage.requestor` == "assistant" for agent tool calls
- [ ] `tool_calls_made` list only contains **successful** tool calls (no errors)
- [ ] `message_history` includes initial state messages from task (if standard mode)
- [ ] Before evaluator replay, `_sanitize_trajectory_for_evaluator()` strips error tool calls

### Scenario Data Flow
- [ ] `build_tau2bench_solver_envs` preserves full dict `{"instructions": str, "actions": list}` (not just instructions string)
- [ ] `_SolverWorker.reset()` sets `self._expected_actions` from scenario dict
- [ ] `_compute_final_reward()` passes `expected_actions` to `compute_synthetic_reward()`
- [ ] `compute_synthetic_reward()` calls `compute_solver_accuracy()` when expected_actions is non-empty

### Prompt Construction
- [ ] Solver gets `policy` from `worker.policy` (set in `_SolverWorker.reset()` from `env.get_policy()`)
- [ ] Solver gets `tool_schemas` from `worker.tool_schemas` (set from `env.get_tools()`)
- [ ] System prompt is built once per reset in `_cache_system_prompts()` — full policy, compact tool signatures
- [ ] History entries have `<think>` tags stripped and are truncated to 600 chars
- [ ] Challenger gets `context_text` from fresh `sample_context()` call on each reset

### Reward Computation
- [ ] Standard mode: `compute_combined_reward()` uses `task.evaluation_criteria.actions` as ground truth
- [ ] Synthetic mode: `compute_synthetic_reward()` uses `_expected_actions` from challenger JSON
- [ ] API fail: reward=0, `TerminationReason.TOO_MANY_ERRORS`, no evaluator call
- [ ] MAX_STEPS: `_compute_final_reward()` called with `TerminationReason.MAX_STEPS`, r_completion=0.1
- [ ] `compute_solver_accuracy()` uses greedy matching (not 1:1), handles extra call penalty

### Environment State
- [ ] Fresh `Environment` created on each `_SolverWorker.reset()` (no state leaks between episodes)
- [ ] `set_state()` only called in standard mode (synthetic mode skips initialization_actions)
- [ ] `sync_tools()` called after every tool execution

### DB Path
- [ ] `db_sampler.py` resolves `TAU2_DATA_DIR` relative to its own file location
- [ ] Expected path: `<repo_root>/tau2-bench/data/tau2/domains/{domain}/db.json`
- [ ] tau2 registry loads from its own installed package data (separate from db_sampler)