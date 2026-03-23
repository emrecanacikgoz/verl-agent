# tau2-bench Integration for TOD-Zero Self-Play

This module integrates [tau2-bench](https://github.com/sierra-research/tau2-bench) into verl-agent, providing the environment, reward, and communication infrastructure for training multi-turn conversational agents via self-play reinforcement learning with zero human-annotated dialogue data.

---

## 1. Overview

TOD-Zero trains a task-oriented dialogue agent through an iterative self-play loop between three roles:

- **Challenger** $C_\theta$: Generates realistic customer service scenarios grounded in real database entities.
- **User Simulator** $U$: Drives multi-turn conversations by role-playing a customer with a specific goal (frozen in Paper 1; trained in Paper 2).
- **Agent (Solver)** $A_\phi$: Learns to resolve customer requests through interleaved tool calls and natural language responses.

At each iteration: (1) the Challenger generates $K$ scenarios, (2) the Agent interacts with the User Simulator on those scenarios, and (3) both Challenger and Agent update via GRPO. The tau2-bench environment provides the domain database, tool APIs, policy documents, and evaluation infrastructure.

This module contains six files, each handling a distinct responsibility:

```
tau2bench/
├── __init__.py        §2   Public API exports
├── envs.py            §3   Environment workers and parallel wrappers
├── projection.py      §4   Action parsing (text → structured actions)
├── rewards.py         §5   Reward computation
├── user_sim.py        §6   Lightweight user simulator client
├── db_sampler.py      §7   Database context sampling for Challenger
└── README.md               This document
```

---

## 2. `__init__.py` — Public API

Exports the two builder functions and two projection functions consumed by `env_manager.py`:

```python
from .projection import solver_projection, challenger_projection
from .envs import build_tau2bench_solver_envs, build_tau2bench_challenger_envs
```

The `make_envs()` factory in `env_manager.py` dispatches to these builders based on the `env.env_name` config key (`tau2bench_solver` or `tau2bench_challenger`).

---

## 3. `envs.py` — Environment Workers and Parallel Wrappers

This is the core file. It defines the single-worker logic for both Solver and Challenger, wraps them in parallelized `gym.Env` classes, and provides builder functions.

### 3.1 `_SolverWorker`

Manages a single solver episode. Each worker holds:

- A **tau2-bench `Environment`** instance (loaded fresh on each `reset()` via the domain registry). This provides the database state, tool execution, and policy document.
- A **`LightUserSimulator`** instance (§6) connected to an external vLLM server.
- A **message history** (`list[Message]`) built in the exact format the tau2-bench evaluator expects.

**`reset(kwargs)`**: Selects a task from the tau2-bench registry (provides DB state and tools). In synthetic mode (TOD-Zero), samples a challenger-generated scenario dict containing `instructions` (user goal) and `actions` (expected tool calls, used as pseudo-ground-truth for rewards). The user simulator is reset with the scenario instructions and generates the first user message. Returns the first user message as the observation.

**`step(parsed_action)`**: Dispatches to one of three handlers based on the action type produced by `solver_projection` (§4):

| Action Type | Handler | Behavior |
|---|---|---|
| `tool_call` | `_handle_tool_calls` | Executes each tool call against the tau2-bench environment via `env.make_tool_call()`, records `AssistantMessage(tool_calls=[...])` + `ToolMessage(...)` pairs in the message history, calls `env.sync_tools()`, returns tool results as observation. Intermediate reward is 0. |
| `response` | `_handle_response` | Records agent's `AssistantMessage(content=...)`, sends it to the user simulator, records the user's reply as `UserMessage(content=...)`. If the user emits a stop signal (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`), computes final reward and terminates. |
| `stop` | `_handle_stop` | Agent signals end-of-conversation. Sets `TerminationReason.AGENT_STOP`, computes final reward. |
| `invalid` | — | Returns error message, reward=0, not done. |

**`_compute_final_reward()`**: Two modes:

- **Standard mode** (registry tasks with ground-truth evaluation criteria): Calls `compute_combined_reward()` which runs both tool-call accuracy scoring (continuous, F1-based) and the tau2-bench native evaluator (binary, DB state + action matching + communication checks).
- **Synthetic mode** (TOD-Zero, challenger-generated scenarios): Calls `compute_synthetic_reward()` with three components: user satisfaction (did user say `###STOP###`?), tool usage (did agent make API calls?), and action matching (how well do the agent's tool calls match the challenger's expected actions?).

**API failure handling**: If the user simulator API fails, `TerminationReason.TOO_MANY_ERRORS` is set and reward=0 is returned immediately (no evaluation attempted).

### 3.2 `_ChallengerWorker`

Manages a single challenger episode. Each episode is single-step (generate one scenario):

**`reset(kwargs)`**: Samples a fresh database context (random user + their reservations/orders) via `db_sampler.sample_context()`. Constructs the full challenger prompt with domain info, policy, tool schemas, and sampled DB context.

**`step(parsed_action)`**: Receives the parsed challenger output from `challenger_projection`. Calls `compute_challenger_reward()` which evaluates format correctness (valid XML tags, parseable JSON) and semantic validity (tool names exist in domain, argument structure matches schemas). Returns reward ∈ [0, 1], always done=True (single-step).

### 3.3 `Tau2BenchSolverEnvs` / `Tau2BenchChallengerEnvs`

Parallel wrappers using `ThreadPoolExecutor` + `asyncio` event loop. Each wrapper holds `batch_size = env_num × group_n` workers and dispatches `reset()`/`step()` calls to all workers concurrently. This is necessary because each solver step involves an HTTP call to the user simulator (I/O bound).

### 3.4 Builder Functions

**`build_tau2bench_solver_envs(...)`**: Loads challenger scenarios from JSON if `challenger_scenarios_path` is provided (TOD-Zero mode). Preserves the full scenario dict (`instructions` + `actions`) so the solver can use expected actions as pseudo-ground-truth for reward computation.

**`build_tau2bench_challenger_envs(...)`**: Simple instantiation; the challenger has no external dependencies beyond the DB files.

---

## 4. `projection.py` — Action Parsing

Converts raw text generated by the LLM into structured action dicts consumed by the environment workers. Uses regex-based XML tag extraction.

### 4.1 `solver_projection(actions: List[str]) → (results, valids)`

Parses each agent output into one of four types:

| Output Format | Parsed Type | Valid? |
|---|---|---|
| `<tool_call>{"name":..., "arguments":...}</tool_call>` | `{"type": "tool_call", "calls": [...]}` | ✓ |
| `<response>text</response>` | `{"type": "response", "content": "text"}` | ✓ |
| `<response>###STOP### text</response>` | `{"type": "stop", "content": "..."}` | ✓ |
| Both `<tool_call>` and `<response>` present | `{"type": "invalid"}` | ✗ |
| No recognized tags | `{"type": "response", "content": raw}` | ✗ |

**Stop signal detection**: Recognizes both tau2-bench convention (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`) and legacy bracket format (`[STOP]`, `[TRANSFER]`, `[OUT_OF_SCOPE]`) for backward compatibility.

**Tool call parsing**: Accepts both single objects and JSON arrays within `<tool_call>` tags. Each tool call is normalized to `{"name": str, "arguments": dict}`.

### 4.2 `challenger_projection(actions: List[str]) → (results, valids)`

Parses challenger output into structured scenario specifications:

| Output Format | Parsed Type | Valid? |
|---|---|---|
| `<instructions>...</instructions>` + `<actions>[...]</actions>` | `{"type": "challenger_output", "instructions": str, "actions": list}` | ✓ (if instructions ≥ 20 chars, actions non-empty, all have "name") |
| Missing tags or malformed JSON | `{"type": "invalid"}` | ✗ |

---

## 5. `rewards.py` — Reward Computation

### 5.1 Solver Rewards

**`compute_solver_accuracy(predicted_calls, ground_truth_calls)`**: Greedy matching of predicted tool calls against ground truth. For each ground-truth call, finds the best-matching predicted call using a weighted score:

$$r_{\text{call}} = 0.2 \cdot \mathbb{1}[\text{name match}] + 0.3 \cdot F_1(\text{param keys}) + 0.5 \cdot \text{value match ratio}$$

Extra call penalty: $\frac{1}{1 + 0.25 \cdot \max(0, |\text{pred}| - |\text{gt}|)}$

**`compute_task_success_reward(...)`**: Runs the full tau2-bench native evaluator by constructing a `SimulationRun` and calling `evaluate_simulation(evaluation_type=EvaluationType.ALL)`. This compares:
- **DB state**: Hashes of predicted vs. gold database states after replaying tool calls
- **Action matching**: Whether all expected actions were performed (using `Action.compare_with_tool_call` with `compare_args` for flexible matching)
- **Communication**: Whether required information was communicated to the user

Returns binary 0/1.

**`compute_combined_reward(...)`**: Weighted sum used in standard (non-synthetic) mode:

$$r = w_{\text{tool}} \cdot r_{\text{tool\_accuracy}} + w_{\text{task}} \cdot r_{\text{task\_success}}$$

Default weights: $w_{\text{tool}} = w_{\text{task}} = 0.5$.

**`compute_synthetic_reward(...)`**: Used in TOD-Zero mode (challenger-generated scenarios with no ground-truth evaluation criteria). Three components:

| Component | Weight | Signal |
|---|---|---|
| $r_{\text{completion}}$ | 0.4 | USER_STOP → 1.0, AGENT_STOP → 0.5, timeout → 0.0 |
| $r_{\text{tool\_usage}}$ | 0.1 | $\min(1, n_{\text{calls}} / 2)$ |
| $r_{\text{action\_match}}$ | 0.5 | `compute_solver_accuracy` against challenger's expected actions |

When no expected actions are available (legacy scenarios), falls back to $0.7 \cdot r_{\text{completion}} + 0.3 \cdot r_{\text{tool\_usage}}$.

### 5.2 Challenger Rewards

**`compute_challenger_format_reward(parsed_action)`**: Checks structural validity of the output — presence of `<instructions>` and `<actions>` tags, minimum instruction length, parseable JSON actions, valid action structure. Returns ∈ [0, 1].

**`compute_challenger_validity_reward(parsed_action, domain_tool_names)`**: Checks semantic validity — tool names exist in domain, argument keys match expected parameters. Returns ∈ [0, 1].

**`compute_challenger_reward(...)`**: Combined: $0.4 \cdot r_{\text{format}} + 0.3 \cdot r_{\text{tool\_valid}} + 0.3 \cdot r_{\text{arg\_valid}}$.

> **Note**: This reward does not yet include a learnability signal (rewarding scenarios at the solver's competence frontier). Adding this is the next priority — see §11.

---

## 6. `user_sim.py` — Lightweight User Simulator Client

A stateless HTTP client that wraps an external vLLM-compatible OpenAI API endpoint. The user simulator LLM is frozen throughout training (Paper 1).

### 6.1 `LightUserSimulator`

**System prompt**: Instructs the LLM to role-play a customer following a specific scenario. Stop tokens match the tau2-bench convention:

- `###STOP###` — task resolved
- `###TRANSFER###` — transferred to another department
- `###OUT-OF-SCOPE###` — scenario doesn't provide enough information

**`reset(user_instructions)`**: Sets the scenario and clears conversation history.

**`generate_first_message()`**: Produces the customer's opening message to initiate the conversation.

**`respond(agent_message)`**: Generates the customer's reply to the agent. Maintains a conversation history where user sim messages are `role: "assistant"` and agent messages are `role: "user"` (role-flipped, matching tau2-bench's `UserState.flip_roles()` convention).

**Context overflow handling**: If the API returns a context-length error, estimates the overshoot and progressively drops oldest conversation turns until the request fits. This prevents training from crashing on long episodes.

**Failure tracking**: Counts consecutive API failures. After 3 consecutive failures, raises `RuntimeError` to crash the training run (better than silently producing garbage data). Single failures return `[API_FAIL]` which the solver worker handles by assigning reward=0.

**Model auto-discovery**: Class-level cache for model name resolution — if the configured model name isn't served, auto-discovers the available model. Only queries the server once per API URL across all workers.

### 6.2 `check_user_sim_connection(api_url)`

Health check called during `_SolverWorker.__init__` to fail fast if the user sim server isn't running, rather than discovering the problem mid-rollout.

---

## 7. `db_sampler.py` — Database Context Sampling

Provides the Challenger with real database entities to ground its generated scenarios.

**`sample_context(domain)`**: Samples a random user and their associated entities (reservations for airline, orders for retail) from the tau2-bench `db.json` file. Returns a dict with the sampled entities.

**`format_context_for_prompt(context)`**: Converts the sampled context into a human-readable text block inserted into the challenger's system prompt.

**`get_tools(domain)` / `get_policy(domain)`**: Load static tool schemas and policy documents from the tau2-bench data directory.

**`DOMAIN_TOOLS`**: Static dict mapping domain names to their tool schemas (used by challenger reward for validity checking).

---

## 8. Interaction Flow

### 8.1 Solver Episode (Multi-Turn)

```
                    ┌──────────────────────────────────────────┐
                    │          tau2-bench Environment           │
                    │   (DB state, tool execution, policy)      │
                    └──────────┬───────────────────────────────┘
                               │ make_tool_call() / sync_tools()
                               │
┌─────────┐  scenario  ┌──────▼──────┐  text action   ┌────────────┐
│Challenger│──────────→│ SolverWorker │←─────────────→│ User Sim    │
│(offline) │           │             │  user reply     │ (vLLM API) │
└─────────┘            │  • parse action (projection)  └────────────┘
                       │  • execute tool OR send to user
                       │  • accumulate message_history
                       │  • compute final reward
                       └──────────────────────────────────────────
```

Step-by-step for one episode:
1. `reset()`: Load task from registry, create fresh tau2 environment, sample scenario (synthetic or registry), reset user sim, generate first user message.
2. Loop until done or max_steps:
   - Agent generates text → `solver_projection` parses it
   - If `tool_call`: execute via `env.make_tool_call()`, return results as observation
   - If `response`: send to user sim, get reply; if user says `###STOP###`, compute reward, done
   - If `stop`: agent ends conversation, compute reward, done
3. `_compute_final_reward()`: Standard mode uses tau2-bench evaluator; synthetic mode uses completion + tool usage + action matching.

### 8.2 Challenger Episode (Single-Turn)

```
┌───────────────┐  DB context  ┌──────────────────┐  parsed output  ┌────────────────┐
│ db_sampler.py │────────────→│ ChallengerWorker  │───────────────→│ rewards.py      │
│ sample_context│              │ build prompt      │                │ format+validity │
└───────────────┘              │ → LLM generates   │                └────────────────┘
                               └──────────────────┘
```

1. `reset()`: Sample fresh DB context, format challenger prompt with domain/policy/tools/context.
2. `step()`: Parse LLM output via `challenger_projection`, compute format + validity reward.

### 8.3 Self-Play Iteration

Orchestrated by `run_tod_zero.sh`:
```
for iter in 1..N:
    1. Train Challenger (GRPO, 8 GPUs)         → learns to generate grounded scenarios
    2. Generate K scenarios (vLLM offline)      → challenger produces JSON of instructions+actions
    3. Start User Sim server (1 GPU)            → frozen LLM serves user responses
    4. Train Solver (GRPO, 7 GPUs)              → multi-turn RL against user sim on synthetic scenarios
    5. Merge checkpoints                        → feed into next iteration
```

---

## 9. Configuration Reference

### Solver
```yaml
env.env_name: tau2bench_solver
env.max_steps: 30                           # max turns per episode
env.history_length: 4                       # turns of history in prompt
env.rollout.n: 4                            # group size for GRPO
env.tau2bench.domain: airline               # airline | retail | telecom
env.tau2bench.user_sim_url: http://localhost:8000/v1
env.tau2bench.user_sim_model: Qwen/Qwen2.5-7B-Instruct
env.tau2bench.tool_call_reward_coef: 0.5    # weight for tool-call F1
env.tau2bench.task_success_reward_coef: 0.5 # weight for tau2 evaluator
env.tau2bench.challenger_scenarios_path: null  # path to JSON for TOD-Zero mode
```

### Challenger
```yaml
env.env_name: tau2bench_challenger
env.max_steps: 1                            # single-step generation
env.rollout.n: 4
env.tau2bench.domain: airline
```

---

## 10. Testing

```bash
# Unit tests (projections, rewards, env wrappers)
pytest tests/test_tau2bench.py -v

# End-to-end self-play (requires GPUs + tau2-bench installed)
bash run.sh
```

---

## 11. Known Limitations and Future Work

1. **No learnability reward for Challenger**: Currently the challenger is rewarded only for format and validity, not for generating scenarios at the solver's competence frontier (AZR-style $r = 1 - \bar{r}_{\text{solve}}$ when $\bar{r}_{\text{solve}} > 0$, else 0).

2. **No persona diversity**: The user simulator uses a fixed system prompt template. The challenger does not generate personas, and the user sim does not consume them. Adding persona generation to the challenger output and persona-conditioned behavior to the user sim would increase training distribution diversity.

3. **User Simulator is frozen**: Paper 1 does not train $U_\gamma$. Paper 2 will add co-evolution where the user simulator is also trained via RL to generate realistic and challenging user behaviors.

4. **Synthetic reward is approximate**: The action-matching component of `compute_synthetic_reward` uses the challenger's expected actions as pseudo-ground-truth. These are noisy (the challenger may propose incorrect action sequences). As the challenger improves across iterations, this signal becomes more reliable.