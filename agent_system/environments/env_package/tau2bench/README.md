# tau2-bench Environment for verl-agent

This module integrates [tau2-bench](https://github.com/sierra-research/tau2-bench) into verl-agent, enabling RL training of tool-calling customer service agents.

## Overview

tau2-bench is a benchmark for evaluating conversational agents in dual-control environments (agent + user interact with shared tools/databases). This integration provides two environment types:

| Environment | Config Name | Description |
|-------------|-------------|-------------|
| **Solver** | `tau2bench_solver` | Multi-turn: agent interacts with a user simulator and domain tools to resolve customer service tasks |
| **Challenger** | `tau2bench_challenger` | Single-step: agent generates realistic task specifications (question + tools + gold answer) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  verl-agent Training Loop (GiGPO / GRPO / PPO)         │
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │  Tau2BenchSolverEnvironmentManager          │        │
│  │    ├── Memory (history management)          │        │
│  │    ├── Prompt construction (system + tools) │        │
│  │    └── Projection (parse <tool_call> tags)  │        │
│  └────────────────┬────────────────────────────┘        │
│                   │                                     │
│  ┌────────────────▼────────────────────────────┐        │
│  │  Tau2BenchSolverEnvs (ThreadPoolExecutor)   │        │
│  │    ├── _SolverWorker[0] ──┐                 │        │
│  │    ├── _SolverWorker[1]   │  Parallel       │        │
│  │    └── _SolverWorker[N]   │  Execution      │        │
│  └───────────────────────────┘                  │        │
│              │         │                                │
│     ┌────────┘         └──────────┐                     │
│     ▼                             ▼                     │
│  tau2-bench Environment      User Simulator             │
│  (tools, DB, policy)        (vLLM endpoint)             │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install tau2-bench
pip install git+https://github.com/emrecanacikgoz/tau2-bench.git

# Install verl-agent (from repo root)
pip install -e .
```

## Configuration

### Solver Environment

Add to your training config (Hydra overrides):

```bash
env.env_name=tau2bench_solver
env.max_steps=30
env.history_length=4
env.rollout.n=8
env.tau2bench.domain=retail          # retail, airline, telecom
env.tau2bench.user_sim_url=http://localhost:8000/v1
env.tau2bench.user_sim_model=Qwen/Qwen2.5-1.5B-Instruct  # same base LLM
```

### Challenger Environment

```bash
env.env_name=tau2bench_challenger
env.max_steps=1
env.rollout.n=8
env.tau2bench.domain=retail
```

## Action Format

### Solver Actions

The agent outputs text with XML tags, parsed by `solver_projection`:

```xml
<think>I need to look up the order details first.</think>
<tool_call>{"name": "get_order_details", "arguments": {"order_id": "12345"}}</tool_call>
```

```xml
<think>The order has been found. Let me inform the customer.</think>
<response>Your order #12345 is currently being shipped.</response>
```

```xml
<think>Issue resolved.</think>
<response>[STOP] Your request has been processed. Is there anything else?</response>
```

### Challenger Actions

The challenger outputs task specifications, parsed by `challenger_projection`:

```xml
<think>I'll create a flight cancellation task.</think>
<question>I need to cancel my flight booking ABC123.</question>
<available_tools>[{"name": "cancel_booking", "parameters": {...}}]</available_tools>
<tool_call_answer>{"name": "cancel_booking", "arguments": {"booking_id": "ABC123"}}</tool_call_answer>
```

## Reward Functions

### Solver Rewards (Sparse, Episode-End)

Combined reward from three components:
- **Format reward** (W=0.1): Fraction of well-formatted agent actions (valid `<tool_call>`/`<response>` tags)
- **Tool-call accuracy** (W=0.4): Granular accuracy with three sub-components:
  - Name match (20%): Exact tool name matching
  - Key F1 (30%): F1 score over argument keys
  - Value match (50%): Robust value comparison (handles numeric coercion, whitespace)
  - Extra call penalty: `1 / (1 + 0.25 * num_extra_calls)`
- **Task success** (W=0.5): Binary 0/1 from tau-bench environment evaluation (1.0 if tool-call accuracy >= 0.99)

### Challenger Rewards

- **Format** (50%): Valid XML tags, parseable JSON tools/answer
- **Validity** (50%): Tool exists in schema, required args present, values non-empty

## Files

```
tau2bench/
├── __init__.py           # Exports builder functions and projections
├── envs.py               # Parallel env wrappers (Solver + Challenger)
├── projection.py         # Action parsing (XML tag extraction)
├── rewards.py            # Reward computation (accuracy + format)
├── user_sim.py           # Lightweight user simulator (OpenAI-compatible API)
└── README.md             # This file
```

## Domains

| Domain | Description | Tools |
|--------|-------------|-------|
| **retail** | E-commerce (orders, returns, shipping) | ~15 tools |
| **airline** | Flight booking and management | ~12 tools |
| **telecom** | Tech support and troubleshooting | ~20 tools |

## Running Tests

```bash
pytest tests/test_tau2bench.py -v
```
