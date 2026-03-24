# TOD-Zero: Multi-turn Self-Play for Task-Oriented Dialogue

This guide covers the **TOD-Zero** training pipeline built on [verl-agent](https://github.com/langfengQ/verl-agent). TOD-Zero trains task-oriented dialogue (TOD) agents via **self-play reinforcement learning**, requiring **zero human-authored dialogues** — only API schemas and a target domain as supervision.

> **Paper:** *TOD-Zero: Multi-turn Self-Play Converts Weak LLMs into Strong Task-Oriented Dialogue Agents*
> **Benchmark:** [Tau-Bench 2](https://github.com/emrecanacikgoz/tau2-bench)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Option A: TOD-Zero Self-Play (Recommended)](#option-a-tod-zero-self-play-recommended)
  - [Option B: Solver Only (Standard RL)](#option-b-solver-only-standard-rl)
  - [Option C: Challenger Only](#option-c-challenger-only)
  - [Option D: Full Pipeline + Eval](#option-d-full-pipeline--eval)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [Reward Functions](#reward-functions)
- [Domains](#domains)
- [File Structure](#file-structure)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

TOD-Zero eliminates the data bottleneck for training task-oriented dialogue agents. Traditional TOD systems require thousands of human-annotated multi-turn dialogues per domain. TOD-Zero replaces this with self-play RL using only:

- **API schemas** (tool definitions)
- **Domain policy** (business rules)
- **A base LLM** (no domain-specific fine-tuning needed to start)

| Component | Role | Trained? |
|-----------|------|----------|
| **Challenger (Cθ)** | Given domain + APIs → generates user goals (what customers want) | ✅ GRPO |
| **User Simulator (U)** | Given user goals → simulates customer conversation turns | ❌ Fixed |
| **Solver/Agent (Sφ)** | Given user turns + APIs → makes tool calls or responds | ✅ GRPO |

All three start from the same base LLM. The challenger and solver co-evolve across iterations.

---

## Architecture

### TOD-Zero Self-Play Loop

```
Initialize: Challenger_0, Solver_0, UserSim = base_model (all identical)

For each iteration i = 1..N:

  ┌─────────────────────────────────────────────────────────────┐
  │  Step 1: Train Challenger_i                                 │
  │                                                             │
  │  Input:  Domain policy + API schemas                        │
  │  Output: <instructions> + <actions> (user scenario + expected tool calls) │
  │  Reward: format validity + tool name validity + arg key validity │
  └────────────────────────┬────────────────────────────────────┘
                           │
                           ▼ (generate_challenger_scenarios.py)
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 2: Generate K user goals offline                      │
  │          → iter{i}_scenarios.json                           │
  └────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
  ┌──────────────────┐     │     ┌──────────────────────────────┐
  │  User Simulator  │◄────┘     │  Solver_i (trains via GRPO)  │
  │  (fixed, vLLM)   │           │                              │
  │  Given user goal:│           │  Per turn:                   │
  │  simulates the   │◄─────────►│  <tool_call>{...}</tool_call>│
  │  customer        │           │  <response>...</response>    │
  └──────────────────┘           │                              │
           │                     │  Reward:                     │
           │ [STOP] = success     │  0.5 × user satisfaction    │
           └─────────────────────│  0.2 × tool usage + 0.3 × action match           │
                                 └──────────────────────────────┘
                                           │
                                 ┌─────────▼──────────────────┐
                                 │  tau2-bench Environment     │
                                 │  - Real database state      │
                                 │  - Domain tools & policy   │
                                 │  - Tool execution engine   │
                                 └────────────────────────────┘
```

### Key Design Principles

1. **Zero human dialogues**: The challenger generates all training scenarios from API schemas + domain policy
2. **Realistic tool execution**: The tau2-bench environment provides real DB state for tool calls
3. **Curriculum via self-play**: Challenger scenarios become harder as the solver improves
4. **Clean evaluation**: Final eval uses the standard tau2-bench test set with human-written scenarios

---

## Hardware Requirements

| Mode | GPUs | Notes |
|------|------|-------|
| **TOD-Zero self-play** | 4 GPUs | 2 training + 1 user-sim + 1 scenario gen |
| **Solver only** | 3 GPUs | 2 training + 1 user-sim |
| **Challenger only** | 2 GPUs | Training only, no user sim needed |

- **GPU type:** NVIDIA A100 (40GB) or H100 recommended
- **RAM:** 64GB+ system memory
- **Disk:** ~50GB for model weights and checkpoints

---

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n verl-agent python=3.12 -y
conda activate verl-agent
```

### Step 2: Install verl-agent

```bash
git clone https://github.com/emrecanacikgoz/verl-agent.git
cd verl-agent

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### Step 3: Install tau2-bench

```bash
pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
```

### Verify

```bash
python -c "import verl; print('verl-agent OK')"
python -c "from tau2.registry import registry; print('tau2-bench OK')"
python -c "import vllm; print('vLLM OK')"
```

---

## Data Preparation

verl-agent uses a lightweight placeholder step. The actual task content comes from the tau2-bench environment (and challenger scenarios) at runtime.

```bash
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size 16 \
    --val_data_size 32
```

Creates parquet files at `~/data/verl-agent/text/{train,test}.parquet`.

---

## Training

### Option A: TOD-Zero Self-Play (Recommended)

Runs the full iterative self-play loop: train challenger → generate scenarios → train solver → repeat.

```bash
# Default: airline domain, Qwen2.5-3B, 5 iterations, 1000 scenarios/iter
bash examples/gigpo_trainer/run_tod_zero.sh

# Custom configuration
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
DOMAIN=airline \
ITERATIONS=5 \
N_SCENARIOS=1000 \
CHALLENGER_EPOCHS=30 \
SOLVER_EPOCHS=50 \
bash examples/gigpo_trainer/run_tod_zero.sh
```

**What each iteration does:**
1. Trains Challenger_i with GRPO (generates user goals from domain + APIs)
2. Runs `generate_challenger_scenarios.py` to produce K user goals offline
3. Starts the user simulator (fixed base model via vLLM)
4. Trains Solver_i with GRPO using challenger-generated goals + user simulator
5. Saves checkpoints, updates for next iteration

**Output structure:**
```
tod_zero_airline/
├── iter1_challenger/          # Challenger checkpoint (iter 1)
├── iter1_scenarios.json       # 1000 challenger-generated user goals
├── iter1_solver/              # Solver checkpoint (iter 1)
├── iter2_challenger/
├── iter2_scenarios.json
├── iter2_solver/
├── ...
└── eval_final_airline.json    # Final tau2-bench evaluation
```

---

### Option B: Solver Only (Standard RL)

Trains the solver against static tau2-bench tasks (no self-play, no challenger).

**Terminal 1 — Start user simulator:**
```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name user_sim \
    --port 8000 \
    --enforce-eager \
    --tensor-parallel-size 1
```

**Terminal 2 — Train solver:**
```bash
DOMAIN=retail \
USER_SIM_URL=http://localhost:8000/v1 \
USER_SIM_MODEL=Qwen/Qwen2.5-3B-Instruct \
MODEL=Qwen/Qwen2.5-3B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/gigpo_trainer/run_tau2bench_solver.sh
```

**With challenger-generated scenarios (TOD-Zero, single iteration):**
```bash
# First generate scenarios from a trained challenger
python examples/gigpo_trainer/generate_challenger_scenarios.py \
    --model ./path/to/challenger/checkpoint \
    --domain airline \
    --n 1000 \
    --output ./my_scenarios.json

# Then train solver with those scenarios
CHALLENGER_SCENARIOS_PATH=./my_scenarios.json \
DOMAIN=airline \
bash examples/gigpo_trainer/run_tau2bench_solver.sh
```

---

### Option C: Challenger Only

Trains the challenger to generate user goals from domain + API schemas.

```bash
DOMAIN=retail \
MODEL=Qwen/Qwen2.5-3B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/gigpo_trainer/run_tau2bench_challenger.sh
```

After training, generate scenarios:
```bash
python examples/gigpo_trainer/generate_challenger_scenarios.py \
    --model ./checkpoints/challenger/global_step_30 \
    --domain retail \
    --n 1000 \
    --output ./retail_scenarios.json \
    --temperature 0.9 \
    --batch_size 64
```

---

### Option D: Full Pipeline + Eval

One-shot script that handles everything: user sim, solver training, and evaluation.

```bash
# Default setup
bash examples/gigpo_trainer/run_tau2bench.sh

# Airline domain
DOMAIN=airline bash examples/gigpo_trainer/run_tau2bench.sh

# Eval only on a saved checkpoint
bash examples/gigpo_trainer/run_tau2bench.sh --eval-only /path/to/checkpoint
```

---

## Evaluation

### During Training

Validation runs every `test_freq` epochs (default: 5) using tau2-bench registry tasks. Metrics logged to W&B:
- `success_rate`: fraction of tasks fully resolved
- `{domain}_success_rate`: per-domain breakdown
- `reward`: mean episode reward

### Post-Training

```bash
# Evaluate on specific domain
tau2 run \
    --domain airline \
    --agent-llm /path/to/solver/checkpoint \
    --num-trials 5 \
    --task-split-name test \
    --save-to eval_airline.json

# Multi-domain
for domain in retail airline telecom; do
    tau2 run \
        --domain $domain \
        --agent-llm /path/to/checkpoint \
        --num-trials 5 \
        --task-split-name test \
        --save-to eval_${domain}.json
done
```

> **Note:** Final evaluation always uses the standard tau2-bench test set with human-written scenarios, regardless of whether the model was trained with self-play or static tasks. This ensures fair comparison.

---

## Configuration Reference

### Solver

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.env_name` | `tau2bench_solver` | Environment identifier |
| `env.max_steps` | `30` | Max turns per episode |
| `env.history_length` | `4` | Recent steps kept in memory |
| `env.rollout.n` | `8` | Group size (rollouts per task) |
| `env.tau2bench.domain` | `retail` | Domain: `retail`, `airline`, `telecom` |
| `env.tau2bench.user_sim_url` | `http://localhost:8000/v1` | User simulator endpoint |
| `env.tau2bench.user_sim_model` | `Qwen/Qwen2.5-3B-Instruct` | User simulator model |
| `env.tau2bench.challenger_scenarios_path` | `null` | Path to challenger scenarios JSON (TOD-Zero mode) |
| `env.tau2bench.tool_call_reward_coef` | `0.5` | Tool-call accuracy weight (standard mode) |
| `env.tau2bench.task_success_reward_coef` | `0.5` | Task success weight (standard mode) |
| `algorithm.adv_estimator` | `grpo` | RL algorithm |
| `algorithm.gamma` | `0.95` | Discount factor |
| `data.max_prompt_length` | `4096` | Max input token length |
| `data.max_response_length` | `512` | Max generation token length |

### Challenger

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.env_name` | `tau2bench_challenger` | Environment identifier |
| `env.max_steps` | `1` | Single-step generation |
| `env.rollout.n` | `8` | Group size |
| `env.tau2bench.domain` | `retail` | Domain |
| `data.max_response_length` | `2048` | Longer for goal generation |
| `algorithm.adv_estimator` | `gigpo` | RL algorithm |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | Learning rate |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | KL regularization |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.01` | KL coefficient |
| `actor_rollout_ref.actor.use_invalid_action_penalty` | `True` | Penalize malformed actions |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | Tensor parallelism |
| `trainer.n_gpus_per_node` | `2` | Training GPUs |
| `trainer.total_epochs` | `100` | Total epochs (override with `SOLVER_EPOCHS` / `CHALLENGER_EPOCHS`) |
| `trainer.save_freq` | `10` | Checkpoint save interval |

---

## Reward Functions

### Solver — Standard Mode (tau2-bench eval criteria available)

Used when training on tau2-bench registry tasks (human-written scenarios).

```
combined = tool_call_reward_coef × tool_call_reward
         + task_success_reward_coef × task_success_reward

tool_call_reward = base_score × extra_call_penalty
  base_score = mean over GT calls:
      0.20 × name_match   (exact tool name)
      0.30 × key_f1       (argument key F1)
      0.50 × value_match  (robust value comparison)
  extra_call_penalty = 1 / (1 + 0.25 × extra_calls)

task_success_reward = tau2-bench native evaluator
  (DB state match + action match + communication check)
```

### Solver — TOD-Zero Mode (challenger-generated scenarios)

Used when training with `challenger_scenarios_path`. No ground-truth eval criteria needed.

```
IF expected_actions available (from challenger):
    reward = 0.5 × completion + 0.2 × tool_usage + 0.3 × action_match

ELSE (no expected actions):
    reward = 0.7 × completion + 0.3 × tool_usage

completion_reward:
    1.0  if user simulator signals ###STOP###  (customer satisfied)
    0.5  if agent signals stop                 (partial credit)
    0.1  if episode hits max_steps             (made progress but timed out)
    0.0  otherwise                             (no resolution)

tool_usage_reward:
    min(1.0, n_successful_tool_calls / 2)
    (reward for making at least 2 successful API calls)

action_match_reward:
    Greedy matching of solver's tool calls vs challenger's expected actions.
    Scores name match (0.2), argument key F1 (0.3), argument value match (0.5).
    Challenger arguments are grounded in real DB context, so this is a
    meaningful signal, not a random guess.
```

### Challenger — Scenario Quality

```
reward = 0.4 × R_format + 0.3 × R_tool_validity + 0.3 × R_arg_validity

R_format (0.0–1.0):
    +0.25  <instructions> tag present and non-empty
    +0.25  instructions length ≥ 20 chars
    +0.25  <actions> tag parses as JSON list with ≥1 item
    +0.25  every action has "name" (str) and "arguments" (dict)

R_tool_validity (0.0–1.0):
    Fraction of action names that exist in the domain's tool set

R_arg_validity (0.0–1.0):
    Fraction of valid argument keys per action (matched against tool schema)
```

---

## Domains

| Domain | Tools | Example Scenarios |
|--------|-------|-------------------|
| **retail** | ~15 (orders, returns, shipping, refunds, account mgmt) | "I want to return an item I ordered last week", "Check where my package is" |
| **airline** | ~12 (bookings, flights, cancellations, upgrades, seats) | "I need to cancel my upcoming flight", "Change my seat to an aisle" |
| **telecom** | ~20 (billing, plans, troubleshooting, devices, SIM) | "My internet has been slow for 3 days", "Upgrade to unlimited data plan" |

---

## File Structure

```
verl-agent/
├── examples/gigpo_trainer/
│   ├── run_tod_zero.sh                  # TOD-Zero self-play loop (NEW)
│   ├── generate_challenger_scenarios.py # Offline scenario generation (NEW)
│   ├── run_tau2bench.sh                 # Full pipeline (solver + eval)
│   ├── run_tau2bench_solver.sh          # Solver training only
│   └── run_tau2bench_challenger.sh      # Challenger training only
├── agent_system/
│   ├── environments/
│   │   ├── env_manager.py               # Environment factory (tau2bench branch)
│   │   ├── prompts/tau2bench.py         # Solver & challenger prompt templates
│   │   └── env_package/tau2bench/
│   │       ├── envs.py                  # Parallel env wrappers (solver + challenger)
│   │       ├── projection.py            # XML action parsing
│   │       ├── rewards.py               # Reward functions (standard + synthetic)
│   │       └── user_sim.py              # User simulator HTTP client
│   ├── memory/                          # History management
│   └── multi_turn_rollout/              # Step-wise trajectory collection
└── verl/trainer/config/ppo_trainer.yaml # Master config (tau2bench section)
```

---

## Environment Variables

### TOD-Zero Self-Play (`run_tod_zero.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Base LLM for all components |
| `DOMAIN` | `airline` | tau2-bench domain |
| `ITERATIONS` | `5` | Number of self-play iterations |
| `N_SCENARIOS` | `1000` | Scenarios generated per iteration |
| `CHALLENGER_EPOCHS` | `30` | Training epochs for challenger per iteration |
| `SOLVER_EPOCHS` | `50` | Training epochs for solver per iteration |
| `USER_SIM_MODEL` | `$BASE_MODEL` | User simulator model (fixed, not trained) |
| `USER_SIM_PORT` | `8000` | vLLM server port for user sim |
| `USER_SIM_GPU` | `3` | GPU index for user sim |
| `TRAIN_GPUS` | `0,1` | GPUs for challenger/solver training |
| `GEN_GPUS` | `2,3` | GPUs for offline scenario generation |
| `BASE_DIR` | `./tod_zero_${DOMAIN}` | Output directory |

### Solver (`run_tau2bench_solver.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAIN` | `retail` | tau2-bench domain |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Agent base model |
| `USER_SIM_URL` | `http://localhost:8000/v1` | User simulator endpoint |
| `USER_SIM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | User simulator model name |
| `CHALLENGER_SCENARIOS_PATH` | `null` | Path to scenarios JSON (TOD-Zero mode) |
| `SOLVER_EPOCHS` | `100` | Training epochs |
| `SOLVER_CKPT_DIR` | `checkpoints/...` | Checkpoint output directory |
| `TOOL_CALL_REWARD_COEF` | `0.5` | Tool-call accuracy weight (standard mode) |
| `TASK_SUCCESS_REWARD_COEF` | `0.5` | Task success weight (standard mode) |

### Challenger (`run_tau2bench_challenger.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAIN` | `retail` | tau2-bench domain |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Challenger base model |
| `CHALLENGER_EPOCHS` | `100` | Training epochs |
| `CHALLENGER_CKPT_DIR` | `checkpoints/...` | Checkpoint output directory |

---

## Troubleshooting

### User simulator server won't start
```bash
# Check GPU memory (3B model needs ~7GB, 7B needs ~14GB)
nvidia-smi

# Check logs
tail -f user_sim_server.log

# Verify port is free
lsof -i :8000

# Download model weights if needed
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

### Out of GPU memory during training
```bash
# Reduce batch size
data.train_batch_size=8

# Enable parameter offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True

# Reduce group size
env.rollout.n=4
```

### Challenger generates poor scenarios
- Increase `CHALLENGER_EPOCHS` (more training time)
- Check W&B for `challenger_success_rate` — should climb above 0.5
- Verify the model is following `<instructions>...</instructions>` and `<actions>[...]</actions>` format

### Solver reward is always 0 in TOD-Zero mode
- Verify user simulator is running: `curl http://localhost:8000/health`
- Check `user_sim_server.log` for errors
- Ensure `CHALLENGER_SCENARIOS_PATH` points to a valid JSON file with `"instructions"` and `"actions"` fields
- Verify scenarios loaded: look for `Loaded N challenger scenarios` in training logs
- Check conv_logs (if enabled) for conversation traces to diagnose agent behavior

### Import errors
```bash
pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
pip install -e .  # from repo root
python --version  # must be 3.12
```

---

## Citation

If you use TOD-Zero in your research, please cite:

```bibtex
@article{acikgoz2025tod,
  title={TOD-Zero: Multi-turn Self-Play Converts Weak LLMs into Strong Task-Oriented Dialogue Agents},
  author={Acikgoz, Emrecan and others},
  year={2025}
}
```

If you use the verl-agent framework or GiGPO algorithm:

```bibtex
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```
