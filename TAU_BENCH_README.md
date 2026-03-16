# End-to-End Training of Tool-Calling Agents on tau-bench

This guide provides complete instructions for reproducing the **tau-bench** end-to-end training pipeline using [verl-agent](https://github.com/langfengQ/verl-agent). The pipeline trains LLM agents via reinforcement learning (GiGPO) to handle multi-turn customer service tasks with tool calling, using a self-play setup with a **solver** (the trainable agent) and a **challenger** (a trainable task generator).

> **Paper reference:** *Group-in-Group Policy Optimization for LLM Agent Training* (NeurIPS 2025) — [arXiv:2505.10978](https://arxiv.org/abs/2505.10978)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
  - [Step 1: Create Conda Environment](#step-1-create-conda-environment)
  - [Step 2: Install verl-agent](#step-2-install-verl-agent)
  - [Step 3: Install tau2-bench](#step-3-install-tau2-bench)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Option A: Full Pipeline (Recommended)](#option-a-full-pipeline-recommended)
  - [Option B: Solver Only](#option-b-solver-only)
  - [Option C: Challenger Only](#option-c-challenger-only)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
  - [Solver Configuration](#solver-configuration)
  - [Challenger Configuration](#challenger-configuration)
  - [Key Hyperparameters](#key-hyperparameters)
- [Self-Play Setup](#self-play-setup)
- [Domains](#domains)
- [Reward Functions](#reward-functions)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

The tau-bench pipeline trains tool-calling agents for customer service domains (retail, airline, telecom) using reinforcement learning. The setup consists of two components:

| Component | Role | Training Mode |
|-----------|------|---------------|
| **Solver** | Multi-turn agent that interacts with customers via tool calls | Multi-step RL (up to 30 turns) |
| **Challenger** | Task generator that creates realistic customer service scenarios | Single-step RL |

The **solver** learns to resolve customer queries by calling domain-specific tools (e.g., `get_order_details`, `cancel_booking`), while the **challenger** learns to generate diverse, well-formed task specifications. Together, they form a self-play loop where the challenger produces training tasks and the solver learns to solve them.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Full Pipeline (run_tau2bench.sh)                │
│                                                                         │
│  ┌──────────────────────────────┐    ┌────────────────────────────────┐ │
│  │   User Simulator (vLLM)     │    │   Challenger (RL-trained)      │ │
│  │   - Fixed external LLM      │    │   - Generates task specs       │ │
│  │   - Qwen2.5-7B-Instruct     │    │   - Trainable via GiGPO       │ │
│  │   - Simulates customers     │    │   - Single-step environment    │ │
│  │   - Served on GPU 3         │    │   - Trained on 2 GPUs         │ │
│  └──────────────┬───────────────┘    └────────────────────────────────┘ │
│                 │                                                       │
│                 │  user messages                                        │
│                 ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Solver Agent (RL-trained)                                       │   │
│  │   - Multi-turn tool-calling agent                                │   │
│  │   - Trainable via GiGPO (Group-in-Group Policy Optimization)    │   │
│  │   - Interacts with tau2-bench environment (tools, DB, policy)    │   │
│  │   - Base model: Qwen2.5-1.5B-Instruct (or 7B)                  │   │
│  │   - Trained on 2 GPUs                                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                 │                                                       │
│                 │  tool calls                                           │
│                 ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  tau2-bench Environment                                          │   │
│  │   - Domain tools (retail: ~15, airline: ~12, telecom: ~20)      │   │
│  │   - Database with realistic customer records                     │   │
│  │   - Domain policy (rules, constraints)                           │   │
│  │   - Reward: tool-call accuracy (name + key F1 + value match)    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

| Configuration | GPUs | Description |
|---------------|------|-------------|
| **Solver only** | 2 training + 1 user-sim = **3 GPUs** | Recommended minimum |
| **Full pipeline** | 2 training + 1 user-sim + 1 spare = **4 GPUs** | Includes challenger training |
| **Multi-node** | 4+ per node | For larger models (7B+) |

- **GPU type:** NVIDIA A100 (40GB) or H100 recommended
- **RAM:** 64GB+ system memory
- **Disk:** ~50GB for model weights and checkpoints

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n verl-agent python=3.12 -y
conda activate verl-agent
```

### Step 2: Install verl-agent

```bash
# Clone the repository
git clone https://github.com/emrecanacikgoz/verl-agent.git
cd verl-agent

# Install vLLM (inference engine for rollouts + user simulator)
pip3 install vllm==0.11.0

# Install flash-attention (for efficient transformer operations)
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Install verl-agent in editable mode
pip install -e .
```

### Step 3: Install tau2-bench

```bash
pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
```

### Verify Installation

```bash
# Verify verl-agent
python -c "import verl; print('verl-agent OK')"

# Verify tau2-bench
python -c "import tau2_bench; print('tau2-bench OK')"

# Verify vLLM
python -c "import vllm; print('vLLM OK')"

# Run tau2-bench unit tests
pytest tests/test_tau2bench.py -v
```

## Data Preparation

verl-agent uses a lightweight data preparation step. For tau2-bench, the data is only used to indicate the modality (`text`) and batch size — the actual task content comes from the tau2-bench environment at runtime.

```bash
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size 16 \
    --val_data_size 32
```

This creates parquet files at `~/data/verl-agent/text/{train,test}.parquet`.

> **Note:** The parquet files serve as placeholders. The environment dynamically generates task instances during training via `env.reset()` — no static dataset of tau-bench tasks is required.

## Training

### Option A: Full Pipeline (Recommended)

The full pipeline handles everything: starting the user simulator server, training the solver, and running evaluation.

```bash
# Default: retail domain, Qwen2.5-1.5B agent, Qwen2.5-7B user simulator
bash examples/gigpo_trainer/run_tau2bench.sh

# Airline domain
DOMAIN=airline bash examples/gigpo_trainer/run_tau2bench.sh

# Custom models
MODEL=Qwen/Qwen2.5-7B-Instruct \
USER_SIM_MODEL=Qwen/Qwen2.5-7B-Instruct \
bash examples/gigpo_trainer/run_tau2bench.sh
```

**What the pipeline does:**
1. Starts a vLLM server for user simulation (GPU 3, port 8000)
2. Trains the solver agent with GiGPO on 2 GPUs
3. Evaluates the trained checkpoint with `tau2 run`
4. Cleans up the user simulator server

### Option B: Solver Only

If you want to manage the user simulator server separately:

**Terminal 1 — Start user simulator server:**
```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name user_sim \
    --port 8000 \
    --enforce-eager \
    --tensor-parallel-size 1
```

Wait until the server reports `Uvicorn running on http://0.0.0.0:8000`.

**Terminal 2 — Train solver:**
```bash
DOMAIN=retail \
USER_SIM_URL=http://localhost:8000/v1 \
USER_SIM_MODEL=Qwen/Qwen2.5-7B-Instruct \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/gigpo_trainer/run_tau2bench_solver.sh
```

### Option C: Challenger Only

The challenger does not require a user simulator — it only generates task specifications.

```bash
DOMAIN=retail \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/gigpo_trainer/run_tau2bench_challenger.sh
```

## Evaluation

### During Training

Validation runs automatically every `test_freq` epochs (default: 5). Metrics are logged to W&B.

### Post-Training Evaluation

Use the tau2-bench CLI to evaluate a trained checkpoint:

```bash
# Evaluate on retail domain
tau2 run \
    --domain retail \
    --agent-llm /path/to/checkpoint \
    --num-trials 5 \
    --task-split-name test \
    --save-to eval_results_retail.json

# Evaluate on airline domain
tau2 run \
    --domain airline \
    --agent-llm /path/to/checkpoint \
    --num-trials 5 \
    --task-split-name test \
    --save-to eval_results_airline.json
```

### Eval-Only Mode (via pipeline script)

```bash
bash examples/gigpo_trainer/run_tau2bench.sh --eval-only /path/to/checkpoint
```

## Configuration Reference

### Solver Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.env_name` | `tau2bench_solver` | Environment identifier |
| `env.max_steps` | `30` | Maximum turns per episode |
| `env.history_length` | `4` | Number of recent steps in memory |
| `env.rollout.n` | `8` | Group size (rollouts per task) |
| `env.tau2bench.domain` | `retail` | Domain: `retail`, `airline`, `telecom` |
| `env.tau2bench.user_sim_url` | `http://localhost:8000/v1` | User simulator endpoint |
| `env.tau2bench.user_sim_model` | `Qwen/Qwen2.5-7B-Instruct` | User simulator model |
| `data.max_prompt_length` | `4096` | Max input token length |
| `data.max_response_length` | `512` | Max generation token length |
| `algorithm.adv_estimator` | `gigpo` | RL algorithm |
| `algorithm.gamma` | `0.95` | Discount factor |
| `algorithm.gigpo.step_advantage_w` | `1.0` | Step-level advantage weight |

### Challenger Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.env_name` | `tau2bench_challenger` | Environment identifier |
| `env.max_steps` | `1` | Single-step generation |
| `env.rollout.n` | `8` | Group size |
| `env.tau2bench.domain` | `retail` | Domain |
| `data.max_response_length` | `2048` | Longer for task spec generation |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | Learning rate |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | KL divergence regularization |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.01` | KL loss coefficient |
| `actor_rollout_ref.actor.use_invalid_action_penalty` | `True` | Penalize malformed actions |
| `actor_rollout_ref.actor.invalid_action_penalty_coef` | `0.1` | Invalid action penalty weight |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | Tensor parallelism for rollout |
| `trainer.n_gpus_per_node` | `2` | Training GPUs |
| `trainer.total_epochs` | `100` | Total training epochs |
| `trainer.save_freq` | `10` | Checkpoint save frequency |
| `trainer.test_freq` | `5` | Validation frequency |

## Self-Play Setup

The tau-bench pipeline supports a **self-play** configuration where the solver and challenger co-evolve:

```
┌─────────────────────┐        ┌──────────────────────┐
│  Challenger Agent    │ ─────► │  Task Specifications │
│  (generates tasks)   │        │  (question + tools   │
│                      │        │   + gold answer)     │
└─────────────────────┘        └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │  Solver Agent         │
                               │  (resolves tasks      │
                               │   via tool calls)     │
                               └──────────────────────┘
```

**To run self-play training:**

1. **Train the challenger** to generate diverse, valid task specifications:
   ```bash
   DOMAIN=retail bash examples/gigpo_trainer/run_tau2bench_challenger.sh
   ```

2. **Train the solver** on the domain using the user simulator:
   ```bash
   DOMAIN=retail bash examples/gigpo_trainer/run_tau2bench.sh
   ```

3. **Iterate:** Use challenger-generated tasks to augment solver training data, and use solver failure cases to guide challenger toward harder tasks.

The challenger is rewarded for producing well-formed task specs (50% format, 50% validity), while the solver is rewarded for tool-call accuracy (name match 20%, key F1 30%, value match 50%) with an extra-call penalty.

## Domains

| Domain | Tools | Example Tasks |
|--------|-------|---------------|
| **retail** | ~15 (orders, returns, shipping, refunds, account) | "I want to return my order #12345", "Check my delivery status" |
| **airline** | ~12 (bookings, flights, cancellations, upgrades) | "Cancel my flight ABC123", "Change my seat assignment" |
| **telecom** | ~20 (billing, plans, troubleshooting, devices) | "My internet is slow", "Upgrade my phone plan" |

## Reward Functions

### Solver (Sparse, Episode-End)

The solver reward is computed at the end of each episode based on tool-call accuracy:

```
reward = base_score * extra_call_penalty

base_score = mean over ground-truth calls:
    0.20 * name_match +      # exact tool name match
    0.30 * key_f1 +          # F1 score over argument keys
    0.50 * value_match        # robust value comparison

extra_call_penalty = 1.0 / (1.0 + 0.25 * num_extra_calls)
```

Value matching uses robust comparison: exact equality, numeric coercion with tolerance (<1e-9), and whitespace-normalized string comparison.

### Challenger (Per-Step)

```
reward = 0.50 * format_score + 0.50 * validity_score

format_score:
    +0.33 if question is non-empty (>5 chars)
    +0.33 if available_tools parses as valid JSON list
    +0.34 if tool_call_answer parses and normalizes correctly

validity_score:
    +0.40 if gold tool name exists in available_tools
    +0.40 if all required arguments are present
    +0.20 if argument values are non-empty
```

## Troubleshooting

### User simulator server won't start
- Ensure the model weights are downloaded: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct`
- Check GPU memory: the 7B user sim needs ~14GB VRAM
- Check the log: `tail -f user_sim_server.log`
- Verify port is free: `lsof -i :8000`

### Out of GPU memory during training
- Reduce `data.train_batch_size` (default: 16)
- Enable parameter offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- Reduce `env.rollout.n` (group size, default: 8)
- Use a smaller model (1.5B instead of 7B)

### Training loss is not decreasing
- Verify the user simulator is responsive: `curl http://localhost:8000/health`
- Check W&B logs for `success_rate` and `reward` metrics
- Ensure `env.tau2bench.domain` matches the intended domain
- Try increasing `trainer.total_epochs` — tau-bench tasks are complex

### Import errors
- Ensure tau2-bench is installed: `pip install git+https://github.com/emrecanacikgoz/tau2-bench.git`
- Ensure verl-agent is installed in editable mode: `pip install -e .` from repo root
- Check Python version is 3.12: `python --version`

### Data preparation fails
- The `prepare.py` script downloads from HuggingFace. Ensure internet access.
- The downloaded data is only used as a modality placeholder (see [Data Preparation](#data-preparation)).

## File Structure

```
verl-agent/
├── examples/gigpo_trainer/
│   ├── run_tau2bench.sh              # Full pipeline (server + train + eval)
│   ├── run_tau2bench_solver.sh       # Solver training only
│   └── run_tau2bench_challenger.sh   # Challenger training only
├── agent_system/
│   ├── environments/
│   │   ├── env_manager.py            # Tau2BenchSolver/ChallengerEnvironmentManager
│   │   ├── prompts/tau2bench.py      # Solver & challenger prompt templates
│   │   └── env_package/tau2bench/    # Environment implementation
│   │       ├── envs.py               # Parallel env wrappers
│   │       ├── projection.py         # XML action parsing
│   │       ├── rewards.py            # Reward computation
│   │       └── user_sim.py           # User simulator client
│   ├── memory/                       # History management (SimpleMemory)
│   ├── multi_turn_rollout/           # Step-wise trajectory collection
│   └── reward_manager/               # Episode-level reward assignment
├── verl/trainer/
│   ├── main_ppo.py                   # Training entry point (Hydra)
│   └── config/ppo_trainer.yaml       # Master config template
├── gigpo/core_gigpo.py               # GiGPO algorithm implementation
├── examples/data_preprocess/
│   └── prepare.py                    # Data placeholder generation
└── tests/test_tau2bench.py           # Unit tests
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DOMAIN` | tau2-bench domain | `retail`, `airline`, `telecom` |
| `MODEL` | Agent base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| `USER_SIM_MODEL` | User simulator model | `Qwen/Qwen2.5-7B-Instruct` |
| `USER_SIM_URL` | User simulator endpoint | `http://localhost:8000/v1` |
| `USER_SIM_PORT` | vLLM server port | `8000` |
| `USER_SIM_GPU` | GPU index for user sim | `3` |
| `ENGINE` | Inference engine | `vllm` or `sglang` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `WANDB_API_KEY` | Weights & Biases API key | — |
| `CUDA_VISIBLE_DEVICES` | GPU visibility for training | `0,1` |

## Citation

If you use this tau-bench training pipeline in your research, please cite:

```bibtex
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```
