#!/bin/bash
# ============================================================================
# TOD-Zero: Multi-turn Self-Play for Task-Oriented Dialogue
# ============================================================================
# Co-evolves a Challenger (generates user goals) and a Solver (handles them)
# using self-play RL with zero human-authored dialogue data.
#
# Algorithm (per iteration):
#   1. Train Challenger: given domain + API schemas → generate user goals
#   2. Generate scenarios: run challenger offline to produce K user goals
#   3. Train Solver: interact with user simulator using challenger goals
#
# Usage:
#   bash run_tod_zero.sh                          # defaults
#   DOMAIN=airline ITERATIONS=5 bash run_tod_zero.sh
#
# Requirements:
#   - 8x GPUs: 7 for training, 1 for user sim
#   - tau2-bench: pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
#   - verl-agent: pip install -e . (from repo root)
#
# Environment variables:
#   BASE_MODEL, DOMAIN, ITERATIONS, N_SCENARIOS
#   CHALLENGER_EPOCHS, SOLVER_EPOCHS
#   USER_SIM_MODEL, USER_SIM_PORT, USER_SIM_GPU
#   HF_HOME, WANDB_API_KEY, WANDB_DIR
# ============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
DOMAIN=${DOMAIN:-"airline"}
ITERATIONS=${ITERATIONS:-5}
N_SCENARIOS=${N_SCENARIOS:-1000}         # scenarios generated per iteration

# Training durations (epochs in verl RL training)
CHALLENGER_EPOCHS=${CHALLENGER_EPOCHS:-75}
SOLVER_EPOCHS=${SOLVER_EPOCHS:-100}

# User simulator (fixed, not trained)
USER_SIM_MODEL=${USER_SIM_MODEL:-"$BASE_MODEL"}
USER_SIM_PORT=${USER_SIM_PORT:-8000}
USER_SIM_GPU=${USER_SIM_GPU:-7}

# GPU allocation — steps are sequential, so each step maximizes GPU usage:
# Challenger training:   GPUs 0-7 (all 8, no user sim needed)
# Scenario generation:   GPUs 0-7 (all 8, vLLM TP=8)
# Solver training:       GPUs 0-6 (7 GPUs, rollout TP=1 so any count works)
# User sim server:       GPU 7   (runs concurrently with solver training)

# Output directory
BASE_DIR=${BASE_DIR:-"./tod_zero_${DOMAIN}"}

export USER_SIM_URL="http://localhost:${USER_SIM_PORT}/v1"
export WANDB_PROJECT=${WANDB_PROJECT:-"tod-zero-${DOMAIN}"}
export HF_HOME=${HF_HOME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DIR=${WANDB_DIR:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================"
echo "  TOD-Zero Self-Play"
echo "================================================================"
echo "  Base model:  $BASE_MODEL"
echo "  Domain:      $DOMAIN"
echo "  Iterations:  $ITERATIONS"
echo "  N scenarios: $N_SCENARIOS per iteration"
echo "  Output dir:  $BASE_DIR"
echo "================================================================"

mkdir -p "$BASE_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: start user simulator vLLM server
# ─────────────────────────────────────────────────────────────────────────────
start_user_sim() {
    local model_path="$1"
    echo "[user-sim] Starting on GPU ${USER_SIM_GPU}, port ${USER_SIM_PORT}: $model_path"

    pkill -f "vllm.entrypoints.openai.api_server.*--port ${USER_SIM_PORT}" 2>/dev/null || true
    sleep 2

    CUDA_VISIBLE_DEVICES=$USER_SIM_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name user_sim \
        --port "$USER_SIM_PORT" \
        --enforce-eager \
        --tensor-parallel-size 1 > "${BASE_DIR}/user_sim.log" 2>&1 &

    echo "[user-sim] Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${USER_SIM_PORT}/health" > /dev/null 2>&1; then
            echo "[user-sim] Ready!"
            return 0
        fi
        sleep 5
    done
    echo "[user-sim] ERROR: server failed to start. Check ${BASE_DIR}/user_sim.log"
    exit 1
}

stop_user_sim() {
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${USER_SIM_PORT}" 2>/dev/null || true
    sleep 2
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: find the latest saved checkpoint in a directory
# ─────────────────────────────────────────────────────────────────────────────
find_latest_checkpoint() {
    local dir="$1"
    # verl saves under checkpoints/<project>/<experiment>/global_step_*
    local ckpt
    ckpt=$(ls -td "${dir}"/global_step_* 2>/dev/null | head -1)
    echo "$ckpt"
}

# ─────────────────────────────────────────────────────────────────────────────
# Self-play loop
# ─────────────────────────────────────────────────────────────────────────────
CHALLENGER_PREV="$BASE_MODEL"
SOLVER_PREV="$BASE_MODEL"

for (( ITER=1; ITER<=ITERATIONS; ITER++ )); do
    echo ""
    echo "================================================================"
    echo "  ITERATION ${ITER} / ${ITERATIONS}"
    echo "================================================================"

    CHALLENGER_CKPT_DIR="${BASE_DIR}/iter${ITER}_challenger"
    SOLVER_CKPT_DIR="${BASE_DIR}/iter${ITER}_solver"
    SCENARIOS_FILE="${BASE_DIR}/iter${ITER}_scenarios.json"

    mkdir -p "$CHALLENGER_CKPT_DIR" "$SOLVER_CKPT_DIR"

    # ------------------------------------------------------------------
    # Step 1: Train Challenger
    # ------------------------------------------------------------------
    echo ""
    echo "[Iter ${ITER}] Step 1: Training Challenger (from: $CHALLENGER_PREV)..."

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    N_GPUS_PER_NODE=8 \
    MODEL=$CHALLENGER_PREV \
    DOMAIN=$DOMAIN \
    CHALLENGER_EPOCHS=$CHALLENGER_EPOCHS \
    CHALLENGER_CKPT_DIR=$CHALLENGER_CKPT_DIR \
    bash "$SCRIPT_DIR/run_tau2bench_challenger.sh"

    # Find saved challenger checkpoint
    CHALLENGER_CKPT=$(find_latest_checkpoint "$CHALLENGER_CKPT_DIR")
    if [ -z "$CHALLENGER_CKPT" ]; then
        echo "[Iter ${ITER}] ERROR: No challenger checkpoint found in $CHALLENGER_CKPT_DIR"
        exit 1
    fi
    echo "[Iter ${ITER}] Challenger checkpoint: $CHALLENGER_CKPT"

    # Merge FSDP shards → HuggingFace format so vLLM can load the checkpoint
    CHALLENGER_HF_DIR="${CHALLENGER_CKPT}_hf"
    echo "[Iter ${ITER}] Merging FSDP shards → HF model at $CHALLENGER_HF_DIR"
    python "$SCRIPT_DIR/../../scripts/model_merger.py" merge \
        --backend fsdp \
        --local_dir "${CHALLENGER_CKPT}/actor" \
        --target_dir "$CHALLENGER_HF_DIR"

    # ------------------------------------------------------------------
    # Step 2: Generate user goal scenarios with trained challenger
    # ------------------------------------------------------------------
    echo ""
    echo "[Iter ${ITER}] Step 2: Generating ${N_SCENARIOS} scenarios → ${SCENARIOS_FILE}"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python "$SCRIPT_DIR/generate_challenger_scenarios.py" \
        --model "$CHALLENGER_HF_DIR" \
        --domain "$DOMAIN" \
        --n "$N_SCENARIOS" \
        --output "$SCENARIOS_FILE" \
        --iter "$ITER" \
        --batch_size 64 \
        --temperature 0.7 \
        --tensor_parallel_size 4

    N_GENERATED=$(python -c "import json; d=json.load(open('${SCENARIOS_FILE}')); print(len(d))" 2>/dev/null || echo "?")
    echo "[Iter ${ITER}] Generated ${N_GENERATED} valid scenarios."

    # ------------------------------------------------------------------
    # Step 3: Start user simulator (fixed, never trained)
    # ------------------------------------------------------------------
    echo ""
    echo "[Iter ${ITER}] Step 3: Starting user simulator (fixed: $USER_SIM_MODEL)..."
    start_user_sim "$USER_SIM_MODEL"

    # ------------------------------------------------------------------
    # Step 4: Train Solver using challenger-generated scenarios
    # ------------------------------------------------------------------
    echo ""
    echo "[Iter ${ITER}] Step 4: Training Solver (from: $SOLVER_PREV)..."

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    N_GPUS_PER_NODE=7 \
    MODEL=$SOLVER_PREV \
    DOMAIN=$DOMAIN \
    SOLVER_EPOCHS=$SOLVER_EPOCHS \
    SOLVER_CKPT_DIR=$SOLVER_CKPT_DIR \
    CHALLENGER_SCENARIOS_PATH=$SCENARIOS_FILE \
    USER_SIM_URL=$USER_SIM_URL \
    USER_SIM_MODEL=user_sim \
    bash "$SCRIPT_DIR/run_tau2bench_solver.sh"

    stop_user_sim

    # Find saved solver checkpoint
    SOLVER_CKPT=$(find_latest_checkpoint "$SOLVER_CKPT_DIR")
    if [ -z "$SOLVER_CKPT" ]; then
        echo "[Iter ${ITER}] ERROR: No solver checkpoint found in $SOLVER_CKPT_DIR"
        exit 1
    fi
    echo "[Iter ${ITER}] Solver checkpoint: $SOLVER_CKPT"

    # Merge solver FSDP shards → HF format for next iteration's init
    SOLVER_HF_DIR="${SOLVER_CKPT}_hf"
    echo "[Iter ${ITER}] Merging solver FSDP shards → HF model at $SOLVER_HF_DIR"
    python "$SCRIPT_DIR/../../scripts/model_merger.py" merge \
        --backend fsdp \
        --local_dir "${SOLVER_CKPT}/actor" \
        --target_dir "$SOLVER_HF_DIR"

    # ------------------------------------------------------------------
    # Update for next iteration
    # ------------------------------------------------------------------
    CHALLENGER_PREV="$CHALLENGER_HF_DIR"
    SOLVER_PREV="$SOLVER_HF_DIR"

    echo ""
    echo "[Iter ${ITER}] Done. Challenger → $CHALLENGER_CKPT | Solver → $SOLVER_CKPT"
done

# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Self-Play Complete — Running Final Evaluation"
echo "================================================================"
echo "  Final solver: $SOLVER_PREV"

EVAL_DOMAINS=${EVAL_DOMAINS:-"retail airline"}
NUM_TRIALS=${NUM_TRIALS:-5}
EVAL_PORT=${EVAL_PORT:-8001}

# Start vLLM server with final solver model for tau2 evaluation
# (tau2-bench uses litellm which needs an OpenAI-compatible API)
echo "[eval] Starting solver vLLM server on port $EVAL_PORT..."
pkill -f "vllm.entrypoints.openai.api_server.*--port ${EVAL_PORT}" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$SOLVER_PREV" \
    --served-model-name solver_eval \
    --port "$EVAL_PORT" \
    --enforce-eager \
    --tensor-parallel-size 4 > "${BASE_DIR}/eval_server.log" 2>&1 &

echo "[eval] Waiting for solver server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${EVAL_PORT}/health" > /dev/null 2>&1; then
        echo "[eval] Solver server ready!"
        break
    fi
    sleep 5
done

# Start user simulator for evaluation (strong model if available, else base)
EVAL_USER_SIM_MODEL=${EVAL_USER_SIM_MODEL:-"$USER_SIM_MODEL"}
EVAL_USER_SIM_PORT=${EVAL_USER_SIM_PORT:-8002}
echo "[eval] Starting user sim server on port $EVAL_USER_SIM_PORT..."
start_user_sim_eval() {
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${EVAL_USER_SIM_PORT}" 2>/dev/null || true
    sleep 2
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
        --model "$EVAL_USER_SIM_MODEL" \
        --served-model-name eval_user_sim \
        --port "$EVAL_USER_SIM_PORT" \
        --enforce-eager \
        --tensor-parallel-size 4 > "${BASE_DIR}/eval_user_sim.log" 2>&1 &
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${EVAL_USER_SIM_PORT}/health" > /dev/null 2>&1; then
            echo "[eval] User sim server ready!"
            return 0
        fi
        sleep 5
    done
    echo "[eval] WARNING: User sim server failed to start"
    return 1
}
start_user_sim_eval

for domain in $EVAL_DOMAINS; do
    echo "[eval] tau2-bench evaluation on domain=$domain ..."
    tau2 run \
        --domain "$domain" \
        --agent-llm "openai/solver_eval" \
        --agent-llm-args "{\"temperature\": 0.3, \"api_base\": \"http://localhost:${EVAL_PORT}/v1\"}" \
        --user-llm "openai/eval_user_sim" \
        --user-llm-args "{\"temperature\": 0.7, \"api_base\": \"http://localhost:${EVAL_USER_SIM_PORT}/v1\"}" \
        --num-trials "$NUM_TRIALS" \
        --task-split-name test \
        --save-to "${BASE_DIR}/eval_final_${domain}.json" \
        2>&1 || echo "[eval] WARNING: $domain evaluation failed (non-fatal)"
done

# Cleanup eval servers
pkill -f "vllm.entrypoints.openai.api_server.*--port ${EVAL_PORT}" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*--port ${EVAL_USER_SIM_PORT}" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  TOD-Zero Complete"
echo "  Final solver:     $SOLVER_PREV"
echo "  Eval results in:  $BASE_DIR/eval_final_*.json"
echo "================================================================"
