#!/bin/bash
# ============================================================================
# tau2-bench: Full Training Pipeline
# ============================================================================
# End-to-end script for training tool-calling agents on tau2-bench.
#
# Pipeline (3 models initialized from same base LLM):
#   1. Train challenger (task generator) with format + validity rewards
#   2. Construct dataset: generate with challenger, validate pseudo labels,
#      optionally order easy-to-hard, drop noisy samples, select target N
#   3. Train solver (tool-calling agent) with format + tool-call + task success rewards
#   4. Evaluate
#
# The three models (all initialized from MODEL):
#   (i)   Challenger: learns to generate realistic task specs (trained)
#   (ii)  User simulator: simulates customer behavior (fixed, separate LLM)
#   (iii) Solver: learns to solve tool-calling tasks (trained)
#
# Usage:
#   bash run_tau2bench.sh                                 # defaults
#   DOMAIN=airline bash run_tau2bench.sh                   # airline domain
#   bash run_tau2bench.sh --eval-only /path/to/checkpoint  # eval only
#
# Requirements:
#   - 4x GPUs (2 for training, 1-2 for user sim / challenger vLLM servers)
#   - tau2-bench: pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
#   - verl-agent: pip install -e . (from repo root)
# ============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN=${DOMAIN:-"retail"}
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}             # Base LLM for all 3 models
USER_SIM_MODEL=${USER_SIM_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
USER_SIM_PORT=${USER_SIM_PORT:-8000}
USER_SIM_GPU=${USER_SIM_GPU:-3}
CHALLENGER_PORT=${CHALLENGER_PORT:-8001}
CHALLENGER_GPU=${CHALLENGER_GPU:-2}
ENGINE=${ENGINE:-"vllm"}
EVAL_ONLY=false
EVAL_CHECKPOINT=""
EVAL_DOMAINS=${EVAL_DOMAINS:-"retail airline"}
NUM_TRIALS=${NUM_TRIALS:-5}

# Dataset construction config
NUM_GENERATE=${NUM_GENERATE:-500}
NUM_TARGET=${NUM_TARGET:-200}
ORDER_EASY_TO_HARD=${ORDER_EASY_TO_HARD:-true}
NOISE_THRESHOLD=${NOISE_THRESHOLD:-0.4}

export USER_SIM_URL="http://localhost:${USER_SIM_PORT}/v1"
export CHALLENGER_URL="http://localhost:${CHALLENGER_PORT}/v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)    EVAL_ONLY=true; EVAL_CHECKPOINT="$2"; shift 2 ;;
        --domain)       DOMAIN="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash run_tau2bench.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN         tau2-bench domain (default: retail)"
            echo "  --model MODEL           Base model for all 3 roles (default: Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --eval-only CHECKPOINT  Only run evaluation on given checkpoint"
            echo ""
            echo "Environment variables:"
            echo "  DOMAIN, MODEL, USER_SIM_MODEL, USER_SIM_PORT, USER_SIM_GPU"
            echo "  CHALLENGER_PORT, CHALLENGER_GPU, ENGINE"
            echo "  NUM_GENERATE, NUM_TARGET, ORDER_EASY_TO_HARD, NOISE_THRESHOLD"
            echo "  EVAL_DOMAINS, NUM_TRIALS"
            echo "  HF_HOME, WANDB_API_KEY, WANDB_DIR, CUDA_VISIBLE_DEVICES"
            exit 0
            ;;
        *) break ;;
    esac
done

echo "================================================================"
echo "  tau2-bench Full Training Pipeline"
echo "================================================================"
echo "  Domain:           $DOMAIN"
echo "  Base model:       $MODEL  (challenger + solver initialized from this)"
echo "  User sim model:   $USER_SIM_MODEL"
echo "  Engine:           $ENGINE"
echo "  Dataset config:   generate=$NUM_GENERATE target=$NUM_TARGET easy2hard=$ORDER_EASY_TO_HARD"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: start vLLM server for a given model
# ─────────────────────────────────────────────────────────────────────────────
start_vllm_server() {
    local model="$1"
    local served_name="$2"
    local port="$3"
    local gpu="$4"
    local log_file="$5"

    echo "[server] Starting $served_name server on GPU $gpu, port $port..."

    # Kill existing server if running
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
    sleep 2

    CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$served_name" \
        --port "$port" \
        --enforce-eager \
        --tensor-parallel-size 1 > "$log_file" 2>&1 &

    local pid=$!
    echo "[server] $served_name PID: $pid"

    # Wait for server to be ready
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[server] $served_name server is ready!"
            return 0
        fi
        sleep 5
    done

    echo "[server] ERROR: $served_name server failed to start. Check $log_file"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run evaluation with tau2-bench CLI
# ─────────────────────────────────────────────────────────────────────────────
run_tau2_eval() {
    local model_path="$1"
    local tag="$2"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Evaluation: $tag"
    echo "  Model: $model_path"
    echo "  Domains: $EVAL_DOMAINS"
    echo "────────────────────────────────────────────────────────────────"

    for domain in $EVAL_DOMAINS; do
        echo "[eval] Running tau2 eval on domain=$domain ..."
        tau2 run \
            --domain "$domain" \
            --agent-llm "$model_path" \
            --num-trials "$NUM_TRIALS" \
            --task-split-name test \
            --save-to "eval_${tag}_${domain}.json" \
            2>&1 || echo "[eval] WARNING: $domain evaluation failed (non-fatal)"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Eval-only mode
# ─────────────────────────────────────────────────────────────────────────────
if [ "$EVAL_ONLY" = true ]; then
    echo "[eval-only] Evaluating checkpoint: $EVAL_CHECKPOINT"
    run_tau2_eval "$EVAL_CHECKPOINT" "eval_only"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Train Challenger
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  STEP 1: Training Challenger (task generator)"
echo "  Rewards: format (0.5) + validity (0.5)"
echo "================================================================"

DOMAIN=$DOMAIN \
MODEL=$MODEL \
bash "$(dirname "$0")/run_tau2bench_challenger.sh" "$ENGINE"

# Find latest challenger checkpoint
CHALLENGER_CKPT=$(ls -td checkpoints/verl_agent_tau2bench_challenger_*/global_step_* 2>/dev/null | head -1)
if [ -z "$CHALLENGER_CKPT" ]; then
    echo "[step 1] ERROR: No challenger checkpoint found after training."
    exit 1
fi
echo "[step 1] Challenger checkpoint: $CHALLENGER_CKPT"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Dataset Construction
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  STEP 2: Dataset Construction (Tool-R0 style)"
echo "  Generate ${NUM_GENERATE} → validate → filter → select ${NUM_TARGET}"
echo "================================================================"

# Start challenger vLLM server for generation
start_vllm_server "$CHALLENGER_CKPT" "challenger" "$CHALLENGER_PORT" "$CHALLENGER_GPU" "challenger_server.log"

# Build ordering flag
EASY_TO_HARD_FLAG=""
if [ "$ORDER_EASY_TO_HARD" = true ]; then
    EASY_TO_HARD_FLAG="--order_easy_to_hard"
fi

SOLVER_DATA_DIR="$HOME/data/verl-agent/tau2bench_solver/${DOMAIN}"

python3 -m examples.data_preprocess.construct_tau2bench_dataset \
    --challenger_model "$CHALLENGER_CKPT" \
    --challenger_url "$CHALLENGER_URL" \
    --domain "$DOMAIN" \
    --num_generate "$NUM_GENERATE" \
    --num_target "$NUM_TARGET" \
    --noise_threshold "$NOISE_THRESHOLD" \
    --output_dir "$SOLVER_DATA_DIR" \
    $EASY_TO_HARD_FLAG

# Stop challenger server
pkill -f "vllm.entrypoints.openai.api_server.*--port ${CHALLENGER_PORT}" 2>/dev/null || true

# Verify dataset files exist
if [ ! -f "$SOLVER_DATA_DIR/train.parquet" ]; then
    echo "[step 2] ERROR: Dataset construction failed (no train.parquet)."
    exit 1
fi
echo "[step 2] Dataset ready at: $SOLVER_DATA_DIR"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Start User Simulator
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  STEP 3: Starting User Simulator"
echo "================================================================"

start_vllm_server "$USER_SIM_MODEL" "user_sim" "$USER_SIM_PORT" "$USER_SIM_GPU" "user_sim_server.log"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Train Solver
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  STEP 4: Training Solver (tool-calling agent)"
echo "  Rewards: format (0.1) + tool-call accuracy (0.4) + task success (0.5)"
echo "================================================================"

DOMAIN=$DOMAIN \
USER_SIM_URL=$USER_SIM_URL \
USER_SIM_MODEL=$USER_SIM_MODEL \
MODEL=$MODEL \
bash "$(dirname "$0")/run_tau2bench_solver.sh" "$ENGINE"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  STEP 5: Post-training Evaluation"
echo "================================================================"

# Find latest solver checkpoint
LATEST_CKPT=$(ls -td checkpoints/verl_agent_tau2bench_*/global_step_* 2>/dev/null | grep -v challenger | head -1)
if [ -n "$LATEST_CKPT" ]; then
    run_tau2_eval "$LATEST_CKPT" "post_train"
else
    echo "[step 5] No solver checkpoint found, skipping evaluation."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[cleanup] Stopping servers..."
pkill -f "vllm.entrypoints.openai.api_server.*--port ${USER_SIM_PORT}" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*--port ${CHALLENGER_PORT}" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  Done! Pipeline complete."
echo "  1. Challenger trained at: $CHALLENGER_CKPT"
echo "  2. Dataset at: $SOLVER_DATA_DIR"
echo "  3. Solver checkpoint: ${LATEST_CKPT:-'N/A'}"
echo "  Check W&B for training curves."
echo "================================================================"
