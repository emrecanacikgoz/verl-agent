#!/bin/bash
# ============================================================================
# tau2-bench: Full Training Pipeline
# ============================================================================
# End-to-end script for training tool-calling agents on tau2-bench.
# Handles: user simulator server startup, solver training, and evaluation.
#
# Usage:
#   bash run_tau2bench.sh                                 # defaults
#   DOMAIN=airline bash run_tau2bench.sh                   # airline domain
#   bash run_tau2bench.sh --eval-only /path/to/checkpoint  # eval only
#
# Requirements:
#   - 8x GPUs (6 for training, 1 for user sim vLLM server)
#   - tau2-bench: pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
#   - verl-agent: pip install -e . (from repo root)
# ============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN=${DOMAIN:-"airline"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
USER_SIM_MODEL=${USER_SIM_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
USER_SIM_PORT=${USER_SIM_PORT:-8000}
USER_SIM_GPU=${USER_SIM_GPU:-7}
ENGINE=${ENGINE:-"vllm"}
EVAL_ONLY=false
EVAL_CHECKPOINT=""
EVAL_DOMAINS=${EVAL_DOMAINS:-"retail airline"}
NUM_TRIALS=${NUM_TRIALS:-5}

export USER_SIM_URL="http://localhost:${USER_SIM_PORT}/v1"

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
            echo "  --model MODEL           Base model (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --eval-only CHECKPOINT  Only run evaluation on given checkpoint"
            echo ""
            echo "Environment variables:"
            echo "  DOMAIN, MODEL, USER_SIM_MODEL, USER_SIM_PORT, USER_SIM_GPU"
            echo "  ENGINE, EVAL_DOMAINS, NUM_TRIALS"
            echo "  HF_HOME, WANDB_API_KEY, WANDB_DIR, CUDA_VISIBLE_DEVICES"
            ;;
        *) break ;;
    esac
done

echo "================================================================"
echo "  tau2-bench Training Pipeline"
echo "================================================================"
echo "  Domain:          $DOMAIN"
echo "  Agent model:     $MODEL"
echo "  User sim model:  $USER_SIM_MODEL"
echo "  Engine:          $ENGINE"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Start user simulator vLLM server
# ─────────────────────────────────────────────────────────────────────────────
start_user_sim_server() {
    echo "[step 1] Starting user simulator server on GPU $USER_SIM_GPU, port $USER_SIM_PORT..."

    # Kill existing server if running
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${USER_SIM_PORT}" 2>/dev/null || true
    sleep 2

    CUDA_VISIBLE_DEVICES=$USER_SIM_GPU python -m vllm.entrypoints.openai.api_server \
        --model "$USER_SIM_MODEL" \
        --served-model-name user_sim \
        --port "$USER_SIM_PORT" \
        --enforce-eager \
        --tensor-parallel-size 1 > user_sim_server.log 2>&1 &

    USER_SIM_PID=$!
    echo "[step 1] User sim server PID: $USER_SIM_PID"
    echo "[step 1] Waiting for server to initialize..."

    # Wait for server to be ready
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${USER_SIM_PORT}/health" > /dev/null 2>&1; then
            echo "[step 1] User sim server is ready!"
            return 0
        fi
        sleep 5
    done

    echo "[step 1] ERROR: User sim server failed to start. Check user_sim_server.log"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run evaluation with tau2-bench CLI
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
fi

if [ "$EVAL_ONLY" = false ]; then

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Start user sim
# ─────────────────────────────────────────────────────────────────────────────
start_user_sim_server

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train solver agent
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[step 2] Training solver agent on domain=$DOMAIN ..."

DOMAIN=$DOMAIN \
USER_SIM_URL=$USER_SIM_URL \
USER_SIM_MODEL=user_sim \
MODEL=$MODEL \
bash "$(dirname "$0")/run_tau2bench_solver.sh" "$ENGINE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[step 3] Post-training evaluation..."

# Find latest checkpoint
LATEST_CKPT=$(ls -td checkpoints/verl_agent_tau2bench_*/global_step_* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    run_tau2_eval "$LATEST_CKPT" "post_train"
else
    echo "[step 3] No checkpoint found, skipping evaluation."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[cleanup] Stopping user simulator server..."
pkill -f "vllm.entrypoints.openai.api_server.*--port ${USER_SIM_PORT}" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  Done! Check W&B for training curves."
echo "================================================================"

fi # EVAL_ONLY=false
