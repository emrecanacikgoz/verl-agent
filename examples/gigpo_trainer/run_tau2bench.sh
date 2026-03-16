#!/bin/bash
# ============================================================================
# tau2-bench: End-to-End Training Pipeline
# ============================================================================
# Three-stage pipeline for training tool-calling agents on tau2-bench:
#   Stage 1: Train the challenger (task generator)
#   Stage 2: Dataset construction (generate, pseudo-label, filter, order)
#   Stage 3: Train the solver agent on generated + original tasks
#
# All three models (challenger, user simulator, solver) are initialized
# from the same base LLM.
#
# Usage:
#   bash run_tau2bench.sh                                 # defaults
#   DOMAIN=airline bash run_tau2bench.sh                   # airline domain
#   bash run_tau2bench.sh --eval-only /path/to/checkpoint  # eval only
#   bash run_tau2bench.sh --skip-challenger                # skip stage 1
#
# Requirements:
#   - 4x GPUs (2 for training, 1-2 for vLLM servers)
#   - tau2-bench: pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
#   - verl-agent: pip install -e . (from repo root)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN=${DOMAIN:-"retail"}
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}       # Same base LLM for all three models
ENGINE=${ENGINE:-"vllm"}
EVAL_ONLY=false
EVAL_CHECKPOINT=""
EVAL_DOMAINS=${EVAL_DOMAINS:-"retail airline"}
NUM_TRIALS=${NUM_TRIALS:-5}
SKIP_CHALLENGER=false

# vLLM server ports
USER_SIM_PORT=${USER_SIM_PORT:-8000}
CHALLENGER_PORT=${CHALLENGER_PORT:-8001}
SOLVER_REF_PORT=${SOLVER_REF_PORT:-8002}
USER_SIM_GPU=${USER_SIM_GPU:-3}

# Dataset generation config
NUM_GENERATE=${NUM_GENERATE:-500}
NUM_TARGET=${NUM_TARGET:-200}
NUM_SOLVER_ATTEMPTS=${NUM_SOLVER_ATTEMPTS:-4}
NOISE_THRESHOLD=${NOISE_THRESHOLD:-0.25}

export USER_SIM_URL="http://localhost:${USER_SIM_PORT}/v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)        EVAL_ONLY=true; EVAL_CHECKPOINT="$2"; shift 2 ;;
        --domain)           DOMAIN="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --skip-challenger)  SKIP_CHALLENGER=true; shift ;;
        --help|-h)
            echo "Usage: bash run_tau2bench.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --domain DOMAIN         tau2-bench domain (default: retail)"
            echo "  --model MODEL           Base model for all 3 roles (default: Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --eval-only CHECKPOINT  Only run evaluation on given checkpoint"
            echo "  --skip-challenger       Skip stage 1 (challenger training)"
            echo ""
            echo "Environment variables:"
            echo "  DOMAIN, MODEL, ENGINE, NUM_GENERATE, NUM_TARGET"
            echo "  HF_HOME, WANDB_API_KEY, WANDB_DIR, CUDA_VISIBLE_DEVICES"
            exit 0
            ;;
        *) break ;;
    esac
done

echo "================================================================"
echo "  tau2-bench End-to-End Training Pipeline"
echo "================================================================"
echo "  Domain:              $DOMAIN"
echo "  Base model (all 3):  $MODEL"
echo "  Engine:              $ENGINE"
echo "  Skip challenger:     $SKIP_CHALLENGER"
echo "  Dataset: generate=$NUM_GENERATE target=$NUM_TARGET"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Utility: start a vLLM server
# ─────────────────────────────────────────────────────────────────────────────
start_vllm_server() {
    local model_path="$1"
    local served_name="$2"
    local port="$3"
    local gpu="$4"
    local log_file="${served_name}_server.log"

    echo "[server] Starting $served_name on GPU $gpu, port $port..."

    # Kill existing server on this port
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
    sleep 2

    CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$served_name" \
        --port "$port" \
        --enforce-eager \
        --tensor-parallel-size 1 > "$log_file" 2>&1 &

    echo "[server] $served_name PID: $!"
    echo "[server] Waiting for $served_name to initialize..."

    for i in $(seq 1 60); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[server] $served_name is ready!"
            return 0
        fi
        sleep 5
    done

    echo "[server] ERROR: $served_name failed to start. Check $log_file"
    exit 1
}

stop_vllm_server() {
    local port="$1"
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
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
# Stage 1: Train the Challenger
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_CHALLENGER" = false ]; then
    echo ""
    echo "================================================================"
    echo "  Stage 1: Training Challenger (task generator)"
    echo "================================================================"

    DOMAIN=$DOMAIN \
    MODEL=$MODEL \
    bash "${SCRIPT_DIR}/run_tau2bench_challenger.sh" "$ENGINE"

    CHALLENGER_CKPT=$(ls -td checkpoints/verl_agent_tau2bench_challenger_*/global_step_* 2>/dev/null | head -1)
    if [ -z "$CHALLENGER_CKPT" ]; then
        echo "[stage 1] WARNING: No challenger checkpoint found, using base model for generation"
        CHALLENGER_CKPT="$MODEL"
    else
        echo "[stage 1] Challenger checkpoint: $CHALLENGER_CKPT"
    fi
else
    echo ""
    echo "[stage 1] Skipped challenger training (--skip-challenger)"
    CHALLENGER_CKPT="$MODEL"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Dataset Construction
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Stage 2: Dataset Construction"
echo "  Generate ${NUM_GENERATE} samples, select ${NUM_TARGET} after filtering"
echo "================================================================"

# Start challenger vLLM server for generation
start_vllm_server "$CHALLENGER_CKPT" "challenger" "$CHALLENGER_PORT" "$USER_SIM_GPU"

# Start reference solver server for pseudo-labeling (base model)
start_vllm_server "$MODEL" "solver" "$SOLVER_REF_PORT" "$USER_SIM_GPU"

DATASET_DIR="$HOME/data/verl-agent/tau2bench_generated"

python3 -m examples.gigpo_trainer.generate_tau2bench_dataset \
    --challenger_url "http://localhost:${CHALLENGER_PORT}/v1" \
    --challenger_model "challenger" \
    --solver_url "http://localhost:${SOLVER_REF_PORT}/v1" \
    --solver_model "solver" \
    --domain "$DOMAIN" \
    --num_generate "$NUM_GENERATE" \
    --num_target "$NUM_TARGET" \
    --num_solver_attempts "$NUM_SOLVER_ATTEMPTS" \
    --noise_threshold "$NOISE_THRESHOLD" \
    --order_easy_to_hard \
    --output_dir "$DATASET_DIR"

# Stop generation servers
stop_vllm_server "$CHALLENGER_PORT"
stop_vllm_server "$SOLVER_REF_PORT"

echo "[stage 2] Dataset generated at: $DATASET_DIR"

# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: Train the Solver
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Stage 3: Training Solver Agent"
echo "================================================================"

# Start user simulator server (same base model)
start_vllm_server "$MODEL" "user_sim" "$USER_SIM_PORT" "$USER_SIM_GPU"

DOMAIN=$DOMAIN \
USER_SIM_URL=$USER_SIM_URL \
USER_SIM_MODEL="user_sim" \
MODEL=$MODEL \
bash "${SCRIPT_DIR}/run_tau2bench_solver.sh" "$ENGINE"

# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: Evaluate
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Stage 4: Post-training Evaluation"
echo "================================================================"

LATEST_CKPT=$(ls -td checkpoints/verl_agent_tau2bench_*/global_step_* 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    run_tau2_eval "$LATEST_CKPT" "post_train"
else
    echo "[stage 4] No checkpoint found, skipping evaluation."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[cleanup] Stopping all vLLM servers..."
stop_vllm_server "$USER_SIM_PORT"
stop_vllm_server "$CHALLENGER_PORT"
stop_vllm_server "$SOLVER_REF_PORT"

echo ""
echo "================================================================"
echo "  Done! Check W&B for training curves."
echo "  Pipeline: Challenger -> Dataset (${NUM_TARGET} samples) -> Solver"
echo "================================================================"
