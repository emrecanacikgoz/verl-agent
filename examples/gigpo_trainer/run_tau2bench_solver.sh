#!/bin/bash
# ============================================================================
# tau2-bench Solver Training with GiGPO
# ============================================================================
# Trains a tool-calling customer service agent on tau2-bench domains using
# GiGPO (Group-in-Group Policy Optimization) with verl-agent.
#
# Usage:
#   bash run_tau2bench_solver.sh                         # defaults (vLLM)
#   bash run_tau2bench_solver.sh sglang                  # use SGLang
#   DOMAIN=airline bash run_tau2bench_solver.sh           # airline domain
#
# Prerequisites:
#   1. Install tau2-bench: pip install git+https://github.com/emrecanacikgoz/tau2-bench.git
#   2. Start user simulator vLLM server (see below)
#
# Start user simulator server:
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --port 8000 --tensor-parallel-size 1
# ============================================================================

set -x
ENGINE=${1:-vllm}
shift $# # consume positional args so $@ doesn't leak into hydra
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME=${HF_HOME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DIR=${WANDB_DIR:-}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN=${DOMAIN:-"retail"}                              # retail, airline, telecom
USER_SIM_URL=${USER_SIM_URL:-"http://localhost:8000/v1"}
USER_SIM_MODEL=${USER_SIM_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=32
group_size=8
history_length=4
max_steps=30
mode="mean_std_norm"

# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=tau2bench_solver \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.tau2bench.domain=$DOMAIN \
    env.tau2bench.user_sim_url=$USER_SIM_URL \
    env.tau2bench.user_sim_model=$USER_SIM_MODEL \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="verl_agent_tau2bench_${DOMAIN}" \
    trainer.experiment_name="gigpo_qwen2.5_1.5b_${DOMAIN}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True $@
