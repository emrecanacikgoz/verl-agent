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

ENGINE=${1:-vllm}
shift $# # consume positional args so $@ doesn't leak into hydra
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
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
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
# TOD-Zero self-play: path to challenger-generated scenarios JSON
# null = standard mode (use tau2-bench registry tasks)
CHALLENGER_SCENARIOS_PATH=${CHALLENGER_SCENARIOS_PATH:-null}
# Self-play: override training duration and checkpoint directory
SOLVER_EPOCHS=${SOLVER_EPOCHS:-100}
SOLVER_CKPT_DIR=${SOLVER_CKPT_DIR:-"checkpoints/verl_agent_tau2bench_${DOMAIN}"}

num_cpus_per_env_worker=0.1
train_data_size=28
val_data_size=28
group_size=4
history_length=4
max_steps=30
# Reward combination: weighted sum of tool-call accuracy + tau2-bench task success
# Set both to 0.5 for equal weighting; set task_success to 0 to disable
TOOL_CALL_REWARD_COEF=${TOOL_CALL_REWARD_COEF:-0.5}
TASK_SUCCESS_REWARD_COEF=${TASK_SUCCESS_REWARD_COEF:-0.5}

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
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=28 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=tau2bench_solver \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.history_length=$history_length \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.tau2bench.domain=$DOMAIN \
    env.tau2bench.user_sim_url=$USER_SIM_URL \
    env.tau2bench.user_sim_model=$USER_SIM_MODEL \
    env.tau2bench.tool_call_reward_coef=$TOOL_CALL_REWARD_COEF \
    env.tau2bench.task_success_reward_coef=$TASK_SUCCESS_REWARD_COEF \
    env.tau2bench.challenger_scenarios_path=$CHALLENGER_SCENARIOS_PATH \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="verl_agent_tau2bench_${DOMAIN}" \
    trainer.experiment_name="grpo_qwen2.5_7b_${DOMAIN}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-7} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$SOLVER_EPOCHS \
    trainer.default_local_dir=$SOLVER_CKPT_DIR \
    trainer.val_before_train=True $@
