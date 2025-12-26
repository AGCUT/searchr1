#!/bin/bash
# Stage 2 GRPO Training: Efficiency Optimization
# 在第一阶段 PPO 训练完成后，使用 GRPO 优化检索效率
# 目标：在保持答案正确率的前提下，减少不必要的检索次数

data_name=nq_hotpotqa_train

# ==================== GPU 配置 ====================
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_DIR=/usr/yuque/guo/searchr1/data/${data_name}

# ==================== 模型配置 ====================
# 使用第一阶段训练好的模型作为起点
export STAGE1_CHECKPOINT=/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em-resume200/actor/global_step_50

export EXPERIMENT_NAME=${data_name}-search-r1-grpo-stage2-efficiency

# ==================== WandB 配置 ====================
WAND_PROJECT="Search-R1-NQ-HotpotQA-Stage2"

# ==================== 环境配置 ====================
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "============================================"
echo "Stage 2 GRPO Training: Efficiency Optimization"
echo "============================================"
echo "Data: $DATA_DIR"
echo "Stage 1 Model: $STAGE1_CHECKPOINT"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4 GPUs)"
echo "============================================"
echo "目标: 减少检索次数，提高推理效率"
echo "============================================"

# ==================== 开始第二阶段训练 ====================
# 关键变化:
# 1. algorithm.adv_estimator=grpo (无需 Critic)
# 2. 使用 main_grpo_stage2 入口（效率奖励）
# 3. n_agent=5 (GRPO 需要多个采样进行组内对比)
# 4. 移除所有 critic 相关配置

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_grpo_stage2 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path=$STAGE1_CHECKPOINT \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ${EXPERIMENT_NAME}.log

echo ""
echo "============================================"
echo "Stage 2 Training Completed!"
echo "============================================"
echo "Checkpoints: verl_checkpoints/$EXPERIMENT_NAME"
echo "============================================"
