#!/bin/bash
# Official-style Training Script for NQ + HotpotQA on 4x A800 GPUs
# Reference: Search-R1/scripts/nq_hotpotqa/v0.2/train_ppo.sh
# Modified for: GPU 4,5,6,7 and Qwen2.5-3B-Instruct

data_name=nq_hotpotqa_train

# ==================== GPU 配置 ====================
# 指定使用 GPU 4,5,6,7（根据你的服务器配置）
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_DIR= 'data/${data_name}'

# ==================== 模型配置 ====================
# 选择你要训练的模型（取消注释一行）
export BASE_MODEL='/usr/yuque/guo/models/qwen2.5-3b-instruct'
export EXPERIMENT_NAME=${data_name}-search-r1-ppo-qwen2.5-3b-it-bm25-em

# 其他可选模型：
# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=${data_name}-search-r1-ppo-qwen2.5-3b-bm25-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=${data_name}-search-r1-ppo-qwen2.5-7b-it-bm25-em

# ==================== WandB 配置 ====================
WAND_PROJECT="Search-R1-NQ-HotpotQA"

# ==================== 环境配置 ====================
export VLLM_ATTENTION_BACKEND=XFORMERS  # Qwen2.5 推荐使用 XFORMERS

echo "============================================"
echo "NQ + HotpotQA Training (Official Config)"
echo "============================================"
echo "Data: $DATA_DIR"
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4 GPUs)"
echo "============================================"

# ==================== 开始训练 ====================
# 配置参数参考官方 v0.2，针对 4 卡进行了调整

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo ""
echo "============================================"
echo "Training Completed!"
echo "============================================"
echo "Checkpoints: verl_checkpoints/$EXPERIMENT_NAME"
echo "Training log: $EXPERIMENT_NAME.log"
echo ""
echo "Next steps:"
echo "1. View training progress:"
echo "   tail -f $EXPERIMENT_NAME.log"
echo ""
echo "2. Evaluate on TriviaQA:"
echo "   bash eval_triviaqa.sh"
echo ""
echo "3. Check WandB dashboard:"
echo "   https://wandb.ai"
echo "============================================"
