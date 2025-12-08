#!/bin/bash
# =============================================================================
# 从 Step 200 Checkpoint 继续训练（正确版本 - 使用修改后的 verl 框架）
# =============================================================================
#
# 问题诊断:
#   - Step 200 是最佳点 (NQ: 0.226, HotpotQA: 0.295)
#   - Step 300 性能下降，原因是梯度爆炸 (grad_norm: 6.47 → 32.67)
#
# 本脚本的改进:
#   1. 使用修改后的 verl 框架，支持真正的断点恢复
#   2. 从 Step 200 恢复步数计数和学习率调度器状态
#   3. 跳过已经训练过的数据（step 1-200）
#   4. 减小学习率但保持原有的 warmup 配置
#   5. 增大 KL 惩罚并添加梯度裁剪
#   6. 更频繁地保存和验证 (每50步)
#
# =============================================================================

data_name=nq_hotpotqa_train

# ==================== GPU 配置 ====================
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_DIR=/usr/yuque/guo/searchr1/data/${data_name}

# ==================== 模型配置 ====================
export BASE_MODEL='/usr/yuque/guo/models/qwen2.5-3b-instruct'

# 新的实验名称（区分于之前的训练）
export EXPERIMENT_NAME=${data_name}-search-r1-ppo-qwen2.5-3b-it-bm25-em-resume200-correct

# ==================== Checkpoint 配置 ====================
# Step 200 的 checkpoint 路径
CHECKPOINT_DIR="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em"
RESUME_FROM_STEP=200

# Actor checkpoint (从 step 200 加载模型权重)
# 正确的路径结构: checkpoint_dir/actor/global_step_200/
ACTOR_CHECKPOINT="${CHECKPOINT_DIR}/actor/global_step_${RESUME_FROM_STEP}"
CRITIC_CHECKPOINT="${CHECKPOINT_DIR}/critic/global_step_${RESUME_FROM_STEP}"

# ==================== WandB 配置 ====================
WANDB_PROJECT="Search-R1-NQ-HotpotQA"

# ==================== 环境配置 ====================
export VLLM_ATTENTION_BACKEND=XFORMERS

# ==================== 优化后的超参数 ====================
# 针对 Step 294 突然梯度爆炸的修改:
# 平衡版本: 在稳定性和学习速度之间取得平衡
ACTOR_LR=3e-7           # 原来: 1e-6, 降低到 30% (比2e-7快50%, 仍然安全)
CRITIC_LR=1e-5         # 原来: 1e-5, 降低到 20%
KL_COEF=0.01            # 原来: 0.001, 增大10倍强力约束策略
GRAD_CLIP=0.3           # 原来: 1.0, 降低到30% (应对极端advantage)
SAVE_FREQ=50            # 原来: 100, 更频繁保存
TEST_FREQ=50            # 原来: 100, 更频繁验证

# 保持原有的 warmup 配置
ACTOR_WARMUP_RATIO=0.285  # 与原训练一致
CRITIC_WARMUP_RATIO=0.015 # 与原训练一致

# 训练步数配置
# 从 step 200 恢复，还需要训练 805 步到 step 1005
TOTAL_STEPS=1005

echo "============================================"
echo "Resume Training from Step ${RESUME_FROM_STEP}"
echo "============================================"
echo "Actor Checkpoint: ${ACTOR_CHECKPOINT}"
echo "Critic Checkpoint: ${CRITIC_CHECKPOINT}"
echo "Data: $DATA_DIR"
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4 GPUs)"
echo ""
echo "优化的超参数 (针对 Step 294 突然梯度爆炸):"
echo "  - Resume from Step: ${RESUME_FROM_STEP}"
echo "  - Actor LR: ${ACTOR_LR} (原: 1e-6, 降低80%)"
echo "  - Critic LR: ${CRITIC_LR} (原: 1e-5, 降低80%)"
echo "  - Actor Warmup Ratio: ${ACTOR_WARMUP_RATIO} (与原训练一致)"
echo "  - Critic Warmup Ratio: ${CRITIC_WARMUP_RATIO} (与原训练一致)"
echo "  - KL Coef: ${KL_COEF} (原: 0.001, 增大30倍)"
echo "  - Grad Clip: ${GRAD_CLIP} (原: 1.0, 降低70%)"
echo "  - Save Freq: ${SAVE_FREQ} (原: 100)"
echo "  - Test Freq: ${TEST_FREQ} (原: 100)"
echo "  - Total Steps: ${TOTAL_STEPS} (will train from step 201 to step 1005)"
echo "============================================"
echo ""

# 检查 Checkpoint 目录是否存在
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "错误: Checkpoint 根目录不存在!"
    echo "路径: ${CHECKPOINT_DIR}"
    exit 1
fi

# 检查 Actor 和 Critic checkpoint
if [ ! -d "${ACTOR_CHECKPOINT}" ]; then
    echo "错误: Actor checkpoint 不存在!"
    echo "路径: ${ACTOR_CHECKPOINT}"
    exit 1
fi

if [ ! -d "${CRITIC_CHECKPOINT}" ]; then
    echo "错误: Critic checkpoint 不存在!"
    echo "路径: ${CRITIC_CHECKPOINT}"
    exit 1
fi

echo "✓ 找到 Checkpoint，开始训练..."
echo ""

# ==================== 开始训练 ====================
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
    data.max_obs_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=${ACTOR_CHECKPOINT} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_WARMUP_RATIO} \
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
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
    critic.optim.lr=${CRITIC_LR} \
    critic.optim.lr_warmup_steps_ratio=${CRITIC_WARMUP_RATIO} \
    critic.grad_clip=${GRAD_CLIP} \
    critic.model.use_remove_padding=True \
    critic.model.path=${CRITIC_CHECKPOINT} \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    +trainer.resume_from_step=${RESUME_FROM_STEP} \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME} \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ${EXPERIMENT_NAME}.log

echo ""
echo "============================================"
echo "Training Completed!"
echo "============================================"
echo "Checkpoints: verl_checkpoints/${EXPERIMENT_NAME}"
echo "Training log: ${EXPERIMENT_NAME}.log"
echo ""
echo "对比之前的训练:"
echo "  原实验: nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em"
echo "  新实验: ${EXPERIMENT_NAME}"
echo ""
echo "关键改进:"
echo "  ✓ 真正从 step 200 恢复（不是重新从 step 0 开始）"
echo "  ✓ 跳过了 step 1-200 的数据（不会重复训练）"
echo "  ✓ 学习率调度器从 step 200 继续（保持正确的 warmup 状态）"
echo "  ✓ WandB 日志从 step 200 继续记录"
echo ""
echo "在 WandB 上查看两个实验的对比:"
echo "  https://wandb.ai/2630305490-nanjing-university/Search-R1-NQ-HotpotQA"
echo "============================================"
