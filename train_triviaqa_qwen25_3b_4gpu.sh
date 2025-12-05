#!/bin/bash
# TriviaQA PPO Training Script for 4x A800 GPUs (GPU 4,5,6,7)
# Model: Qwen2.5-3B-Instruct
# Retriever: BM25 (CPU-based)

echo "============================================"
echo "TriviaQA Training with Qwen2.5-3B-Instruct"
echo "4x A800 GPUs (GPU 4,5,6,7)"
echo "============================================"

# ==================== 配置部分 ====================
# 修改这些路径为你的实际路径
DATA_DIR=/path/to/Search-R1/data/triviaqa_search
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
EXPERIMENT_NAME=triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu

# GPU 配置：指定使用 GPU 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Qwen 优化配置
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

# ==================== 训练参数 ====================
# 4卡训练配置（每卡约 18-20GB 显存）
TRAIN_BATCH_SIZE=1024        # 总 batch size
VAL_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=512      # PPO mini batch
PPO_MICRO_BATCH_SIZE=64      # 每卡的 micro batch

# 模型参数
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=500
MAX_START_LENGTH=2048
MAX_OBS_LENGTH=500

# 训练超参数
ACTOR_LR=1e-6
CRITIC_LR=1e-5
KL_COEF=0.001
TOTAL_EPOCHS=10
TOTAL_STEPS=500
SAVE_FREQ=100

# 检索器配置
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"
RETRIEVER_TOPK=3
MAX_TURNS=4

# GPU 配置
N_GPUS_PER_NODE=4
GPU_MEMORY_UTIL=0.7          # A800 80GB 可以设置高一些

echo "Data directory: $DATA_DIR"
echo "Base model: $BASE_MODEL"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $N_GPUS_PER_NODE"
echo "============================================"

# ==================== 开始训练 ====================
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    critic.optim.lr=$CRITIC_LR \
    critic.ppo_micro_batch_size=8 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.adv_estimator=gae \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    max_turns=$MAX_TURNS \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=$RETRIEVER_TOPK \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=./checkpoints/$EXPERIMENT_NAME \
    trainer.project_name=search-r1-triviaqa \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console','tracking','wandb']" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1

echo "============================================"
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints/$EXPERIMENT_NAME"
echo "============================================"