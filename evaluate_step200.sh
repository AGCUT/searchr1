#!/bin/bash
# 评估 Step 200 checkpoint

data_name=nq_hotpotqa_train

export CUDA_VISIBLE_DEVICES=1,2,6,7
export DATA_DIR=data/${data_name}
export BASE_MODEL=verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "============================================"
echo "评估 Step 200 Checkpoint"
echo "============================================"
echo "Model: $BASE_MODEL"
echo "Data: $DATA_DIR/test_5k.parquet"
echo "Batch Size: 256"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4 GPUs)"
echo "============================================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.val_files=$DATA_DIR/test_5k.parquet \
    data.val_data_num=null \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.logger=['wandb'] \
    +trainer.val_only=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.project_name=Search-R1-Evaluation \
    trainer.experiment_name=step200-eval \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee eval_step200.log

echo ""
echo "============================================"
echo "评估完成！"
echo "日志: eval_step200.log"
echo "结果已上传到 WandB"
echo "============================================"
