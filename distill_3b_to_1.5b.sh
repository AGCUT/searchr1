#!/bin/bash
# 3B → 1.5B 蒸馏脚本
#
# 前置条件：
# 1. 已经训练好3B模型 (verl_checkpoints/xxx/actor/step_200)
# 2. 检索服务正在运行 (bash retrieval_launch_bm25.sh)
#
# 使用方法：
#   bash distill_3b_to_1.5b.sh

set -e

# ==================== 配置 ====================
# 教师模型：你训练好的3B checkpoint
TEACHER_MODEL="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"

# 学生模型：1.5B base model
STUDENT_MODEL="/usr/yuque/guo/models/qwen2.5-1.5b-instruct"

# 数据路径
TRAIN_DATA="/usr/yuque/guo/searchr1/data/nq_hotpotqa_train/train.parquet"

# 输出路径
DISTILL_DATA="/usr/yuque/guo/searchr1/distill_data.jsonl"
OUTPUT_MODEL="/usr/yuque/guo/searchr1/checkpoints/qwen2.5-1.5b-distilled"

# 检索服务
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# 蒸馏参数
NUM_SAMPLES=10000   # 生成多少个样本
NUM_RETURN=4        # 每个问题生成几个响应（选最好的）
MAX_TURNS=4         # 最大检索轮数

# 训练参数
EPOCHS=3
LR=1e-5
BATCH_SIZE=4

# GPU配置
export CUDA_VISIBLE_DEVICES=0  # 蒸馏数据生成用单卡就够

# ==================== 检查 ====================
echo "============================================"
echo "3B → 1.5B 模型蒸馏"
echo "============================================"
echo "教师模型: $TEACHER_MODEL"
echo "学生模型: $STUDENT_MODEL"
echo "训练数据: $TRAIN_DATA"
echo "检索服务: $RETRIEVER_URL"
echo "============================================"

# 检查教师模型是否存在
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "错误: 教师模型不存在: $TEACHER_MODEL"
    echo "请先完成3B模型的PPO训练"
    exit 1
fi

# 检查检索服务是否运行
echo "检查检索服务..."
if curl -s -o /dev/null -w "%{http_code}" "$RETRIEVER_URL" | grep -q "000\|404\|503"; then
    echo "警告: 检索服务可能未启动"
    echo "请在另一个终端运行: bash retrieval_launch_bm25.sh"
    read -p "确认已启动检索服务? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# ==================== 步骤1: 生成蒸馏数据 ====================
echo ""
echo "============================================"
echo "步骤1: 生成蒸馏数据"
echo "============================================"

if [ -f "$DISTILL_DATA" ]; then
    echo "蒸馏数据已存在: $DISTILL_DATA"
    read -p "是否重新生成? [y/N] " regenerate
    if [ "$regenerate" != "y" ] && [ "$regenerate" != "Y" ]; then
        echo "跳过数据生成，使用现有数据"
    else
        rm -f "$DISTILL_DATA"
    fi
fi

if [ ! -f "$DISTILL_DATA" ]; then
    echo "开始生成蒸馏数据..."
    echo "这可能需要几个小时，取决于样本数量和模型速度"

    python distill_3b_to_1.5b.py generate \
        --teacher_model "$TEACHER_MODEL" \
        --data "$TRAIN_DATA" \
        --output "$DISTILL_DATA" \
        --retriever_url "$RETRIEVER_URL" \
        --num_samples $NUM_SAMPLES \
        --num_return $NUM_RETURN \
        --max_turns $MAX_TURNS

    echo "✓ 蒸馏数据生成完成: $DISTILL_DATA"
fi

# 统计数据
echo ""
echo "蒸馏数据统计:"
wc -l "$DISTILL_DATA"
head -1 "$DISTILL_DATA" | python -m json.tool | head -20

# ==================== 步骤2: 训练1.5B学生模型 ====================
echo ""
echo "============================================"
echo "步骤2: 训练1.5B学生模型"
echo "============================================"

# 多GPU训练
export CUDA_VISIBLE_DEVICES=0,1,2,3

python distill_3b_to_1.5b.py train \
    --student_model "$STUDENT_MODEL" \
    --data "$DISTILL_DATA" \
    --output "$OUTPUT_MODEL" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE

echo ""
echo "============================================"
echo "✓ 蒸馏完成!"
echo "============================================"
echo "蒸馏数据: $DISTILL_DATA"
echo "1.5B模型: $OUTPUT_MODEL"
echo ""
echo "下一步:"
echo "1. 在测试集上评估1.5B模型"
echo "2. 如果效果不够好，可以继续用PPO微调"
echo ""
echo "评估命令:"
echo "  python infer.py --model $OUTPUT_MODEL --data test.jsonl"
echo "============================================"
