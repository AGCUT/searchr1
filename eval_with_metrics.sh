#!/bin/bash
# 评估脚本 - 支持记录 search 次数和回答时间
#
# 功能：
# 1. 记录每个问题的 search 次数
# 2. 记录每个问题的回答时间
# 3. 按数据源分别统计（NQ / HotpotQA / TriviaQA）
# 4. 输出详细的统计报告
#
# 使用方法：
#   bash eval_with_metrics.sh

set -e

# ==================== 配置 ====================
# 模型路径
MODEL_PATH="/usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200"

# 测试数据路径
TEST_DATA="/usr/yuque/guo/searchr1/data/nq_hotpotqa_train/test.parquet"

# 输出路径
OUTPUT_DIR="/usr/yuque/guo/searchr1/eval_results"
OUTPUT_FILE="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).json"

# 检索服务
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# 评估参数
MAX_SAMPLES=500     # 评估样本数，设为 null 评估全部
MAX_TURNS=4         # 最大检索轮数
TEMPERATURE=0.7     # 生成温度（0 表示贪婪解码）

# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# ==================== 检查 ====================
echo "============================================"
echo "评估脚本 - 支持记录 search 次数和回答时间"
echo "============================================"
echo "模型: $MODEL_PATH"
echo "数据: $TEST_DATA"
echo "检索服务: $RETRIEVER_URL"
echo "最大样本数: $MAX_SAMPLES"
echo "输出: $OUTPUT_FILE"
echo "============================================"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型不存在: $MODEL_PATH"
    exit 1
fi

# 检查数据是否存在
if [ ! -f "$TEST_DATA" ]; then
    echo "错误: 数据文件不存在: $TEST_DATA"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查检索服务
echo ""
echo "检查检索服务..."
if curl -s --connect-timeout 5 "$RETRIEVER_URL" > /dev/null 2>&1; then
    echo "✓ 检索服务正常"
else
    echo "警告: 检索服务可能未启动"
    echo "请先在另一个终端运行: bash retrieval_launch_bm25.sh"
    read -p "确认已启动检索服务? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# ==================== 开始评估 ====================
echo ""
echo "============================================"
echo "开始评估..."
echo "============================================"

python eval_with_metrics.py \
    --model "$MODEL_PATH" \
    --data "$TEST_DATA" \
    --output "$OUTPUT_FILE" \
    --retriever_url "$RETRIEVER_URL" \
    --max_samples $MAX_SAMPLES \
    --max_turns $MAX_TURNS \
    --temperature $TEMPERATURE

echo ""
echo "============================================"
echo "评估完成!"
echo "============================================"
echo "结果文件: $OUTPUT_FILE"
echo "CSV 文件: ${OUTPUT_FILE%.json}.csv"
echo ""
echo "查看统计:"
echo "  cat ${OUTPUT_FILE} | python -c \"import json,sys; d=json.load(sys.stdin); print('准确率:', d['stats']['accuracy']*100, '%'); print('平均search:', d['stats']['avg_searches']); print('平均时间:', d['stats']['avg_time'], '秒')\""
echo ""
echo "查看详细结果:"
echo "  head -20 ${OUTPUT_FILE%.json}.csv"
echo "============================================"