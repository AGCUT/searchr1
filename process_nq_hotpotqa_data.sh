#!/bin/bash
# Official-style Data Processing Script for NQ + HotpotQA
# Reference: Search-R1/scripts/nq_hotpotqa/data_process.sh
# This script processes training and test data following the official approach

WORK_DIR=$(pwd)
LOCAL_DIR=/usr/yuque/guo/searchr1/data/nq_hotpotqa_train

echo "============================================"
echo "Processing NQ + HotpotQA Data (Official Method)"
echo "============================================"
echo "Working directory: $WORK_DIR"
echo "Output directory: $LOCAL_DIR"
echo "============================================"

# 创建输出目录
mkdir -p $LOCAL_DIR

# ==================== Step 1: Process Training Data ====================
echo ""
echo "Step 1: Processing training data (NQ + HotpotQA)..."
echo "This will download from HuggingFace: RUC-NLPIR/FlashRAG_datasets"
echo ""

DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA \
    --template_type base

if [ $? -eq 0 ]; then
    echo "✓ Training data processed successfully!"
    echo "  Output: $LOCAL_DIR/train.parquet"
else
    echo "✗ Error processing training data"
    exit 1
fi

# ==================== Step 2: Process Test Data ====================
echo ""
echo "Step 2: Processing test data (7 datasets for evaluation)..."
echo "Datasets: NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA, Musique, Bamboogle"
echo ""

DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA \
    --template_type base

if [ $? -eq 0 ]; then
    echo "✓ Test data processed successfully!"
    echo "  Output: $LOCAL_DIR/test.parquet"
else
    echo "✗ Error processing test data"
    exit 1
fi

# ==================== Step 3: Display Statistics ====================
echo ""
echo "============================================"
echo "Data Processing Completed!"
echo "============================================"
echo ""
echo "Files created:"
echo "  1. $LOCAL_DIR/train.parquet (Training: NQ + HotpotQA)"
echo "  2. $LOCAL_DIR/test.parquet (Test: 7 datasets)"
echo ""

# 显示统计信息
if command -v python &> /dev/null; then
    echo "Training data statistics:"
    python -c "
import pandas as pd
try:
    df = pd.read_parquet('$LOCAL_DIR/train.parquet')
    print(f'  Total samples: {len(df)}')
    print('  Breakdown by dataset:')
    counts = df['data_source'].value_counts()
    for dataset, count in counts.items():
        print(f'    - {dataset}: {count}')
except Exception as e:
    print(f'  Error reading train.parquet: {e}')
"
    echo ""
    echo "Test data statistics:"
    python -c "
import pandas as pd
try:
    df = pd.read_parquet('$LOCAL_DIR/test.parquet')
    print(f'  Total samples: {len(df)}')
    print('  Breakdown by dataset:')
    counts = df['data_source'].value_counts()
    for dataset, count in counts.items():
        print(f'    - {dataset}: {count}')
except Exception as e:
    print(f'  Error reading test.parquet: {e}')
"
fi

echo ""
echo "============================================"
echo "Next Steps:"
echo "1. Launch BM25 retrieval server:"
echo "   bash retrieval_launch_bm25.sh"
echo ""
echo "2. Start training:"
echo "   bash train_nq_hotpotqa_qwen25_3b_4gpu.sh"
echo ""
echo "3. Evaluate on TriviaQA:"
echo "   bash eval_triviaqa.sh"
echo "============================================"