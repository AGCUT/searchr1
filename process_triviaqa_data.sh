#!/bin/bash
# TriviaQA Data Processing Script for Linux

echo "============================================"
echo "Processing TriviaQA Dataset"
echo "============================================"

# 设置保存路径
LOCAL_DIR=./data/triviaqa_search

echo "Output directory: $LOCAL_DIR"
echo "============================================"

# 运行数据处理
python scripts/data_process/triviaqa_search.py \
    --local_dir $LOCAL_DIR \
    --template_type base

echo "============================================"
echo "Data processing completed!"
echo "Files saved to: $LOCAL_DIR"
echo "- train.parquet"
echo "- test.parquet"
echo "============================================"