#!/bin/bash
#
# Launch Hybrid Retrieval Server (BM25 + E5 Dense with RRF Fusion)
#
# This script starts a FastAPI server at http://127.0.0.1:8000/retrieve
# Combines sparse (BM25) and dense (E5) retrieval using Reciprocal Rank Fusion
#
# Make sure to run this BEFORE starting RL training
#

# Set GPUs for dense retrieval (BM25 runs on CPU)
export CUDA_VISIBLE_DEVICES=0,1

# FAISS GPU temporary memory (GB per GPU)
export FAISS_GPU_TEMP_MEM_GB=30

# Paths - modify these to match your setup
FILE_PATH=/usr/yuque/guo/search-r1/data/wiki-corpus
BM25_INDEX=$FILE_PATH/bm25
DENSE_INDEX=$FILE_PATH/e5_Flat.index
CORPUS_FILE=$FILE_PATH/wiki-18.jsonl

# Dense model
DENSE_MODEL=intfloat/e5-base-v2
DENSE_MODEL_TYPE=e5

# Retrieval settings
TOPK=3

# Fusion settings
FUSION_METHOD=rrf       # Options: rrf, weighted_rrf, linear
RRF_K=60                # RRF parameter (default 60)
BM25_WEIGHT=1.0         # BM25 weight in fusion
DENSE_WEIGHT=1.0        # Dense weight in fusion

echo "============================================================================"
echo "Starting Hybrid Retrieval Server (BM25 + E5)"
echo "============================================================================"
echo "BM25 Index: $BM25_INDEX"
echo "Dense Index: $DENSE_INDEX"
echo "Corpus: $CORPUS_FILE"
echo "Dense Model: $DENSE_MODEL"
echo "Top-K: $TOPK"
echo "Fusion: $FUSION_METHOD (k=$RRF_K)"
echo "Weights: BM25=$BM25_WEIGHT, Dense=$DENSE_WEIGHT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

# Launch server without reranking
python search_r1/search/retrieval_hybrid_server.py \
    --bm25_index_path $BM25_INDEX \
    --dense_index_path $DENSE_INDEX \
    --corpus_path $CORPUS_FILE \
    --dense_model_path $DENSE_MODEL \
    --dense_model_type $DENSE_MODEL_TYPE \
    --topk $TOPK \
    --fusion_method $FUSION_METHOD \
    --rrf_k $RRF_K \
    --bm25_weight $BM25_WEIGHT \
    --dense_weight $DENSE_WEIGHT \
    --faiss_gpu \
    --port 8000

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"