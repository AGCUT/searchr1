#!/bin/bash
#
# Launch Hybrid Retrieval + Rerank Server (BM25 + E5 + Reranker)
#
# This is the most powerful retrieval pipeline:
# Stage 1: BM25 sparse retrieval
# Stage 2: E5 dense retrieval
# Stage 3: RRF fusion of Stage 1 & 2
# Stage 4: CrossEncoder reranking
#
# Make sure to run this BEFORE starting RL training
#

# Set GPUs for dense retrieval and reranker
export CUDA_VISIBLE_DEVICES=0,1

# FAISS GPU temporary memory (GB per GPU)
export FAISS_GPU_TEMP_MEM_GB=30

# Paths - modify these to match your setup
# BM25 索引和语料库
BM25_INDEX=/usr/yuque/guo/searchr1/data/wiki_data/bm25
BM25_CORPUS=/usr/yuque/guo/searchr1/data/wiki_data/wiki-18.jsonl

# E5 索引和语料库
DENSE_INDEX=/usr/yuque/guo/search-r1/data/wiki-corpus/e5_Flat.index
DENSE_CORPUS=/usr/yuque/guo/search-r1/data/wiki-corpus/wiki-18.jsonl

# 重排模型
RERANKER_MODEL=/usr/yuque/guo/searchr1/models/BAAI/bge-reranker-base
RERANKER_TYPE=bge_reranker

# Dense model
DENSE_MODEL=intfloat/e5-base-v2
DENSE_MODEL_TYPE=e5

# Retrieval settings
TOPK=3                  # Final number of documents to return

# Fusion settings
FUSION_METHOD=rrf       # Options: rrf, weighted_rrf, linear
RRF_K=60                # RRF parameter
BM25_WEIGHT=1.0         # BM25 weight
DENSE_WEIGHT=1.0        # Dense weight

RERANKER_BATCH_SIZE=32

echo "============================================================================"
echo "Starting Hybrid + Rerank Retrieval Server"
echo "============================================================================"
echo "BM25 Index: $BM25_INDEX"
echo "BM25 Corpus: $BM25_CORPUS"
echo "Dense Index: $DENSE_INDEX"
echo "Dense Corpus: $DENSE_CORPUS"
echo "Dense Model: $DENSE_MODEL"
echo "Fusion: $FUSION_METHOD (k=$RRF_K)"
echo "Weights: BM25=$BM25_WEIGHT, Dense=$DENSE_WEIGHT"
echo "Reranker: $RERANKER_MODEL ($RERANKER_TYPE)"
echo "Top-K: $TOPK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

# Launch server with reranking enabled
python search_r1/search/retrieval_hybrid_server.py \
    --bm25_index_path $BM25_INDEX \
    --bm25_corpus_path $BM25_CORPUS \
    --dense_index_path $DENSE_INDEX \
    --dense_corpus_path $DENSE_CORPUS \
    --dense_model_path $DENSE_MODEL \
    --dense_model_type $DENSE_MODEL_TYPE \
    --topk $TOPK \
    --fusion_method $FUSION_METHOD \
    --rrf_k $RRF_K \
    --bm25_weight $BM25_WEIGHT \
    --dense_weight $DENSE_WEIGHT \
    --faiss_gpu \
    --enable_rerank \
    --reranker_model $RERANKER_MODEL \
    --reranker_type $RERANKER_TYPE \
    --reranker_batch_size $RERANKER_BATCH_SIZE \
    --port 8000

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"
