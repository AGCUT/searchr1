#!/bin/bash
#
# Launch BM25 + Rerank Two-Stage Retrieval Server
#
# This script starts a FastAPI server at http://127.0.0.1:8000/retrieve
# Stage 1: BM25 retrieval (fast, sparse)
# Stage 2: CrossEncoder reranking (accurate, neural)
#
# Make sure to run this BEFORE starting RL training
#

# Set GPU for reranker (BM25 runs on CPU)
export CUDA_VISIBLE_DEVICES=0

# Paths - modify these to match your setup
FILE_PATH=/usr/yuque/guo/search-r1/data/wiki-corpus
BM25_INDEX=$FILE_PATH/bm25
CORPUS_FILE=$FILE_PATH/wiki-18.jsonl

# Retrieval settings
RETRIEVAL_TOPK=10     # Number of candidates from BM25
RERANK_TOPK=3         # Final number of documents after reranking

# Reranker model - choose one:
# Option 1: MS MARCO MiniLM (fast, good quality)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L12-v2
RERANKER_TYPE=sentence_transformer

# Option 2: BGE Reranker (Chinese + English, very good quality)
# RERANKER_MODEL=BAAI/bge-reranker-base
# RERANKER_TYPE=bge_reranker

# Option 3: BGE Reranker v2 (multilingual, best quality)
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3
# RERANKER_TYPE=bge_reranker

RERANKER_BATCH_SIZE=32

echo "============================================================================"
echo "Starting BM25 + Rerank Two-Stage Retrieval Server"
echo "============================================================================"
echo "BM25 Index: $BM25_INDEX"
echo "Corpus: $CORPUS_FILE"
echo "Retrieval Top-K: $RETRIEVAL_TOPK"
echo "Rerank Top-K: $RERANK_TOPK"
echo "Reranker: $RERANKER_MODEL ($RERANKER_TYPE)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

# Launch server
python search_r1/search/retrieval_rerank_server.py \
    --index_path $BM25_INDEX \
    --corpus_path $CORPUS_FILE \
    --retrieval_topk $RETRIEVAL_TOPK \
    --retriever_name bm25 \
    --reranking_topk $RERANK_TOPK \
    --reranker_model $RERANKER_MODEL \
    --reranker_type $RERANKER_TYPE \
    --reranker_batch_size $RERANKER_BATCH_SIZE \
    --port 8000

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"