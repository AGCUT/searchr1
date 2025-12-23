#!/bin/bash
#
# Launch E5 + Rerank Two-Stage Retrieval Server
#
# This script starts a FastAPI server at http://127.0.0.1:8000/retrieve
# Stage 1: E5 dense retrieval (semantic matching)
# Stage 2: CrossEncoder reranking (fine-grained relevance)
#
# Make sure to run this BEFORE starting RL training
#

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1

# FAISS GPU temporary memory (GB per GPU)
export FAISS_GPU_TEMP_MEM_GB=30

# Paths - modify these to match your setup
FILE_PATH=/usr/yuque/guo/search-r1/data/wiki-corpus
INDEX_FILE=$FILE_PATH/e5_Flat.index
CORPUS_FILE=$FILE_PATH/wiki-18.jsonl

# E5 model
RETRIEVER_NAME=e5
RETRIEVER_MODEL=intfloat/e5-base-v2

# Retrieval settings
RETRIEVAL_TOPK=10     # Number of candidates from E5
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
echo "Starting E5 + Rerank Two-Stage Retrieval Server"
echo "============================================================================"
echo "E5 Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "E5 Model: $RETRIEVER_MODEL"
echo "Retrieval Top-K: $RETRIEVAL_TOPK"
echo "Rerank Top-K: $RERANK_TOPK"
echo "Reranker: $RERANKER_MODEL ($RERANKER_TYPE)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

# Launch server
python search_r1/search/retrieval_rerank_server.py \
    --index_path $INDEX_FILE \
    --corpus_path $CORPUS_FILE \
    --retrieval_topk $RETRIEVAL_TOPK \
    --retriever_name $RETRIEVER_NAME \
    --retriever_model $RETRIEVER_MODEL \
    --faiss_gpu \
    --reranking_topk $RERANK_TOPK \
    --reranker_model $RERANKER_MODEL \
    --reranker_type $RERANKER_TYPE \
    --reranker_batch_size $RERANKER_BATCH_SIZE \
    --port 8000

echo "============================================================================"
echo "Retrieval server is running at http://127.0.0.1:8000/retrieve"
echo "Press Ctrl+C to stop"
echo "============================================================================"
