#!/bin/bash
# BM25 + Reranker Retrieval Server Launcher
# This script launches BM25 retrieval with reranking for better accuracy

file_path=/usr/yuque/guo/searchr1/data/wiki_data
index_file=$file_path/bm25
corpus_file=$file_path/wiki-18.jsonl
retriever_name=bm25

echo "============================================"
echo "Starting BM25 + Reranker Retrieval Server"
echo "============================================"
echo "Index path: $index_file"
echo "Corpus path: $corpus_file"
echo "Retriever: $retriever_name"
echo "Retrieval Top-k: 10"
echo "Reranking Top-k: 3"
echo "Reranker: cross-encoder/ms-marco-MiniLM-L12-v2"
echo "============================================"

python search_r1/search/retrieval_rerank_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --retrieval_topk 10 \
    --retriever_name $retriever_name \
    --reranking_topk 3 \
    --reranker_model cross-encoder/ms-marco-MiniLM-L12-v2 \
    --reranker_batch_size 32
