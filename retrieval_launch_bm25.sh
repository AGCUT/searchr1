#!/bin/bash
# BM25 Retrieval Server Launcher for Linux
# This script launches the BM25 retrieval service for Search-R1

file_path=/path/to/wiki_data  # 修改为你的实际路径
index_file=$file_path/bm25
corpus_file=$file_path/wiki-18.jsonl
retriever_name=bm25

echo "============================================"
echo "Starting BM25 Retrieval Server"
echo "============================================"
echo "Index path: $index_file"
echo "Corpus path: $corpus_file"
echo "Retriever: $retriever_name"
echo "Top-k: 3"
echo "============================================"

python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name