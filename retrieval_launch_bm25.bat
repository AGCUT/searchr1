@echo off
REM BM25 Retrieval Server Launcher
REM This script launches the BM25 retrieval service for Search-R1

set file_path=D:\search-r1\wiki_data
set index_file=%file_path%\bm25
set corpus_file=%file_path%\wiki-18.jsonl
set retriever_name=bm25

echo ============================================
echo Starting BM25 Retrieval Server
echo ============================================
echo Index path: %index_file%
echo Corpus path: %corpus_file%
echo Retriever: %retriever_name%
echo Top-k: 3
echo ============================================

python search_r1\search\retrieval_server.py ^
    --index_path %index_file% ^
    --corpus_path %corpus_file% ^
    --topk 3 ^
    --retriever_name %retriever_name%
