#!/usr/bin/env python3
"""
混合检索服务器：支持多种融合策略

功能特性：
1. BM25 + Dense (E5/BGE) 混合检索
2. 支持 RRF (Reciprocal Rank Fusion) 和线性加权融合
3. 可选的重排阶段 (CrossEncoder/BGE-Reranker)
4. 批量处理优化

使用示例：
    # 仅混合检索
    python retrieval_hybrid_server.py \
        --bm25_index_path ./index/bm25 \
        --dense_index_path ./index/e5_Flat.index \
        --corpus_path ./data/wiki.jsonl \
        --dense_model_path intfloat/e5-base-v2

    # 混合检索 + 重排
    python retrieval_hybrid_server.py \
        --bm25_index_path ./index/bm25 \
        --dense_index_path ./index/e5_Flat.index \
        --corpus_path ./data/wiki.jsonl \
        --dense_model_path intfloat/e5-base-v2 \
        --enable_rerank \
        --reranker_model cross-encoder/ms-marco-MiniLM-L12-v2
"""
import os
import re
import sys
import json
import argparse
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from retrieval_server import (
    Config,
    BM25Retriever,
    DenseRetriever,
    get_retriever,
    load_corpus
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ 枚举类型 ============
class FusionMethod(str, Enum):
    """融合方法"""
    RRF = "rrf"                    # Reciprocal Rank Fusion
    LINEAR = "linear"              # 线性加权融合
    WEIGHTED_RRF = "weighted_rrf"  # 加权RRF


class RerankerType(str, Enum):
    """重排器类型"""
    NONE = "none"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    BGE_RERANKER = "bge_reranker"


# ============ 融合算法 ============
class FusionAlgorithm:
    """文档融合算法集合"""

    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[Dict]],
        scores_list: List[List[float]],
        k: int = 60,
        weights: Optional[List[float]] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        RRF (Reciprocal Rank Fusion) 算法

        公式: score(doc) = sum(weight_i / (k + rank_i))

        Args:
            results_list: 多个检索器的结果列表
            scores_list: 多个检索器的分数列表
            k: RRF参数，默认60
            weights: 各检索器的权重，默认均等

        Returns:
            融合后的文档列表和分数
        """
        if weights is None:
            weights = [1.0] * len(results_list)

        doc_map = {}
        rrf_scores = {}

        for retriever_idx, (results, scores) in enumerate(zip(results_list, scores_list)):
            weight = weights[retriever_idx]
            for rank, (doc, score) in enumerate(zip(results, scores), 1):
                # 生成文档ID
                doc_id = doc.get('id') or doc.get('title', '') or hash(doc.get('contents', '')[:200])
                doc_map[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (k + rank)

        # 按分数排序
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_results = [doc_map[doc_id] for doc_id, _ in ranked]
        merged_scores = [score for _, score in ranked]

        return merged_results, merged_scores

    @staticmethod
    def linear_fusion(
        results_list: List[List[Dict]],
        scores_list: List[List[float]],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> Tuple[List[Dict], List[float]]:
        """
        线性加权融合

        公式: score(doc) = sum(weight_i * normalized_score_i)

        Args:
            results_list: 多个检索器的结果列表
            scores_list: 多个检索器的分数列表
            weights: 各检索器的权重
            normalize: 是否对分数进行归一化

        Returns:
            融合后的文档列表和分数
        """
        if weights is None:
            weights = [1.0] * len(results_list)

        doc_map = {}
        combined_scores = {}

        for retriever_idx, (results, scores) in enumerate(zip(results_list, scores_list)):
            weight = weights[retriever_idx]

            # 分数归一化
            if normalize and scores:
                min_s, max_s = min(scores), max(scores)
                if max_s > min_s:
                    scores = [(s - min_s) / (max_s - min_s) for s in scores]
                else:
                    scores = [1.0] * len(scores)

            for doc, score in zip(results, scores):
                doc_id = doc.get('id') or doc.get('title', '') or hash(doc.get('contents', '')[:200])
                doc_map[doc_id] = doc
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weight * score

        # 按分数排序
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        merged_results = [doc_map[doc_id] for doc_id, _ in ranked]
        merged_scores = [score for _, score in ranked]

        return merged_results, merged_scores


# ============ 混合检索器 ============
class HybridRetriever:
    """
    混合检索器：支持 BM25 + Dense 检索的融合

    特性：
    - 支持多种融合算法（RRF、线性加权）
    - 可选的重排阶段
    - 批量处理优化
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion_method: FusionMethod = FusionMethod.RRF,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        reranker=None
    ):
        """
        初始化混合检索器

        Args:
            bm25_retriever: BM25检索器
            dense_retriever: 密集检索器
            fusion_method: 融合方法
            rrf_k: RRF参数
            bm25_weight: BM25权重
            dense_weight: Dense权重
            reranker: 可选的重排器
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.weights = [bm25_weight, dense_weight]
        self.reranker = reranker

        logger.info(f"混合检索器初始化完成")
        logger.info(f"  融合方法: {fusion_method.value}")
        logger.info(f"  权重: BM25={bm25_weight}, Dense={dense_weight}")
        if reranker:
            logger.info(f"  重排器: 已启用")

    def _fuse_results(
        self,
        bm25_results: List[Dict],
        bm25_scores: List[float],
        dense_results: List[Dict],
        dense_scores: List[float]
    ) -> Tuple[List[Dict], List[float]]:
        """融合两个检索器的结果"""
        results_list = [bm25_results, dense_results]
        scores_list = [bm25_scores, dense_scores]

        if self.fusion_method == FusionMethod.RRF:
            return FusionAlgorithm.reciprocal_rank_fusion(
                results_list, scores_list, k=self.rrf_k
            )
        elif self.fusion_method == FusionMethod.WEIGHTED_RRF:
            return FusionAlgorithm.reciprocal_rank_fusion(
                results_list, scores_list, k=self.rrf_k, weights=self.weights
            )
        elif self.fusion_method == FusionMethod.LINEAR:
            return FusionAlgorithm.linear_fusion(
                results_list, scores_list, weights=self.weights
            )
        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")

    def _search(
        self,
        query: str,
        num: int = 3,
        retrieval_num: Optional[int] = None,
        return_score: bool = False
    ) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        """
        单个查询的混合检索

        Args:
            query: 查询文本
            num: 最终返回的文档数
            retrieval_num: 召回阶段的文档数
            return_score: 是否返回分数
        """
        # 召回更多候选用于融合
        retrieve_num = retrieval_num or max(num * 5, 20)

        # BM25 检索
        bm25_results, bm25_scores = self.bm25_retriever.search(
            query, num=retrieve_num, return_score=True
        )

        # Dense 检索
        dense_results, dense_scores = self.dense_retriever.search(
            query, num=retrieve_num, return_score=True
        )

        # 融合
        merged_results, merged_scores = self._fuse_results(
            bm25_results, bm25_scores, dense_results, dense_scores
        )

        # 截取topk
        final_results = merged_results[:num]
        final_scores = merged_scores[:num]

        if return_score:
            return final_results, final_scores
        return final_results

    def batch_search(
        self,
        query_list: List[str],
        num: int = 3,
        retrieval_num: Optional[int] = None,
        return_score: bool = False,
        show_progress: bool = False
    ) -> Union[List[List[Dict]], Tuple[List[List[Dict]], List[List[float]]]]:
        """
        批量混合检索

        Args:
            query_list: 查询列表
            num: 每个查询返回的文档数
            retrieval_num: 召回阶段的文档数
            return_score: 是否返回分数
            show_progress: 是否显示进度条
        """
        all_results = []
        all_scores = []

        iterator = tqdm(query_list, desc="混合检索") if show_progress else query_list

        for query in iterator:
            result, score = self._search(
                query, num=num, retrieval_num=retrieval_num, return_score=True
            )
            all_results.append(result)
            all_scores.append(score)

        if return_score:
            return all_results, all_scores
        return all_results

    def search_with_rerank(
        self,
        query: str,
        num: int = 3,
        retrieval_num: int = 20,
        return_score: bool = False
    ) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        """
        混合检索 + 重排

        Args:
            query: 查询文本
            num: 最终返回的文档数
            retrieval_num: 召回阶段的文档数
            return_score: 是否返回分数
        """
        if not self.reranker:
            raise ValueError("未配置重排器")

        # 第一阶段：混合检索
        candidates, _ = self._search(query, num=retrieval_num, return_score=True)

        # 第二阶段：重排
        reranked = self.reranker.rerank([query], [candidates])

        # 提取结果
        doc_scores = reranked.get(0, [])[:num]

        if return_score:
            results = []
            scores = []
            for item in doc_scores:
                if len(item) >= 3:
                    doc_str, score, doc_item = item
                    results.append(doc_item)
                    scores.append(score)
            return results, scores

        return [item[2] if len(item) >= 3 else item for item in doc_scores]

    def batch_search_with_rerank(
        self,
        query_list: List[str],
        num: int = 3,
        retrieval_num: int = 20,
        return_score: bool = False,
        show_progress: bool = False
    ) -> Union[List[List[Dict]], Tuple[List[List[Dict]], List[List[float]]]]:
        """批量混合检索 + 重排"""
        if not self.reranker:
            raise ValueError("未配置重排器")

        # 第一阶段：批量混合检索
        all_candidates, _ = self.batch_search(
            query_list, num=retrieval_num, return_score=True, show_progress=show_progress
        )

        # 确保候选数量与查询数量一致
        if len(all_candidates) != len(query_list):
            logger.warning(f"候选数量 ({len(all_candidates)}) 与查询数量 ({len(query_list)}) 不匹配，进行补齐")
            while len(all_candidates) < len(query_list):
                all_candidates.append([])

        # 第二阶段：批量重排
        reranked = self.reranker.rerank(query_list, all_candidates)

        # 格式化结果，确保每个查询都有结果
        all_results = []
        all_scores = []

        for i in range(len(query_list)):
            doc_scores = reranked.get(i, [])[:num]
            if return_score:
                results = []
                scores = []
                for item in doc_scores:
                    if len(item) >= 3:
                        doc_str, score, doc_item = item
                        results.append(doc_item)
                        scores.append(score)
                all_results.append(results)
                all_scores.append(scores)
            else:
                all_results.append([item[2] if len(item) >= 3 else item for item in doc_scores])

        if return_score:
            return all_results, all_scores
        return all_results


# ============ FastAPI 服务 ============
class QueryRequest(BaseModel):
    """查询请求"""
    queries: List[str]
    topk: Optional[int] = 3
    topk_retrieval: Optional[int] = None  # 召回数量，如果使用重排
    return_scores: bool = False
    use_rerank: Optional[bool] = None  # 是否使用重排（如果服务器配置了重排器）

    class Config:
        json_schema_extra = {
            "example": {
                "queries": ["What is machine learning?"],
                "topk": 3,
                "topk_retrieval": 20,
                "return_scores": True,
                "use_rerank": True
            }
        }


class QueryResponse(BaseModel):
    """查询响应"""
    result: List[Any]
    fusion_method: str = ""
    reranker_used: bool = False


def convert_title_format(text: str) -> str:
    """转换标题格式"""
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'"{title}"\n{content}'
    return text


def format_document(doc: Dict) -> str:
    """格式化文档为字符串"""
    if 'contents' in doc:
        content = doc['contents']
        title = content.split("\n")[0].strip('"')
        text = "\n".join(content.split("\n")[1:])
        return f'"{title}"\n{text}'
    return str(doc)


app = FastAPI(
    title="混合检索服务",
    description="BM25 + Dense 混合检索，支持RRF融合和可选重排",
    version="2.0.0"
)


@app.post("/retrieve", response_model=QueryResponse)
def retrieve_endpoint(request: QueryRequest):
    """
    混合检索端点

    支持三种模式：
    1. 纯混合检索（BM25 + Dense RRF融合）
    2. 混合检索 + 重排（需要配置重排器）
    """
    try:
        topk = request.topk or 3
        use_rerank = request.use_rerank if request.use_rerank is not None else enable_rerank

        logger.info(f"处理 {len(request.queries)} 个查询, topk={topk}, rerank={use_rerank}")

        if use_rerank and retriever.reranker:
            retrieval_num = request.topk_retrieval or 20
            results, scores = retriever.batch_search_with_rerank(
                query_list=request.queries,
                num=topk,
                retrieval_num=retrieval_num,
                return_score=True
            )
        else:
            results, scores = retriever.batch_search(
                query_list=request.queries,
                num=topk,
                return_score=True
            )

        # 格式化响应
        resp = []
        for i, (docs, doc_scores) in enumerate(zip(results, scores)):
            if request.return_scores:
                combined = []
                for doc, score in zip(docs, doc_scores):
                    combined.append({
                        "document": format_document(doc),
                        "score": float(score),
                        "doc_id": doc.get('id', None)
                    })
                resp.append(combined)
            else:
                resp.append([format_document(doc) for doc in docs])

        return QueryResponse(
            result=resp,
            fusion_method=fusion_method.value,
            reranker_used=use_rerank and retriever.reranker is not None
        )

    except Exception as e:
        logger.error(f"检索错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "retriever": "hybrid (BM25 + Dense)",
        "fusion_method": fusion_method.value,
        "reranker_enabled": retriever.reranker is not None
    }


@app.get("/config")
def get_config():
    """获取当前配置"""
    return {
        "fusion": {
            "method": fusion_method.value,
            "rrf_k": rrf_k,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight
        },
        "bm25": {
            "index_path": bm25_index_path
        },
        "dense": {
            "index_path": dense_index_path,
            "model": dense_model_path
        },
        "reranker": {
            "enabled": enable_rerank,
            "model": reranker_model if enable_rerank else None
        }
    }


# ============ 主程序 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="混合检索服务器 (BM25 + Dense)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")

    # BM25 配置
    parser.add_argument("--bm25_index_path", type=str, required=True, help="BM25索引路径")
    parser.add_argument("--bm25_corpus_path", type=str, default=None, help="BM25语料库路径（可选，默认使用--corpus_path）")

    # Dense 配置
    parser.add_argument("--dense_index_path", type=str, required=True, help="Dense检索FAISS索引路径")
    parser.add_argument("--dense_corpus_path", type=str, default=None, help="Dense语料库路径（可选，默认使用--corpus_path）")
    parser.add_argument("--dense_model_path", type=str, default="intfloat/e5-base-v2",
                       help="Dense检索模型路径")
    parser.add_argument("--dense_model_type", type=str, default="e5",
                       choices=["e5", "bge", "dpr"], help="Dense检索模型类型")

    # 通用配置
    parser.add_argument("--corpus_path", type=str, default=None, help="语料库路径（当BM25和Dense使用相同语料库时）")
    parser.add_argument("--topk", type=int, default=3, help="默认返回文档数")
    parser.add_argument("--faiss_gpu", action="store_true", help="使用GPU加速FAISS")

    # 融合配置
    parser.add_argument("--fusion_method", type=str, default="rrf",
                       choices=["rrf", "weighted_rrf", "linear"], help="融合方法")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF参数k")
    parser.add_argument("--bm25_weight", type=float, default=1.0, help="BM25权重")
    parser.add_argument("--dense_weight", type=float, default=1.0, help="Dense权重")

    # 重排配置
    parser.add_argument("--enable_rerank", action="store_true", help="启用重排")
    parser.add_argument("--reranker_model", type=str,
                       default="cross-encoder/ms-marco-MiniLM-L12-v2", help="重排模型路径")
    parser.add_argument("--reranker_type", type=str, default="sentence_transformer",
                       choices=["sentence_transformer", "bge_reranker"], help="重排器类型")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="重排批处理大小")

    args = parser.parse_args()

    # 处理语料库路径
    bm25_corpus = args.bm25_corpus_path or args.corpus_path
    dense_corpus = args.dense_corpus_path or args.corpus_path

    if not bm25_corpus or not dense_corpus:
        parser.error("必须指定语料库路径：使用 --corpus_path 或分别指定 --bm25_corpus_path 和 --dense_corpus_path")

    # 保存配置到全局变量
    fusion_method = FusionMethod(args.fusion_method)
    rrf_k = args.rrf_k
    bm25_weight = args.bm25_weight
    dense_weight = args.dense_weight
    bm25_index_path = args.bm25_index_path
    dense_index_path = args.dense_index_path
    dense_model_path = args.dense_model_path
    enable_rerank = args.enable_rerank
    reranker_model = args.reranker_model

    logger.info("=" * 60)
    logger.info("正在启动混合检索服务器...")
    logger.info("=" * 60)

    # 创建 BM25 检索器
    logger.info("初始化 BM25 检索器...")
    bm25_config = Config(
        retrieval_method="bm25",
        index_path=args.bm25_index_path,
        corpus_path=bm25_corpus,
        retrieval_topk=args.topk,
    )
    bm25_retriever = BM25Retriever(bm25_config)

    # 创建 Dense 检索器
    logger.info(f"初始化 Dense 检索器 ({args.dense_model_type})...")
    dense_config = Config(
        retrieval_method=args.dense_model_type,
        index_path=args.dense_index_path,
        corpus_path=dense_corpus,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.dense_model_path,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    dense_retriever = DenseRetriever(dense_config)

    # 创建重排器（可选）
    reranker = None
    if args.enable_rerank:
        logger.info(f"初始化重排器: {args.reranker_model}")
        if args.reranker_type == "sentence_transformer":
            from rerank_server import SentenceTransformerCrossEncoder
            reranker = SentenceTransformerCrossEncoder.load(
                args.reranker_model,
                batch_size=args.reranker_batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        elif args.reranker_type == "bge_reranker":
            from retrieval_rerank_server import BGEReranker
            reranker = BGEReranker.load(
                args.reranker_model,
                batch_size=args.reranker_batch_size,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    # 创建混合检索器
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        fusion_method=fusion_method,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        reranker=reranker
    )

    # 打印配置
    logger.info("=" * 60)
    logger.info("服务配置:")
    logger.info(f"  融合方法: {args.fusion_method}")
    logger.info(f"  RRF k: {args.rrf_k}")
    logger.info(f"  权重: BM25={args.bm25_weight}, Dense={args.dense_weight}")
    logger.info(f"  重排: {'启用 (' + args.reranker_model + ')' if args.enable_rerank else '禁用'}")
    logger.info(f"  服务地址: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
