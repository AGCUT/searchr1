#!/usr/bin/env python3
"""
BM25/Dense + Rerank 两阶段检索服务器

支持的检索方法：
- bm25: BM25稀疏检索
- e5: E5密集检索
- bge: BGE密集检索

支持的重排模型：
- sentence_transformer: 基于sentence-transformers的CrossEncoder
- bge_reranker: BGE-reranker系列模型
"""
import os
import re
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import defaultdict

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 设置路径以便导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from retrieval_server import get_retriever, Config as RetrieverConfig
from rerank_server import SentenceTransformerCrossEncoder, BaseCrossEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BM25/Dense + Rerank 检索服务",
    description="两阶段检索：先召回后重排",
    version="1.0.0"
)


def convert_title_format(text: str) -> str:
    """
    转换标题格式: (Title: xxx) content -> "xxx"\ncontent
    """
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'"{title}"\n{content}'
    return text


# ============ BGE Reranker 支持 ============
class BGEReranker(BaseCrossEncoder):
    """
    BGE-Reranker 模型支持
    支持 BAAI/bge-reranker-base, BAAI/bge-reranker-large, BAAI/bge-reranker-v2-m3
    """
    def __init__(self, model, tokenizer, batch_size=32, max_length=512, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _predict(self, pairs: List[tuple]) -> List[float]:
        """批量预测相关性分数"""
        scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # BGE reranker输出logits，取第一列作为相关性分数
                if hasattr(outputs, 'logits'):
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                else:
                    batch_scores = outputs[0].squeeze(-1).cpu().numpy()

                if batch_scores.ndim == 0:
                    batch_scores = [float(batch_scores)]
                else:
                    batch_scores = batch_scores.tolist()

                scores.extend(batch_scores)

        return scores

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        return cls(model, tokenizer, **kwargs)


# ============ Request/Response Models ============
class SearchRequest(BaseModel):
    """检索请求模型"""
    queries: List[str]
    topk_retrieval: Optional[int] = 10  # 召回阶段返回数量
    topk_rerank: Optional[int] = 3      # 重排阶段返回数量
    return_scores: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "queries": ["What is machine learning?"],
                "topk_retrieval": 10,
                "topk_rerank": 3,
                "return_scores": True
            }
        }


class SearchResponse(BaseModel):
    """检索响应模型"""
    result: List[Any]
    retrieval_method: str = ""
    reranker_model: str = ""


# ============ Reranker Config ============
@dataclass
class RerankerArguments:
    """重排器配置"""
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")  # sentence_transformer 或 bge_reranker


def get_reranker(config: RerankerArguments):
    """根据配置获取重排器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载重排模型: {config.rerank_model_name_or_path} (类型: {config.reranker_type})")

    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device=device
        )
    elif config.reranker_type == "bge_reranker":
        return BGEReranker.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=device
        )
    else:
        raise ValueError(f"未知的重排器类型: {config.reranker_type}")


# ============ API Endpoints ============
@app.post("/retrieve", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    """
    两阶段检索端点：召回 + 重排

    流程:
    1. 使用BM25/Dense检索召回 topk_retrieval 个候选文档
    2. 使用CrossEncoder重排，返回 topk_rerank 个最相关文档
    """
    try:
        topk_retrieval = request.topk_retrieval or retriever_config.retrieval_topk
        topk_rerank = request.topk_rerank or reranker_config.rerank_topk

        logger.info(f"处理 {len(request.queries)} 个查询，召回 {topk_retrieval}，重排后返回 {topk_rerank}")

        # Step 1: 召回阶段
        retrieved_docs = retriever.batch_search(
            query_list=request.queries,
            num=topk_retrieval,
            return_score=False
        )

        # Step 2: 重排阶段
        reranked = reranker.rerank(request.queries, retrieved_docs)

        # Step 3: 格式化响应，确保每个查询都有对应结果
        response = []
        for i in range(len(request.queries)):
            doc_scores = reranked.get(i, [])[:topk_rerank]

            if request.return_scores:
                combined = []
                for item in doc_scores:
                    if len(item) >= 3:
                        doc, score, doc_item = item
                        combined.append({
                            "document": convert_title_format(doc),
                            "score": float(score),
                            "doc_id": doc_item.get('id', None) if isinstance(doc_item, dict) else None
                        })
                response.append(combined)
            else:
                response.append([convert_title_format(item[0]) for item in doc_scores if len(item) >= 3])

        return SearchResponse(
            result=response,
            retrieval_method=retriever_config.retrieval_method,
            reranker_model=reranker_config.rerank_model_name_or_path
        )

    except Exception as e:
        logger.error(f"检索错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "retrieval_method": retriever_config.retrieval_method,
        "reranker_model": reranker_config.rerank_model_name_or_path
    }


@app.get("/config")
def get_config():
    """获取当前配置"""
    return {
        "retriever": {
            "method": retriever_config.retrieval_method,
            "topk": retriever_config.retrieval_topk,
            "index_path": retriever_config.index_path,
            "corpus_path": retriever_config.corpus_path
        },
        "reranker": {
            "model": reranker_config.rerank_model_name_or_path,
            "type": reranker_config.reranker_type,
            "topk": reranker_config.rerank_topk,
            "batch_size": reranker_config.batch_size
        }
    }


# ============ 主程序入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BM25/Dense + Rerank 两阶段检索服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")

    # 检索器配置
    parser.add_argument("--index_path", type=str, required=True, help="索引路径")
    parser.add_argument("--corpus_path", type=str, required=True, help="语料库路径")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="召回阶段返回的文档数")
    parser.add_argument("--retriever_name", type=str, default="bm25",
                       choices=["bm25", "e5", "bge", "dpr"],
                       help="检索器类型")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2",
                       help="密集检索模型路径（仅dense检索需要）")
    parser.add_argument("--faiss_gpu", action="store_true", help="使用GPU加速FAISS")
    parser.add_argument("--retrieval_batch_size", type=int, default=512, help="检索批处理大小")

    # 重排器配置
    parser.add_argument("--reranking_topk", type=int, default=3, help="重排后返回的文档数")
    parser.add_argument("--reranker_model", type=str,
                       default="cross-encoder/ms-marco-MiniLM-L12-v2",
                       help="重排模型路径")
    parser.add_argument("--reranker_type", type=str, default="sentence_transformer",
                       choices=["sentence_transformer", "bge_reranker"],
                       help="重排器类型")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="重排批处理大小")
    parser.add_argument("--reranker_max_length", type=int, default=512, help="重排最大序列长度")

    args = parser.parse_args()

    # 构建检索器配置
    logger.info("=" * 60)
    logger.info("正在初始化检索服务器...")
    logger.info("=" * 60)

    retriever_config = RetrieverConfig(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.retrieval_topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.retrieval_batch_size,
    )

    logger.info(f"加载检索器: {args.retriever_name}")
    retriever = get_retriever(retriever_config)
    logger.info("检索器加载完成!")

    # 构建重排器配置
    reranker_config = RerankerArguments(
        rerank_topk=args.reranking_topk,
        rerank_model_name_or_path=args.reranker_model,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        reranker_type=args.reranker_type,
    )

    reranker = get_reranker(reranker_config)
    logger.info("重排器加载完成!")

    # 打印配置信息
    logger.info("=" * 60)
    logger.info("服务配置:")
    logger.info(f"  检索方法: {args.retriever_name}")
    logger.info(f"  召回数量: {args.retrieval_topk}")
    logger.info(f"  重排模型: {args.reranker_model}")
    logger.info(f"  重排类型: {args.reranker_type}")
    logger.info(f"  最终返回: {args.reranking_topk}")
    logger.info(f"  服务地址: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)
