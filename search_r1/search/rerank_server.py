#!/usr/bin/env python3
"""
重排服务器

支持的重排模型：
- sentence_transformer: 基于sentence-transformers的CrossEncoder (如 ms-marco-MiniLM)
- bge_reranker: BGE-reranker系列模型

使用示例：
    # 使用 MS MARCO CrossEncoder
    python rerank_server.py \
        --rerank_model_name_or_path cross-encoder/ms-marco-MiniLM-L12-v2 \
        --reranker_type sentence_transformer

    # 使用 BGE Reranker
    python rerank_server.py \
        --rerank_model_name_or_path BAAI/bge-reranker-base \
        --reranker_type bge_reranker
"""
import argparse
import logging
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from transformers import HfArgumentParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseCrossEncoder:
    """重排器基类"""

    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        if hasattr(model, 'to'):
            self.model.to(device)

    def _passage_to_string(self, doc_item: Dict) -> str:
        """将文档字典转换为字符串"""
        # 支持多种文档格式
        if "document" in doc_item:
            if isinstance(doc_item["document"], dict):
                content = doc_item["document"].get("contents", "")
            else:
                content = doc_item["document"]
        elif "contents" in doc_item:
            content = doc_item["contents"]
        elif "text" in doc_item:
            content = doc_item["text"]
        elif "content" in doc_item:
            content = doc_item["content"]
        else:
            content = str(doc_item)

        # 提取标题和正文
        if isinstance(content, str) and "\n" in content:
            lines = content.split("\n")
            title = lines[0].strip('"').strip()
            text = "\n".join(lines[1:])
            return f"(Title: {title}) {text}"

        return content

    def rerank(
        self,
        queries: List[str],
        documents: List[List[Dict]],
        return_doc_item: bool = True
    ) -> Dict[int, List[Tuple]]:
        """
        对每个查询的文档进行重排

        Args:
            queries: 查询字符串列表
            documents: 文档列表的列表（每个查询对应一个文档列表）
            return_doc_item: 是否返回原始文档项

        Returns:
            字典：query_id -> 排序后的元组列表
            - 如果 return_doc_item=True: (doc_string, score, doc_item)
            - 如果 return_doc_item=False: (doc_string, score)
        """
        if len(queries) != len(documents):
            raise ValueError(f"查询数量 ({len(queries)}) 与文档列表数量 ({len(documents)}) 不匹配")

        pairs = []
        qids = []
        doc_items = []

        for qid, (query, doc_list) in enumerate(zip(queries, documents)):
            for doc_item in doc_list:
                doc = self._passage_to_string(doc_item)
                pairs.append((query, doc))
                qids.append(qid)
                doc_items.append(doc_item)

        if not pairs:
            return {}

        # 批量预测
        scores = self._predict(pairs)

        # 组织结果
        query_to_doc_scores = defaultdict(list)

        for i in range(len(pairs)):
            _, doc = pairs[i]
            score = scores[i]
            qid = qids[i]
            doc_item = doc_items[i]

            if return_doc_item:
                query_to_doc_scores[qid].append((doc, score, doc_item))
            else:
                query_to_doc_scores[qid].append((doc, score))

        # 按分数排序
        sorted_query_to_doc_scores = {}
        for qid, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[qid] = sorted(
                doc_scores, key=lambda x: x[1], reverse=True
            )

        return sorted_query_to_doc_scores

    def _predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """预测相关性分数（子类实现）"""
        raise NotImplementedError

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        """加载模型（子类实现）"""
        raise NotImplementedError


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    """基于 sentence-transformers 的 CrossEncoder"""

    def __init__(self, model, batch_size=32, device="cuda"):
        super().__init__(model, batch_size, device)

    def _predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """使用 CrossEncoder 预测分数"""
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        if isinstance(scores, (torch.Tensor, np.ndarray)):
            scores = scores.tolist()
        return scores

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        logger.info(f"加载 CrossEncoder: {model_name_or_path}")
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)


class BGEReranker(BaseCrossEncoder):
    """
    BGE-Reranker 模型

    支持的模型:
    - BAAI/bge-reranker-base
    - BAAI/bge-reranker-large
    - BAAI/bge-reranker-v2-m3 (多语言)
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size=32,
        max_length=512,
        device="cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
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
                # BGE reranker 输出 logits
                if hasattr(outputs, 'logits'):
                    batch_scores = outputs.logits.squeeze(-1)
                else:
                    batch_scores = outputs[0].squeeze(-1)

                # 处理不同维度
                if batch_scores.ndim == 0:
                    batch_scores = [float(batch_scores.cpu())]
                else:
                    batch_scores = batch_scores.cpu().numpy().tolist()

                scores.extend(batch_scores)

        return scores

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"加载 BGE Reranker: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        return cls(model, tokenizer, **kwargs)


# ============ Request/Response Models ============
class RerankRequest(BaseModel):
    """重排请求"""
    queries: List[str]
    documents: List[List[Dict[str, Any]]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "queries": ["What is Python?"],
                "documents": [[
                    {"contents": "\"Python\"\nPython is a programming language."},
                    {"contents": "\"Java\"\nJava is another programming language."}
                ]],
                "rerank_topk": 3,
                "return_scores": True
            }
        }


class RerankResponse(BaseModel):
    """重排响应"""
    result: List[Any]
    model: str = ""


# ============ Reranker Config ============
@dataclass
class RerankerArguments:
    """重排器配置"""
    max_length: int = field(default=512, metadata={"help": "最大序列长度"})
    rerank_topk: int = field(default=3, metadata={"help": "重排后返回的文档数"})
    rerank_model_name_or_path: str = field(
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
        metadata={"help": "重排模型路径"}
    )
    batch_size: int = field(default=32, metadata={"help": "批处理大小"})
    reranker_type: str = field(
        default="sentence_transformer",
        metadata={"help": "重排器类型: sentence_transformer 或 bge_reranker"}
    )
    port: int = field(default=6980, metadata={"help": "服务端口"})
    host: str = field(default="0.0.0.0", metadata={"help": "服务主机"})


def get_reranker(config: RerankerArguments):
    """根据配置获取重排器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

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


# ============ FastAPI App ============
app = FastAPI(
    title="重排服务",
    description="文档重排API，支持多种CrossEncoder模型",
    version="1.0.0"
)


@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    """
    重排端点

    对每个查询的候选文档进行重排，返回按相关性排序的文档
    """
    try:
        topk = request.rerank_topk or config.rerank_topk

        logger.info(f"处理 {len(request.queries)} 个查询的重排请求")

        # 执行重排
        query_to_doc_scores = reranker.rerank(request.queries, request.documents)

        # 格式化响应
        resp = []
        for qid in range(len(request.queries)):
            doc_scores = query_to_doc_scores.get(qid, [])[:topk]

            if request.return_scores:
                combined = []
                for doc, score, doc_item in doc_scores:
                    combined.append({
                        "document": doc,
                        "score": float(score),
                        "doc_id": doc_item.get('id', None) if isinstance(doc_item, dict) else None
                    })
                resp.append(combined)
            else:
                resp.append([doc for doc, _, _ in doc_scores])

        return RerankResponse(
            result=resp,
            model=config.rerank_model_name_or_path
        )

    except Exception as e:
        logger.error(f"重排错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model": config.rerank_model_name_or_path,
        "type": config.reranker_type
    }


@app.get("/config")
def get_config_endpoint():
    """获取当前配置"""
    return {
        "model": config.rerank_model_name_or_path,
        "type": config.reranker_type,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "default_topk": config.rerank_topk
    }


if __name__ == "__main__":
    # 解析命令行参数
    parser = HfArgumentParser((RerankerArguments,))
    config = parser.parse_args_into_dataclasses()[0]

    logger.info("=" * 60)
    logger.info("启动重排服务器")
    logger.info("=" * 60)

    # 初始化重排器
    reranker = get_reranker(config)

    logger.info("=" * 60)
    logger.info(f"模型: {config.rerank_model_name_or_path}")
    logger.info(f"类型: {config.reranker_type}")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info("=" * 60)

    # 启动服务
    uvicorn.run(app, host=config.host, port=config.port)
