#!/usr/bin/env python3
"""
检索评估工具

支持的评估指标：
- Recall@K: 召回率
- Precision@K: 精确率
- MRR@K (Mean Reciprocal Rank): 平均倒数排名
- nDCG@K (Normalized Discounted Cumulative Gain): 归一化折损累积增益
- MAP@K (Mean Average Precision): 平均精度均值
- Hit@K: 命中率

使用示例：
    python retrieval_evaluator.py \
        --retriever_url http://127.0.0.1:8000/retrieve \
        --eval_data path/to/eval.jsonl \
        --topk 3 5 10 \
        --output results.json
"""
import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

import requests
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """评估样本"""
    query: str
    relevant_docs: List[str]  # 相关文档ID或内容
    query_id: Optional[str] = None


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    hit: float = 0.0


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(
        self,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        batch_size: int = 32,
        timeout: int = 60
    ):
        """
        初始化评估器

        Args:
            retriever_url: 检索服务器URL
            batch_size: 批处理大小
            timeout: 请求超时时间（秒）
        """
        self.retriever_url = retriever_url
        self.batch_size = batch_size
        self.timeout = timeout

    def _retrieve(
        self,
        queries: List[str],
        topk: int = 10
    ) -> List[List[Dict]]:
        """调用检索服务"""
        try:
            response = requests.post(
                self.retriever_url,
                json={
                    "queries": queries,
                    "topk": topk,
                    "return_scores": True
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["result"]
        except Exception as e:
            logger.error(f"检索请求失败: {e}")
            return [[] for _ in queries]

    @staticmethod
    def _is_relevant(retrieved_doc: Dict, relevant_docs: List[str]) -> bool:
        """判断检索文档是否相关"""
        # 检查多种可能的匹配方式
        doc_id = retrieved_doc.get("doc_id", "")
        doc_content = retrieved_doc.get("document", "")

        for rel in relevant_docs:
            # ID匹配
            if doc_id and rel == str(doc_id):
                return True
            # 内容包含匹配
            if rel in doc_content or doc_content in rel:
                return True
            # 标题匹配
            if rel.lower() in doc_content.lower():
                return True

        return False

    @staticmethod
    def calculate_recall(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 Recall@K"""
        if not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_found = sum(
            1 for doc in retrieved_k
            if RetrievalEvaluator._is_relevant(doc, relevant)
        )
        return relevant_found / len(relevant)

    @staticmethod
    def calculate_precision(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 Precision@K"""
        if k == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_found = sum(
            1 for doc in retrieved_k
            if RetrievalEvaluator._is_relevant(doc, relevant)
        )
        return relevant_found / k

    @staticmethod
    def calculate_mrr(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 MRR@K (Mean Reciprocal Rank)"""
        for i, doc in enumerate(retrieved[:k]):
            if RetrievalEvaluator._is_relevant(doc, relevant):
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def calculate_ndcg(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 nDCG@K"""
        if not relevant:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if RetrievalEvaluator._is_relevant(doc, relevant):
                dcg += 1.0 / np.log2(i + 2)  # log2(rank + 1), rank从1开始

        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        if ideal_dcg == 0:
            return 0.0
        return dcg / ideal_dcg

    @staticmethod
    def calculate_map(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 AP@K (Average Precision)"""
        if not relevant:
            return 0.0

        num_relevant = 0
        precision_sum = 0.0

        for i, doc in enumerate(retrieved[:k]):
            if RetrievalEvaluator._is_relevant(doc, relevant):
                num_relevant += 1
                precision_sum += num_relevant / (i + 1)

        if num_relevant == 0:
            return 0.0
        return precision_sum / min(len(relevant), k)

    @staticmethod
    def calculate_hit(retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """计算 Hit@K (是否命中)"""
        for doc in retrieved[:k]:
            if RetrievalEvaluator._is_relevant(doc, relevant):
                return 1.0
        return 0.0

    def evaluate_sample(
        self,
        retrieved: List[Dict],
        relevant: List[str],
        k: int
    ) -> RetrievalMetrics:
        """评估单个样本"""
        return RetrievalMetrics(
            recall=self.calculate_recall(retrieved, relevant, k),
            precision=self.calculate_precision(retrieved, relevant, k),
            mrr=self.calculate_mrr(retrieved, relevant, k),
            ndcg=self.calculate_ndcg(retrieved, relevant, k),
            map_score=self.calculate_map(retrieved, relevant, k),
            hit=self.calculate_hit(retrieved, relevant, k)
        )

    def evaluate(
        self,
        samples: List[EvalSample],
        topk_values: List[int] = [3, 5, 10],
        show_progress: bool = True
    ) -> Dict[int, RetrievalMetrics]:
        """
        批量评估

        Args:
            samples: 评估样本列表
            topk_values: 要评估的K值列表
            show_progress: 是否显示进度条

        Returns:
            每个K值对应的平均指标
        """
        max_k = max(topk_values)

        # 批量检索
        all_retrieved = []
        queries = [s.query for s in samples]

        iterator = range(0, len(queries), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="检索中")

        for start in iterator:
            batch_queries = queries[start:start + self.batch_size]
            batch_results = self._retrieve(batch_queries, topk=max_k)
            all_retrieved.extend(batch_results)

        # 计算指标
        results = {}
        for k in topk_values:
            metrics_list = []

            for sample, retrieved in zip(samples, all_retrieved):
                metrics = self.evaluate_sample(retrieved, sample.relevant_docs, k)
                metrics_list.append(metrics)

            # 计算平均值
            avg_metrics = RetrievalMetrics(
                recall=np.mean([m.recall for m in metrics_list]),
                precision=np.mean([m.precision for m in metrics_list]),
                mrr=np.mean([m.mrr for m in metrics_list]),
                ndcg=np.mean([m.ndcg for m in metrics_list]),
                map_score=np.mean([m.map_score for m in metrics_list]),
                hit=np.mean([m.hit for m in metrics_list])
            )
            results[k] = avg_metrics

        return results


def load_eval_data(file_path: str) -> List[EvalSample]:
    """
    加载评估数据

    支持的格式：
    {"query": "...", "relevant_docs": ["doc1", "doc2"], "query_id": "q1"}
    或
    {"question": "...", "answers": ["ans1", "ans2"]}
    """
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # 兼容不同格式
            query = item.get('query') or item.get('question') or item.get('input')
            relevant = (
                item.get('relevant_docs') or
                item.get('answers') or
                item.get('positive_passages', [])
            )

            if isinstance(relevant, str):
                relevant = [relevant]

            if query:
                samples.append(EvalSample(
                    query=query,
                    relevant_docs=relevant,
                    query_id=item.get('query_id') or item.get('id')
                ))

    return samples


def print_results(results: Dict[int, RetrievalMetrics]):
    """打印评估结果"""
    print("\n" + "=" * 80)
    print("检索评估结果")
    print("=" * 80)

    # 表头
    header = f"{'K':>5} | {'Recall':>10} | {'Precision':>10} | {'MRR':>10} | {'nDCG':>10} | {'MAP':>10} | {'Hit':>10}"
    print(header)
    print("-" * 80)

    # 数据行
    for k in sorted(results.keys()):
        m = results[k]
        row = f"{k:>5} | {m.recall:>10.4f} | {m.precision:>10.4f} | {m.mrr:>10.4f} | {m.ndcg:>10.4f} | {m.map_score:>10.4f} | {m.hit:>10.4f}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="检索评估工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--retriever_url", type=str,
                       default="http://127.0.0.1:8000/retrieve",
                       help="检索服务器URL")
    parser.add_argument("--eval_data", type=str, required=True,
                       help="评估数据文件路径 (JSONL格式)")
    parser.add_argument("--topk", type=int, nargs="+", default=[3, 5, 10],
                       help="要评估的K值列表")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小")
    parser.add_argument("--timeout", type=int, default=60,
                       help="请求超时时间（秒）")
    parser.add_argument("--output", type=str, default=None,
                       help="结果输出文件路径")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大评估样本数（用于测试）")

    args = parser.parse_args()

    # 加载数据
    logger.info(f"加载评估数据: {args.eval_data}")
    samples = load_eval_data(args.eval_data)
    logger.info(f"加载了 {len(samples)} 个样本")

    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"限制为 {len(samples)} 个样本")

    # 创建评估器
    evaluator = RetrievalEvaluator(
        retriever_url=args.retriever_url,
        batch_size=args.batch_size,
        timeout=args.timeout
    )

    # 执行评估
    logger.info("开始评估...")
    start_time = time.time()

    results = evaluator.evaluate(
        samples=samples,
        topk_values=args.topk,
        show_progress=True
    )

    elapsed = time.time() - start_time
    logger.info(f"评估完成，耗时: {elapsed:.2f}秒")

    # 打印结果
    print_results(results)

    # 保存结果
    if args.output:
        output_data = {
            "config": {
                "retriever_url": args.retriever_url,
                "eval_data": args.eval_data,
                "num_samples": len(samples),
                "topk_values": args.topk
            },
            "results": {
                str(k): {
                    "recall": results[k].recall,
                    "precision": results[k].precision,
                    "mrr": results[k].mrr,
                    "ndcg": results[k].ndcg,
                    "map": results[k].map_score,
                    "hit": results[k].hit
                }
                for k in results
            },
            "elapsed_seconds": elapsed
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
