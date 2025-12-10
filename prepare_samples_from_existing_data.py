#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于现有数据集构造训练样本的实用工具

这个版本更实用：
1. 直接使用你的BM25检索系统
2. 自动分析检索质量
3. 自动标注样本类型
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import re


class RetrievalQualityAnalyzer:
    """检索质量分析器"""

    @staticmethod
    def compute_retrieval_quality(question: str, docs: List[Dict], golden_answer: str) -> float:
        """
        计算检索质量分数 (0-1)

        判断依据：
        1. 文档中是否包含golden answer的关键词
        2. 文档与问题的相关性
        3. 答案在文档中的位置和清晰度
        """
        if not docs:
            return 0.0

        scores = []

        # 标准化golden answer（处理多个可能的答案）
        if isinstance(golden_answer, list):
            golden_answers = golden_answer
        else:
            golden_answers = [golden_answer]

        for doc in docs:
            doc_text = doc.get('text', '') + ' ' + doc.get('title', '')
            doc_text_lower = doc_text.lower()

            score = 0.0

            # 检查1：是否直接包含答案
            for answer in golden_answers:
                answer_clean = answer.strip().lower()
                if answer_clean in doc_text_lower:
                    score = max(score, 1.0)
                    break

            # 检查2：是否包含答案的关键词
            if score < 1.0:
                answer_keywords = set()
                for answer in golden_answers:
                    # 提取关键词（数字、专有名词等）
                    keywords = re.findall(r'\b\w+\b', answer.lower())
                    keywords = [k for k in keywords if len(k) > 2]
                    answer_keywords.update(keywords)

                if answer_keywords:
                    matched_keywords = sum(1 for kw in answer_keywords if kw in doc_text_lower)
                    keyword_ratio = matched_keywords / len(answer_keywords)
                    score = max(score, keyword_ratio * 0.7)

            # 检查3：文档与问题的相关性
            question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
            question_keywords = {k for k in question_keywords if len(k) > 3}

            if question_keywords:
                matched_q_keywords = sum(1 for kw in question_keywords if kw in doc_text_lower)
                relevance = matched_q_keywords / len(question_keywords)
                score = max(score, relevance * 0.5)

            scores.append(score)

        # 返回最高分（只要有一个文档包含答案就算好）
        return max(scores) if scores else 0.0

    @staticmethod
    def classify_sample_type(retrieval_quality: float) -> Tuple[str, str]:
        """
        根据检索质量分类样本类型

        Returns:
            (sample_type, expected_behavior)
        """
        if retrieval_quality >= 0.7:
            return "retrievable_high", "answer_correctly"
        elif retrieval_quality >= 0.4:
            return "retrievable_medium", "answer_with_reasoning"
        elif retrieval_quality >= 0.2:
            return "retrievable_low", "answer_with_caution"
        else:
            return "not_retrievable", "admit_unknown"


class SamplePreparer:
    """样本准备器"""

    def __init__(self, base_dataset_path: str, retrieval_system=None):
        """
        Args:
            base_dataset_path: 原始数据集路径 (jsonl格式)
            retrieval_system: 你的BM25检索系统（可选，如果没有就用预检索的结果）
        """
        self.base_dataset = self._load_dataset(base_dataset_path)
        self.retrieval_system = retrieval_system
        self.analyzer = RetrievalQualityAnalyzer()

    def _load_dataset(self, path: str) -> List[Dict]:
        """加载数据集"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def prepare_phase1_samples(self, n_samples: int, existing_retrieval: bool = True) -> List[Dict]:
        """
        准备阶段1样本：90% 高质量可检索 + 10% 不可检索

        Args:
            n_samples: 需要的样本数
            existing_retrieval: 是否使用数据集中已有的检索结果
        """
        print(f"\n=== 准备阶段1样本 ===")
        print(f"目标: {n_samples} 个样本")
        print(f"配比: 90% 高质量可检索 + 10% 不可检索")

        samples = []
        sample_counts = defaultdict(int)

        # 首先分析所有样本的检索质量
        print("\n分析检索质量...")
        analyzed_samples = []

        for idx, item in enumerate(self.base_dataset):
            if idx % 100 == 0:
                print(f"  已分析 {idx}/{len(self.base_dataset)} 个样本...")

            question = item.get('question', '')
            golden_answer = item.get('answer', item.get('answers', ''))

            # 获取检索结果
            if existing_retrieval and 'docs' in item:
                docs = item['docs']
            elif self.retrieval_system:
                docs = self.retrieval_system.search(question, top_k=3)
            else:
                print("警告: 没有检索结果，跳过该样本")
                continue

            # 计算检索质量
            quality = self.analyzer.compute_retrieval_quality(question, docs, golden_answer)
            sample_type, expected_behavior = self.analyzer.classify_sample_type(quality)

            analyzed_samples.append({
                'original': item,
                'question': question,
                'golden_answer': golden_answer,
                'docs': docs,
                'retrieval_quality': quality,
                'sample_type': sample_type,
                'expected_behavior': expected_behavior
            })

        print(f"\n总共分析了 {len(analyzed_samples)} 个样本")

        # 统计各类型样本数量
        type_distribution = defaultdict(int)
        for s in analyzed_samples:
            type_distribution[s['sample_type']] += 1

        print("\n样本类型分布:")
        for stype, count in sorted(type_distribution.items()):
            print(f"  {stype}: {count} ({count/len(analyzed_samples)*100:.1f}%)")

        # 按照阶段1的配比选择样本
        n_high_quality = int(n_samples * 0.9)
        n_not_retrievable = n_samples - n_high_quality

        # 选择高质量可检索样本
        high_quality_samples = [s for s in analyzed_samples if s['retrieval_quality'] >= 0.7]
        if len(high_quality_samples) < n_high_quality:
            print(f"\n警告: 高质量样本不足! 需要 {n_high_quality}, 实际 {len(high_quality_samples)}")
            print("将降低质量阈值...")
            high_quality_samples = [s for s in analyzed_samples if s['retrieval_quality'] >= 0.5]

        selected_high_quality = random.sample(high_quality_samples, min(n_high_quality, len(high_quality_samples)))
        samples.extend(selected_high_quality)

        # 选择不可检索样本
        not_retrievable_samples = [s for s in analyzed_samples if s['retrieval_quality'] < 0.2]

        # 如果不可检索样本不够，创造一些
        if len(not_retrievable_samples) < n_not_retrievable:
            print(f"\n创造额外的不可检索样本...")
            needed = n_not_retrievable - len(not_retrievable_samples)
            created_samples = self._create_not_retrievable_samples(needed)
            not_retrievable_samples.extend(created_samples)

        selected_not_retrievable = random.sample(not_retrievable_samples, n_not_retrievable)
        samples.extend(selected_not_retrievable)

        # 打乱顺序
        random.shuffle(samples)

        print(f"\n✓ 阶段1样本准备完成: {len(samples)} 个")
        return samples

    def prepare_phase2_samples(self, n_samples: int, existing_retrieval: bool = True) -> List[Dict]:
        """
        准备阶段2样本：60% 可检索 + 40% 不可检索
        """
        print(f"\n=== 准备阶段2样本 ===")
        print(f"目标: {n_samples} 个样本")
        print(f"配比: 60% 可检索 + 40% 不可检索")

        # 类似阶段1的逻辑，但调整配比
        samples = []

        # 分析样本
        analyzed_samples = []
        for idx, item in enumerate(self.base_dataset):
            if idx % 100 == 0:
                print(f"  已分析 {idx}/{len(self.base_dataset)} 个样本...")

            question = item.get('question', '')
            golden_answer = item.get('answer', item.get('answers', ''))

            if existing_retrieval and 'docs' in item:
                docs = item['docs']
            elif self.retrieval_system:
                docs = self.retrieval_system.search(question, top_k=3)
            else:
                continue

            quality = self.analyzer.compute_retrieval_quality(question, docs, golden_answer)
            sample_type, expected_behavior = self.analyzer.classify_sample_type(quality)

            analyzed_samples.append({
                'original': item,
                'question': question,
                'golden_answer': golden_answer,
                'docs': docs,
                'retrieval_quality': quality,
                'sample_type': sample_type,
                'expected_behavior': expected_behavior
            })

        # 60% 可检索（混合高、中质量）
        n_retrievable = int(n_samples * 0.6)
        n_not_retrievable = n_samples - n_retrievable

        retrievable_samples = [s for s in analyzed_samples if s['retrieval_quality'] >= 0.4]
        selected_retrievable = random.sample(retrievable_samples, min(n_retrievable, len(retrievable_samples)))
        samples.extend(selected_retrievable)

        # 40% 不可检索
        not_retrievable_samples = [s for s in analyzed_samples if s['retrieval_quality'] < 0.2]
        if len(not_retrievable_samples) < n_not_retrievable:
            created = self._create_not_retrievable_samples(n_not_retrievable - len(not_retrievable_samples))
            not_retrievable_samples.extend(created)

        selected_not_retrievable = random.sample(not_retrievable_samples, n_not_retrievable)
        samples.extend(selected_not_retrievable)

        random.shuffle(samples)

        print(f"\n✓ 阶段2样本准备完成: {len(samples)} 个")
        return samples

    def prepare_phase3_samples(self, n_samples: int, existing_retrieval: bool = True) -> List[Dict]:
        """
        准备阶段3样本：50% 可检索 + 30% 不可检索 + 20% 边界情况
        """
        print(f"\n=== 准备阶段3样本 ===")
        print(f"目标: {n_samples} 个样本")
        print(f"配比: 50% 可检索 + 30% 不可检索 + 20% 边界")

        samples = []

        # 分析样本
        analyzed_samples = []
        for idx, item in enumerate(self.base_dataset):
            if idx % 100 == 0:
                print(f"  已分析 {idx}/{len(self.base_dataset)} 个样本...")

            question = item.get('question', '')
            golden_answer = item.get('answer', item.get('answers', ''))

            if existing_retrieval and 'docs' in item:
                docs = item['docs']
            elif self.retrieval_system:
                docs = self.retrieval_system.search(question, top_k=3)
            else:
                continue

            quality = self.analyzer.compute_retrieval_quality(question, docs, golden_answer)
            sample_type, expected_behavior = self.analyzer.classify_sample_type(quality)

            analyzed_samples.append({
                'original': item,
                'question': question,
                'golden_answer': golden_answer,
                'docs': docs,
                'retrieval_quality': quality,
                'sample_type': sample_type,
                'expected_behavior': expected_behavior
            })

        # 50% 可检索
        n_retrievable = int(n_samples * 0.5)
        n_not_retrievable = int(n_samples * 0.3)
        n_boundary = n_samples - n_retrievable - n_not_retrievable

        retrievable_samples = [s for s in analyzed_samples if s['retrieval_quality'] >= 0.4]
        samples.extend(random.sample(retrievable_samples, min(n_retrievable, len(retrievable_samples))))

        # 30% 不可检索
        not_retrievable_samples = [s for s in analyzed_samples if s['retrieval_quality'] < 0.2]
        if len(not_retrievable_samples) < n_not_retrievable:
            created = self._create_not_retrievable_samples(n_not_retrievable - len(not_retrievable_samples))
            not_retrievable_samples.extend(created)
        samples.extend(random.sample(not_retrievable_samples, n_not_retrievable))

        # 20% 边界情况（0.2 <= quality < 0.4）
        boundary_samples = [s for s in analyzed_samples if 0.2 <= s['retrieval_quality'] < 0.4]
        if boundary_samples:
            samples.extend(random.sample(boundary_samples, min(n_boundary, len(boundary_samples))))

        random.shuffle(samples)

        print(f"\n✓ 阶段3样本准备完成: {len(samples)} 个")
        return samples

    def _create_not_retrievable_samples(self, n: int) -> List[Dict]:
        """
        创造"检索不到答案"的样本

        策略：从现有样本中随机选取，但替换检索文档为不相关的
        """
        created = []

        for _ in range(n):
            # 随机选一个样本
            base = random.choice(self.base_dataset)

            # 随机选另外一个样本的文档（作为不相关文档）
            other = random.choice(self.base_dataset)

            created.append({
                'question': base.get('question', ''),
                'golden_answer': base.get('answer', base.get('answers', '')),
                'docs': other.get('docs', []),  # 故意用不相关的文档
                'retrieval_quality': 0.0,
                'sample_type': 'not_retrievable',
                'expected_behavior': 'admit_unknown',
                'metadata': {'created': True, 'reason': 'irrelevant_docs'}
            })

        return created

    def save_samples(self, samples: List[Dict], output_path: str):
        """保存样本"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\n✓ 已保存 {len(samples)} 个样本到 {output_path}")

        # 打印统计信息
        type_counts = defaultdict(int)
        quality_scores = []

        for s in samples:
            type_counts[s['sample_type']] += 1
            quality_scores.append(s['retrieval_quality'])

        print("\n样本统计:")
        for stype, count in sorted(type_counts.items()):
            print(f"  {stype}: {count} ({count/len(samples)*100:.1f}%)")

        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\n平均检索质量: {avg_quality:.3f}")


def main():
    """
    主函数

    使用方法：
    1. 准备你的数据集（jsonl格式，每行一个JSON对象）
    2. 数据格式应该包含: question, answer (或answers), docs (可选)
    3. 如果没有docs，需要提供检索系统
    """

    # ========== 配置 ==========
    BASE_DATASET_PATH = "path/to/your/train_dataset.jsonl"  # 修改为你的数据集路径
    OUTPUT_DIR = Path("./prepared_samples")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ========== 初始化 ==========
    print("初始化样本准备器...")
    preparer = SamplePreparer(
        base_dataset_path=BASE_DATASET_PATH,
        retrieval_system=None  # 如果你的数据集中已经有docs，设为None
    )

    # ========== 阶段1 ==========
    phase1_samples = preparer.prepare_phase1_samples(
        n_samples=1000,
        existing_retrieval=True  # 如果数据集中已有检索结果，设为True
    )
    preparer.save_samples(phase1_samples, OUTPUT_DIR / "phase1_train.jsonl")

    # ========== 阶段2 ==========
    phase2_samples = preparer.prepare_phase2_samples(
        n_samples=1500,
        existing_retrieval=True
    )
    preparer.save_samples(phase2_samples, OUTPUT_DIR / "phase2_train.jsonl")

    # ========== 阶段3 ==========
    phase3_samples = preparer.prepare_phase3_samples(
        n_samples=2000,
        existing_retrieval=True
    )
    preparer.save_samples(phase3_samples, OUTPUT_DIR / "phase3_train.jsonl")

    print("\n" + "="*70)
    print("所有阶段样本准备完成!")
    print("="*70)
    print("\n训练建议:")
    print("1. Step 200-400: 使用 phase1_train.jsonl (90% 高质量 + 10% 不可检索)")
    print("2. Step 400-700: 使用 phase2_train.jsonl (60% 可检索 + 40% 不可检索)")
    print("3. Step 700+:    使用 phase3_train.jsonl (50% + 30% + 20% 边界)")


if __name__ == "__main__":
    main()
