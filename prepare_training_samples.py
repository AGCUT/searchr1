#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练样本准备工具
用于生成"学会承认检索失败"的训练数据

分三个阶段准备不同配比的样本：
- 阶段1 (200-400 steps): 90% 可检索 + 10% 不可检索
- 阶段2 (400-700 steps): 60% 可检索 + 40% 不可检索
- 阶段3 (700+ steps): 50% 可检索 + 30% 不可检索 + 20% 边界情况
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import re


@dataclass
class TrainingSample:
    """训练样本数据结构"""
    question: str
    golden_answer: str
    sample_type: str  # "retrievable", "not_retrievable", "boundary"
    docs: List[Dict[str, str]]  # 模拟的检索文档
    expected_behavior: str  # "answer_correctly" or "admit_unknown"
    retrieval_quality_score: float  # 0-1, 检索质量评分
    metadata: Dict[str, Any] = None


class SampleGenerator:
    """训练样本生成器"""

    def __init__(self, base_dataset_path: str, knowledge_base_path: str = None):
        """
        Args:
            base_dataset_path: 原始问答数据集路径 (如 NQ, HotpotQA)
            knowledge_base_path: 知识库路径 (用于模拟检索)
        """
        self.base_dataset = self._load_dataset(base_dataset_path)
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path) if knowledge_base_path else None

    def _load_dataset(self, path: str) -> List[Dict]:
        """加载基础数据集"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_knowledge_base(self, path: str) -> Dict:
        """加载知识库"""
        # 这里根据你的知识库格式调整
        # 假设是文档ID到文档内容的映射
        kb = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                kb[doc['id']] = doc
        return kb

    def generate_phase1_samples(self, n_samples: int) -> List[TrainingSample]:
        """
        阶段1样本：90% 可检索 + 10% 不可检索
        重点：修复推理能力
        """
        samples = []
        n_retrievable = int(n_samples * 0.9)
        n_not_retrievable = n_samples - n_retrievable

        # 生成可检索样本（明确包含答案）
        for _ in range(n_retrievable):
            sample = self._create_retrievable_sample(clarity="high")
            samples.append(sample)

        # 生成不可检索样本（让模型开始接触"不知道"场景）
        for _ in range(n_not_retrievable):
            sample = self._create_not_retrievable_sample()
            samples.append(sample)

        random.shuffle(samples)
        return samples

    def generate_phase2_samples(self, n_samples: int) -> List[TrainingSample]:
        """
        阶段2样本：60% 可检索 + 40% 不可检索
        重点：学习识别检索失败
        """
        samples = []
        n_retrievable = int(n_samples * 0.6)
        n_not_retrievable = n_samples - n_retrievable

        for _ in range(n_retrievable):
            # 混合高清晰度和中等清晰度的可检索样本
            clarity = random.choice(["high", "medium"])
            sample = self._create_retrievable_sample(clarity=clarity)
            samples.append(sample)

        for _ in range(n_not_retrievable):
            sample = self._create_not_retrievable_sample()
            samples.append(sample)

        random.shuffle(samples)
        return samples

    def generate_phase3_samples(self, n_samples: int) -> List[TrainingSample]:
        """
        阶段3样本：50% 可检索 + 30% 不可检索 + 20% 边界情况
        重点：平衡与泛化
        """
        samples = []
        n_retrievable = int(n_samples * 0.5)
        n_not_retrievable = int(n_samples * 0.3)
        n_boundary = n_samples - n_retrievable - n_not_retrievable

        for _ in range(n_retrievable):
            clarity = random.choice(["high", "medium", "low"])
            sample = self._create_retrievable_sample(clarity=clarity)
            samples.append(sample)

        for _ in range(n_not_retrievable):
            sample = self._create_not_retrievable_sample()
            samples.append(sample)

        for _ in range(n_boundary):
            sample = self._create_boundary_sample()
            samples.append(sample)

        random.shuffle(samples)
        return samples

    def _create_retrievable_sample(self, clarity: str = "high") -> TrainingSample:
        """
        创建"可检索到答案"的样本

        Args:
            clarity: 答案清晰度
                - "high": 文档中明确包含答案
                - "medium": 文档中包含但需要简单推理
                - "low": 文档中隐含答案，需要复杂推理
        """
        # 从基础数据集中随机选一个样本
        base_sample = random.choice(self.base_dataset)

        question = base_sample['question']
        golden_answer = base_sample.get('answer', base_sample.get('answers', [''])[0])

        # 根据clarity构造不同质量的检索文档
        if clarity == "high":
            # 文档明确包含答案
            docs = self._create_docs_with_clear_answer(question, golden_answer)
            retrieval_quality = random.uniform(0.8, 1.0)
        elif clarity == "medium":
            # 文档包含答案但需要简单推理
            docs = self._create_docs_with_implicit_answer(question, golden_answer)
            retrieval_quality = random.uniform(0.5, 0.8)
        else:  # low
            # 文档隐含答案，需要复杂推理
            docs = self._create_docs_with_hidden_answer(question, golden_answer)
            retrieval_quality = random.uniform(0.3, 0.5)

        return TrainingSample(
            question=question,
            golden_answer=golden_answer,
            sample_type="retrievable",
            docs=docs,
            expected_behavior="answer_correctly",
            retrieval_quality_score=retrieval_quality,
            metadata={"clarity": clarity}
        )

    def _create_not_retrievable_sample(self) -> TrainingSample:
        """
        创建"检索不到答案"的样本

        策略：
        1. 使用时效性问题（如"2024年世界杯"）
        2. 使用不在知识库范围内的问题
        3. 故意返回不相关的文档
        """
        # 方法1：使用需要最新信息的问题
        temporal_questions = [
            "What is the current world record for 100m sprint?",
            "Who won the 2024 Nobel Prize in Physics?",
            "What is the latest version of Python?",
            "When will the next solar eclipse occur?",
        ]

        # 方法2：使用超出知识库范围的问题
        out_of_scope_questions = [
            "What is the weather in Beijing today?",
            "How many people are currently online on Facebook?",
            "What stocks should I buy tomorrow?",
        ]

        # 方法3：从base_dataset中选取，但返回不相关文档
        base_sample = random.choice(self.base_dataset)
        question = base_sample['question']
        golden_answer = base_sample.get('answer', base_sample.get('answers', [''])[0])

        # 构造不相关的检索文档
        docs = self._create_irrelevant_docs(question)

        return TrainingSample(
            question=question,
            golden_answer=golden_answer,
            sample_type="not_retrievable",
            docs=docs,
            expected_behavior="admit_unknown",
            retrieval_quality_score=random.uniform(0.0, 0.2),
            metadata={"reason": "irrelevant_docs"}
        )

    def _create_boundary_sample(self) -> TrainingSample:
        """
        创建边界情况样本

        包括：
        1. 文档部分相关但信息不完整
        2. 文档有矛盾信息
        3. 问题本身模糊不清
        """
        base_sample = random.choice(self.base_dataset)
        question = base_sample['question']
        golden_answer = base_sample.get('answer', base_sample.get('answers', [''])[0])

        boundary_type = random.choice(["partial", "contradictory", "ambiguous"])

        if boundary_type == "partial":
            # 部分相关但不完整
            docs = self._create_partial_docs(question, golden_answer)
            expected = "answer_with_caution"  # 谨慎回答+承认信息不完整
            quality = random.uniform(0.3, 0.5)
        elif boundary_type == "contradictory":
            # 文档间有矛盾
            docs = self._create_contradictory_docs(question, golden_answer)
            expected = "acknowledge_contradiction"
            quality = random.uniform(0.2, 0.4)
        else:  # ambiguous
            # 问题模糊
            docs = self._create_docs_for_ambiguous_question(question, golden_answer)
            expected = "clarify_question"
            quality = random.uniform(0.2, 0.5)

        return TrainingSample(
            question=question,
            golden_answer=golden_answer,
            sample_type="boundary",
            docs=docs,
            expected_behavior=expected,
            retrieval_quality_score=quality,
            metadata={"boundary_type": boundary_type}
        )

    def _create_docs_with_clear_answer(self, question: str, answer: str) -> List[Dict[str, str]]:
        """
        创建明确包含答案的文档

        这是最重要的方法！用于修复你案例中的推理错误问题
        """
        # 如果有真实的知识库，从中检索
        if self.knowledge_base:
            return self._retrieve_from_kb(question, answer, n_docs=3)

        # 否则，构造包含答案的模拟文档
        docs = []

        # Doc 1: 直接包含答案的文档
        doc1 = {
            "title": self._generate_relevant_title(question),
            "text": self._generate_text_with_answer(question, answer, style="direct")
        }
        docs.append(doc1)

        # Doc 2: 用不同表述包含答案
        doc2 = {
            "title": self._generate_relevant_title(question),
            "text": self._generate_text_with_answer(question, answer, style="paraphrase")
        }
        docs.append(doc2)

        # Doc 3: 相关但不直接包含答案（增加一些噪音）
        doc3 = {
            "title": self._generate_relevant_title(question),
            "text": self._generate_related_text(question)
        }
        docs.append(doc3)

        return docs

    def _create_docs_with_implicit_answer(self, question: str, answer: str) -> List[Dict[str, str]]:
        """创建隐含答案的文档（需要简单推理）"""
        # 例如：问"总统是谁"，文档说"选举在X年，Y赢得选举"
        # 需要推理：Y赢得选举 → Y是总统
        docs = []

        doc = {
            "title": self._generate_relevant_title(question),
            "text": self._generate_text_requiring_inference(question, answer)
        }
        docs.append(doc)

        # 添加一些相关文档
        for _ in range(2):
            docs.append({
                "title": self._generate_relevant_title(question),
                "text": self._generate_related_text(question)
            })

        return docs

    def _create_docs_with_hidden_answer(self, question: str, answer: str) -> List[Dict[str, str]]:
        """创建答案隐藏很深的文档（需要复杂推理）"""
        # 答案散落在多个文档中，需要综合推理
        docs = []

        # 将答案拆分到多个文档
        partial_info = self._split_answer_into_parts(question, answer)
        for info in partial_info:
            docs.append({
                "title": self._generate_relevant_title(question),
                "text": info
            })

        return docs

    def _create_irrelevant_docs(self, question: str) -> List[Dict[str, str]]:
        """创建完全不相关的文档"""
        # 故意返回与问题无关的文档
        irrelevant_topics = [
            "The history of ancient Rome",
            "How to bake a chocolate cake",
            "The biology of dolphins",
            "Introduction to quantum mechanics",
            "Tips for growing tomatoes"
        ]

        docs = []
        for _ in range(3):
            topic = random.choice(irrelevant_topics)
            docs.append({
                "title": topic,
                "text": f"This document discusses {topic}. " + self._generate_filler_text()
            })

        return docs

    def _create_partial_docs(self, question: str, answer: str) -> List[Dict[str, str]]:
        """创建部分相关的文档"""
        docs = []

        # 提供部分信息，但不完整
        doc = {
            "title": self._generate_relevant_title(question),
            "text": self._generate_incomplete_info(question, answer)
        }
        docs.append(doc)

        # 添加相关但不够的信息
        for _ in range(2):
            docs.append({
                "title": self._generate_relevant_title(question),
                "text": self._generate_tangentially_related_text(question)
            })

        return docs

    def _create_contradictory_docs(self, question: str, answer: str) -> List[Dict[str, str]]:
        """创建包含矛盾信息的文档"""
        docs = []

        # Doc 1: 包含正确答案
        docs.append({
            "title": self._generate_relevant_title(question),
            "text": self._generate_text_with_answer(question, answer, style="direct")
        })

        # Doc 2: 包含矛盾的答案
        wrong_answer = self._generate_plausible_wrong_answer(answer)
        docs.append({
            "title": self._generate_relevant_title(question),
            "text": self._generate_text_with_answer(question, wrong_answer, style="direct")
        })

        return docs

    def _create_docs_for_ambiguous_question(self, question: str, answer: str) -> List[Dict[str, str]]:
        """为模糊问题创建文档"""
        # 返回多种可能的解释
        docs = []

        interpretations = self._generate_question_interpretations(question)
        for interp in interpretations[:3]:
            docs.append({
                "title": f"About {interp}",
                "text": self._generate_related_text(interp)
            })

        return docs

    # ========== 辅助方法：文本生成 ==========

    def _generate_relevant_title(self, question: str) -> str:
        """从问题生成相关的标题"""
        # 简单实现：提取问题中的关键词
        words = question.split()
        keywords = [w for w in words if len(w) > 4 and w.lower() not in ['what', 'when', 'where', 'which', 'who']]
        return " ".join(keywords[:3]) if keywords else "Related Document"

    def _generate_text_with_answer(self, question: str, answer: str, style: str = "direct") -> str:
        """生成包含答案的文本"""
        if style == "direct":
            # 直接陈述答案
            templates = [
                f"The answer to '{question}' is {answer}.",
                f"According to records, {answer} is the correct answer.",
                f"It is well known that {answer}.",
            ]
        else:  # paraphrase
            # 用不同方式表达
            templates = [
                f"Historical documents indicate that {answer}.",
                f"Research shows that {answer}.",
                f"The evidence points to {answer}.",
            ]

        text = random.choice(templates)
        text += " " + self._generate_filler_text()
        return text

    def _generate_related_text(self, question: str) -> str:
        """生成相关但不含答案的文本"""
        return f"This document discusses topics related to {question}. " + self._generate_filler_text()

    def _generate_text_requiring_inference(self, question: str, answer: str) -> str:
        """生成需要推理才能得出答案的文本"""
        # 这需要根据具体问题类型设计
        # 简单示例：
        return f"In the context of {question}, multiple sources suggest that {answer} plays a key role. " + self._generate_filler_text()

    def _split_answer_into_parts(self, question: str, answer: str) -> List[str]:
        """将答案拆分成多个部分"""
        # 简单实现
        parts = answer.split()
        return [
            f"Part of the information about {question}: {' '.join(parts[:len(parts)//2])}",
            f"Additional details: {' '.join(parts[len(parts)//2:])}",
        ]

    def _generate_incomplete_info(self, question: str, answer: str) -> str:
        """生成不完整的信息"""
        return f"Some information about {question} is available, but details are limited. " + self._generate_filler_text()

    def _generate_tangentially_related_text(self, question: str) -> str:
        """生成边缘相关的文本"""
        return f"While not directly answering {question}, this provides context. " + self._generate_filler_text()

    def _generate_plausible_wrong_answer(self, correct_answer: str) -> str:
        """生成看起来合理的错误答案"""
        # 简单实现：修改日期、数字等
        if re.search(r'\d{4}', correct_answer):  # 年份
            year = re.search(r'\d{4}', correct_answer).group()
            wrong_year = str(int(year) + random.choice([-1, 1, -2, 2]))
            return correct_answer.replace(year, wrong_year)
        return "alternative answer"

    def _generate_question_interpretations(self, question: str) -> List[str]:
        """生成问题的多种解释"""
        return [question, question + " (interpretation 1)", question + " (interpretation 2)"]

    def _generate_filler_text(self) -> str:
        """生成填充文本"""
        fillers = [
            "Additional context and background information is provided here.",
            "This has been documented in various sources.",
            "Multiple perspectives exist on this topic.",
            "Further research may be needed to fully understand this.",
        ]
        return random.choice(fillers)

    def _retrieve_from_kb(self, question: str, answer: str, n_docs: int = 3) -> List[Dict[str, str]]:
        """从真实知识库中检索"""
        # 这里应该实现真实的检索逻辑
        # 可以使用BM25、向量检索等
        # 暂时返回空，需要根据你的知识库实现
        return []

    def save_samples(self, samples: List[TrainingSample], output_path: str):
        """保存样本到文件"""
        output_data = []
        for sample in samples:
            output_data.append({
                "question": sample.question,
                "golden_answer": sample.golden_answer,
                "sample_type": sample.sample_type,
                "docs": sample.docs,
                "expected_behavior": sample.expected_behavior,
                "retrieval_quality_score": sample.retrieval_quality_score,
                "metadata": sample.metadata
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✓ 已保存 {len(samples)} 个样本到 {output_path}")


def main():
    """主函数：生成分阶段的训练样本"""

    # 配置路径
    base_dataset_path = "path/to/your/nq_hotpotqa_dataset.jsonl"
    knowledge_base_path = "path/to/your/knowledge_base.jsonl"  # 可选
    output_dir = Path("./training_samples")
    output_dir.mkdir(exist_ok=True)

    # 初始化生成器
    generator = SampleGenerator(base_dataset_path, knowledge_base_path)

    # 阶段1：修复推理能力 (200-400 steps)
    print("生成阶段1样本 (90% 可检索 + 10% 不可检索)...")
    phase1_samples = generator.generate_phase1_samples(n_samples=1000)
    generator.save_samples(phase1_samples, output_dir / "phase1_train.jsonl")

    # 阶段2：引入失败识别 (400-700 steps)
    print("\n生成阶段2样本 (60% 可检索 + 40% 不可检索)...")
    phase2_samples = generator.generate_phase2_samples(n_samples=1500)
    generator.save_samples(phase2_samples, output_dir / "phase2_train.jsonl")

    # 阶段3：平衡与泛化 (700+ steps)
    print("\n生成阶段3样本 (50% 可检索 + 30% 不可检索 + 20% 边界)...")
    phase3_samples = generator.generate_phase3_samples(n_samples=2000)
    generator.save_samples(phase3_samples, output_dir / "phase3_train.jsonl")

    print("\n" + "="*70)
    print("样本生成完成!")
    print("="*70)
    print(f"阶段1: {len(phase1_samples)} 样本 -> {output_dir / 'phase1_train.jsonl'}")
    print(f"阶段2: {len(phase2_samples)} 样本 -> {output_dir / 'phase2_train.jsonl'}")
    print(f"阶段3: {len(phase3_samples)} 样本 -> {output_dir / 'phase3_train.jsonl'}")
    print("\n使用建议:")
    print("1. Step 200-400: 使用 phase1_train.jsonl")
    print("2. Step 400-700: 使用 phase2_train.jsonl")
    print("3. Step 700+:    使用 phase3_train.jsonl")


if __name__ == "__main__":
    main()
