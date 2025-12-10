#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多跳问题训练样本准备工具

处理需要多次检索才能回答的问题，核心思路：
1. 跟踪整个检索链的质量
2. 区分"检索链完整"vs"检索链断裂"
3. 识别"何时应该继续检索"vs"何时应该放弃"

样本类型：
- complete_chain: 多次检索后能得到完整答案
- partial_chain: 部分信息可得，但链条不完整
- broken_chain: 检索链断裂，应该承认不知道
- single_hop: 单次检索就能回答（简单问题）
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SearchStep:
    """单次检索步骤"""
    query: str
    docs: List[Dict[str, str]]
    quality_score: float  # 这一步的检索质量
    info_gained: str  # 这一步获得的关键信息
    is_useful: bool  # 这一步是否有用


@dataclass
class MultiHopSample:
    """多跳问题样本"""
    question: str
    golden_answer: str
    search_chain: List[SearchStep]  # 检索链
    chain_type: str  # complete_chain / partial_chain / broken_chain / single_hop
    expected_behavior: str
    max_useful_hops: int  # 最多有用的检索次数
    final_quality: float  # 最终综合质量
    reasoning_path: str  # 期望的推理路径
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiHopQualityAnalyzer:
    """多跳检索质量分析器"""

    def __init__(self):
        pass

    def analyze_search_chain(
        self,
        question: str,
        search_steps: List[Dict],
        golden_answer: str
    ) -> Tuple[str, float, int]:
        """
        分析整个检索链的质量

        Returns:
            (chain_type, final_quality, max_useful_hops)
        """
        if not search_steps:
            return "broken_chain", 0.0, 0

        # 分析每一步的质量
        step_qualities = []
        cumulative_info = []
        useful_hop_count = 0

        for i, step in enumerate(search_steps):
            docs = step.get('docs', [])
            query = step.get('query', '')

            # 计算这一步的质量
            step_quality = self._compute_step_quality(
                query, docs, golden_answer, cumulative_info
            )
            step_qualities.append(step_quality)

            # 判断这一步是否提供了新信息
            new_info = self._extract_new_info(docs, cumulative_info)
            if new_info:
                cumulative_info.extend(new_info)
                useful_hop_count += 1

        # 判断检索链类型
        chain_type, final_quality = self._classify_chain(
            step_qualities, cumulative_info, golden_answer
        )

        return chain_type, final_quality, useful_hop_count

    def _compute_step_quality(
        self,
        query: str,
        docs: List[Dict],
        golden_answer: str,
        previous_info: List[str]
    ) -> float:
        """计算单步检索质量"""
        if not docs:
            return 0.0

        max_score = 0.0

        for doc in docs:
            doc_text = doc.get('text', '') + ' ' + doc.get('title', '')
            doc_text_lower = doc_text.lower()

            score = 0.0

            # 检查是否包含答案
            answer_str = str(golden_answer).lower() if not isinstance(golden_answer, list) else golden_answer[0].lower()
            if answer_str in doc_text_lower:
                score = 1.0
            else:
                # 检查关键词匹配
                keywords = re.findall(r'\b\w+\b', answer_str)
                keywords = [k for k in keywords if len(k) > 2]
                if keywords:
                    matched = sum(1 for k in keywords if k in doc_text_lower)
                    score = max(score, matched / len(keywords) * 0.7)

            # 检查是否提供了新信息（相比之前的检索）
            if previous_info:
                new_info_bonus = 0.0
                for prev in previous_info:
                    if prev.lower() not in doc_text_lower:
                        new_info_bonus += 0.1
                score = min(1.0, score + new_info_bonus)

            max_score = max(max_score, score)

        return max_score

    def _extract_new_info(
        self,
        docs: List[Dict],
        previous_info: List[str]
    ) -> List[str]:
        """从文档中提取新信息"""
        new_info = []

        for doc in docs:
            doc_text = doc.get('text', '')

            # 提取关键实体（简化版本）
            # 日期
            dates = re.findall(r'\b\d{4}\b|\b\d{1,2}\s+\w+\s+\d{4}\b', doc_text)
            for date in dates:
                if date not in previous_info:
                    new_info.append(date)

            # 数字
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', doc_text)
            for num in numbers[:5]:  # 限制数量
                if num not in previous_info:
                    new_info.append(num)

            # 可以扩展：使用NER提取人名、地名等

        return new_info[:10]  # 限制返回数量

    def _classify_chain(
        self,
        step_qualities: List[float],
        cumulative_info: List[str],
        golden_answer: str
    ) -> Tuple[str, float]:
        """
        分类检索链类型

        Returns:
            (chain_type, final_quality)
        """
        if not step_qualities:
            return "broken_chain", 0.0

        # 计算最终质量（加权平均，后面的步骤更重要）
        weights = [1.0 + 0.5 * i for i in range(len(step_qualities))]
        weighted_quality = sum(q * w for q, w in zip(step_qualities, weights)) / sum(weights)

        # 检查最后一步是否包含答案
        last_quality = step_qualities[-1]
        max_quality = max(step_qualities)

        # 分类
        if max_quality >= 0.8:
            # 某一步检索到了完整答案
            return "complete_chain", max_quality
        elif weighted_quality >= 0.5:
            # 综合信息可以推导出答案
            return "complete_chain", weighted_quality
        elif weighted_quality >= 0.3:
            # 部分信息可得
            return "partial_chain", weighted_quality
        else:
            # 检索链断裂
            return "broken_chain", weighted_quality


class MultiHopSamplePreparer:
    """多跳样本准备器"""

    def __init__(self, dataset_path: str):
        self.dataset = self._load_dataset(dataset_path)
        self.analyzer = MultiHopQualityAnalyzer()

    def _load_dataset(self, path: str) -> List[Dict]:
        """加载数据集"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def analyze_and_prepare(self) -> Dict[str, List[Dict]]:
        """
        分析数据集并准备分类样本

        Returns:
            按类型分组的样本
        """
        categorized = {
            "complete_chain": [],
            "partial_chain": [],
            "broken_chain": [],
            "single_hop": []
        }

        for item in self.dataset:
            question = item.get('question', '')
            golden_answer = item.get('answer', item.get('answers', ''))

            # 获取检索历史
            # 假设数据格式包含多次检索的结果
            # 格式可能是: search_history: [{query: ..., docs: ...}, ...]
            # 或者: docs: [...] (单次检索)

            search_history = item.get('search_history', [])

            if not search_history:
                # 单次检索的情况
                docs = item.get('docs', [])
                if docs:
                    search_history = [{'query': question, 'docs': docs}]

            # 分析检索链
            chain_type, final_quality, useful_hops = self.analyzer.analyze_search_chain(
                question, search_history, golden_answer
            )

            # 确定期望行为
            expected_behavior = self._determine_expected_behavior(
                chain_type, final_quality, useful_hops, len(search_history)
            )

            sample = {
                'question': question,
                'golden_answer': golden_answer,
                'search_history': search_history,
                'chain_type': chain_type,
                'final_quality': final_quality,
                'useful_hops': useful_hops,
                'total_hops': len(search_history),
                'expected_behavior': expected_behavior,
                'original': item
            }

            categorized[chain_type].append(sample)

        return categorized

    def _determine_expected_behavior(
        self,
        chain_type: str,
        quality: float,
        useful_hops: int,
        total_hops: int
    ) -> str:
        """确定期望行为"""

        if chain_type == "complete_chain":
            return "answer_correctly"

        elif chain_type == "partial_chain":
            if useful_hops >= 1:
                return "answer_with_caution"  # 提供部分答案并说明不确定性
            else:
                return "request_more_search"  # 应该继续搜索

        elif chain_type == "broken_chain":
            if total_hops >= 3:
                return "admit_unknown"  # 已经搜了3次还找不到，应该承认不知道
            else:
                return "request_more_search"  # 可能还需要更多搜索

        else:  # single_hop
            if quality >= 0.7:
                return "answer_correctly"
            elif quality >= 0.4:
                return "answer_with_caution"
            else:
                return "admit_unknown"

    def prepare_phase_samples(
        self,
        phase: int,
        n_samples: int
    ) -> List[Dict]:
        """
        根据训练阶段准备样本

        Phase 1 (Step 200-400): 重点修复推理，90%完整链 + 10%断裂链
        Phase 2 (Step 400-700): 学习识别失败，60%完整 + 40%断裂/部分
        Phase 3 (Step 700+): 平衡泛化，50%完整 + 30%断裂 + 20%部分
        """
        categorized = self.analyze_and_prepare()

        # 打印统计
        print(f"\n数据集分析结果:")
        for chain_type, samples in categorized.items():
            print(f"  {chain_type}: {len(samples)} 样本")

        samples = []

        if phase == 1:
            # 90% 完整链 + 10% 断裂链
            n_complete = int(n_samples * 0.9)
            n_broken = n_samples - n_complete

            complete = categorized['complete_chain'] + categorized.get('single_hop', [])
            samples.extend(self._safe_sample(complete, n_complete))
            samples.extend(self._safe_sample(categorized['broken_chain'], n_broken))

        elif phase == 2:
            # 60% 完整 + 25% 断裂 + 15% 部分
            n_complete = int(n_samples * 0.6)
            n_broken = int(n_samples * 0.25)
            n_partial = n_samples - n_complete - n_broken

            complete = categorized['complete_chain'] + categorized.get('single_hop', [])
            samples.extend(self._safe_sample(complete, n_complete))
            samples.extend(self._safe_sample(categorized['broken_chain'], n_broken))
            samples.extend(self._safe_sample(categorized['partial_chain'], n_partial))

        else:  # phase 3
            # 50% 完整 + 30% 断裂 + 20% 部分
            n_complete = int(n_samples * 0.5)
            n_broken = int(n_samples * 0.3)
            n_partial = n_samples - n_complete - n_broken

            complete = categorized['complete_chain'] + categorized.get('single_hop', [])
            samples.extend(self._safe_sample(complete, n_complete))
            samples.extend(self._safe_sample(categorized['broken_chain'], n_broken))
            samples.extend(self._safe_sample(categorized['partial_chain'], n_partial))

        random.shuffle(samples)
        return samples

    def _safe_sample(self, data: List, n: int) -> List:
        """安全采样，处理样本不足的情况"""
        if len(data) == 0:
            return []
        if len(data) < n:
            # 样本不足，重复采样
            return random.choices(data, k=n)
        return random.sample(data, n)

    def save_samples(self, samples: List[Dict], output_path: str):
        """保存样本"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for s in samples:
                # 移除original字段减小文件大小
                save_sample = {k: v for k, v in s.items() if k != 'original'}
                f.write(json.dumps(save_sample, ensure_ascii=False) + '\n')

        print(f"✓ 已保存 {len(samples)} 个样本到 {output_path}")


class MultiHopRewardComputer:
    """
    多跳问题的Reward计算器

    这个类展示了如何设计reward，你可以集成到你的训练代码中
    """

    def __init__(self, max_search_allowed: int = 5):
        self.max_search = max_search_allowed

    def compute_reward(
        self,
        question: str,
        model_response: str,
        search_history: List[Dict],
        golden_answer: str,
        chain_type: str,
        expected_behavior: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算reward

        Returns:
            (total_reward, reward_breakdown)
        """
        rewards = {}

        # 1. 答案正确性奖励
        answer_correct = self._check_answer_correct(model_response, golden_answer)
        rewards['answer_correct'] = 1.0 if answer_correct else 0.0

        # 2. 根据chain_type和expected_behavior调整reward
        if chain_type == "complete_chain":
            # 检索链完整，期望正确回答
            if answer_correct:
                rewards['behavior'] = 1.0
            else:
                # 有答案但答错了，惩罚（这是你的主要问题）
                rewards['behavior'] = -0.5

        elif chain_type == "broken_chain":
            # 检索链断裂
            admits_unknown = self._check_admits_unknown(model_response)

            if admits_unknown:
                # 诚实承认不知道，奖励
                rewards['behavior'] = 0.8
            elif answer_correct:
                # 虽然检索失败但猜对了（可能有风险）
                rewards['behavior'] = 0.3
            else:
                # 检索失败还乱猜，惩罚
                rewards['behavior'] = -0.5

        elif chain_type == "partial_chain":
            # 部分信息可得
            has_caution = self._check_has_caution(model_response)

            if answer_correct and has_caution:
                # 正确且谨慎，最好的情况
                rewards['behavior'] = 1.0
            elif answer_correct:
                # 正确但没有表达不确定性
                rewards['behavior'] = 0.7
            elif has_caution:
                # 答错但表达了不确定性
                rewards['behavior'] = 0.2
            else:
                rewards['behavior'] = -0.3

        # 3. 检索效率奖励/惩罚
        n_searches = len(search_history)

        if n_searches == 0:
            # 没有检索就回答（可能好可能坏）
            if answer_correct:
                rewards['efficiency'] = 0.2  # 小奖励
            else:
                rewards['efficiency'] = -0.1
        elif n_searches <= 2:
            # 1-2次检索，效率好
            rewards['efficiency'] = 0.1
        elif n_searches <= 4:
            # 3-4次检索，可接受
            rewards['efficiency'] = 0.0
        else:
            # 超过4次检索，效率低
            rewards['efficiency'] = -0.1 * (n_searches - 4)

        # 4. 防止过度消极
        refusal_count = self._count_refusals(model_response)
        if refusal_count > 2:
            rewards['anti_lazy'] = -0.2 * (refusal_count - 2)
        else:
            rewards['anti_lazy'] = 0.0

        # 5. 推理质量奖励（检查是否有合理的思考过程）
        has_reasoning = '<think>' in model_response and '</think>' in model_response
        if has_reasoning:
            rewards['reasoning'] = 0.1
        else:
            rewards['reasoning'] = 0.0

        # 计算总reward
        total = sum(rewards.values())

        return total, rewards

    def _check_answer_correct(self, response: str, golden: str) -> bool:
        """检查答案是否正确"""
        # 提取<answer>标签中的内容
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not match:
            return False

        model_answer = match.group(1).strip().lower()

        # 标准化golden answer
        if isinstance(golden, list):
            golden_list = [str(g).strip().lower() for g in golden]
        else:
            golden_list = [str(golden).strip().lower()]

        # 检查是否匹配任一正确答案
        for g in golden_list:
            if g in model_answer or model_answer in g:
                return True
            # 处理日期格式差异
            if self._normalize_date(g) == self._normalize_date(model_answer):
                return True

        return False

    def _normalize_date(self, text: str) -> str:
        """标准化日期格式"""
        # 移除多余空格，统一格式
        text = re.sub(r'\s+', ' ', text.strip())
        # 可以添加更多日期格式标准化逻辑
        return text

    def _check_admits_unknown(self, response: str) -> bool:
        """检查是否承认不知道"""
        unknown_phrases = [
            '不知道', '无法确定', '没有找到', '找不到',
            "don't know", "cannot determine", "not found",
            "unable to find", "no information", "cannot answer",
            '需要更多信息', '信息不足'
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in unknown_phrases)

    def _check_has_caution(self, response: str) -> bool:
        """检查是否表达了谨慎/不确定性"""
        caution_phrases = [
            '可能', '大概', '也许', '不确定', '根据现有信息',
            'possibly', 'maybe', 'might', 'perhaps', 'uncertain',
            'based on available information', 'not completely sure'
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in caution_phrases)

    def _count_refusals(self, response: str) -> int:
        """统计拒绝回答的次数"""
        refusal_phrases = ['不知道', "don't know", '无法', 'cannot']
        count = 0
        response_lower = response.lower()
        for phrase in refusal_phrases:
            count += response_lower.count(phrase)
        return count


def main():
    """主函数"""
    import sys

    # 配置
    DATASET_PATH = "path/to/your/train_data.jsonl"  # 修改为你的数据路径
    OUTPUT_DIR = Path("./multihop_samples")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("多跳问题训练样本准备工具")
    print("="*70)

    # 初始化
    preparer = MultiHopSamplePreparer(DATASET_PATH)

    # 阶段1
    print("\n生成阶段1样本 (修复推理能力)...")
    phase1 = preparer.prepare_phase_samples(phase=1, n_samples=1000)
    preparer.save_samples(phase1, OUTPUT_DIR / "phase1_multihop.jsonl")

    # 阶段2
    print("\n生成阶段2样本 (学习识别失败)...")
    phase2 = preparer.prepare_phase_samples(phase=2, n_samples=1500)
    preparer.save_samples(phase2, OUTPUT_DIR / "phase2_multihop.jsonl")

    # 阶段3
    print("\n生成阶段3样本 (平衡与泛化)...")
    phase3 = preparer.prepare_phase_samples(phase=3, n_samples=2000)
    preparer.save_samples(phase3, OUTPUT_DIR / "phase3_multihop.jsonl")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)

    # 展示reward计算示例
    print("\n\n" + "="*70)
    print("Reward计算示例")
    print("="*70)

    reward_computer = MultiHopRewardComputer()

    # 示例1：完整链且答对
    example1_reward, breakdown1 = reward_computer.compute_reward(
        question="when was the constitution of india adopted?",
        model_response="<think>从检索结果看...</think><answer>26 November 1949</answer>",
        search_history=[{"query": "q1", "docs": [{}]}, {"query": "q2", "docs": [{}]}],
        golden_answer="26 November 1949",
        chain_type="complete_chain",
        expected_behavior="answer_correctly"
    )
    print(f"\n示例1 (完整链+答对): {example1_reward:.2f}")
    print(f"  分解: {breakdown1}")

    # 示例2：完整链但答错（你的案例）
    example2_reward, breakdown2 = reward_computer.compute_reward(
        question="when was the constitution of india adopted?",
        model_response="<think>从检索结果看...</think><answer>11 November 1949</answer>",
        search_history=[{"query": "q1", "docs": [{}]}, {"query": "q2", "docs": [{}]}],
        golden_answer="26 November 1949",
        chain_type="complete_chain",
        expected_behavior="answer_correctly"
    )
    print(f"\n示例2 (完整链+答错): {example2_reward:.2f}")
    print(f"  分解: {breakdown2}")

    # 示例3：断裂链且承认不知道
    example3_reward, breakdown3 = reward_computer.compute_reward(
        question="Who won 2025 Nobel Prize?",
        model_response="<think>检索结果中没有相关信息...</think><answer>抱歉，我无法找到相关信息</answer>",
        search_history=[{"query": "q1", "docs": [{}]}, {"query": "q2", "docs": [{}]}, {"query": "q3", "docs": [{}]}],
        golden_answer="Unknown",
        chain_type="broken_chain",
        expected_behavior="admit_unknown"
    )
    print(f"\n示例3 (断裂链+承认不知道): {example3_reward:.2f}")
    print(f"  分解: {breakdown3}")


if __name__ == "__main__":
    main()
