#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练样本质量检查工具

用于检查准备好的训练样本质量，确保数据分布合理
"""

import json
import random
from typing import List, Dict
from pathlib import Path
from collections import defaultdict, Counter
import re


class SampleQualityChecker:
    """样本质量检查器"""

    def __init__(self, sample_file: str):
        self.samples = self._load_samples(sample_file)
        self.sample_file = sample_file

    def _load_samples(self, path: str) -> List[Dict]:
        """加载样本"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def check_all(self):
        """执行所有检查"""
        print("="*70)
        print(f"样本质量检查报告: {Path(self.sample_file).name}")
        print("="*70)

        self.check_basic_stats()
        self.check_retrieval_quality_distribution()
        self.check_sample_type_distribution()
        self.check_answer_format()
        self.check_potential_issues()
        self.show_random_samples()

    def check_basic_stats(self):
        """基础统计"""
        print("\n【1. 基础统计】")
        print(f"  总样本数: {len(self.samples)}")

        # 问题长度
        question_lengths = [len(s.get('question', '').split()) for s in self.samples]
        print(f"  问题平均长度: {sum(question_lengths)/len(question_lengths):.1f} 词")
        print(f"  问题长度范围: [{min(question_lengths)}, {max(question_lengths)}]")

        # 答案长度
        answer_lengths = []
        for s in self.samples:
            answer = s.get('golden_answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            answer_lengths.append(len(str(answer).split()))

        print(f"  答案平均长度: {sum(answer_lengths)/len(answer_lengths):.1f} 词")

        # 文档数量
        doc_counts = [len(s.get('docs', [])) for s in self.samples]
        if doc_counts:
            print(f"  平均文档数: {sum(doc_counts)/len(doc_counts):.1f}")

    def check_retrieval_quality_distribution(self):
        """检查检索质量分布"""
        print("\n【2. 检索质量分布】")

        qualities = [s.get('retrieval_quality', 0) for s in self.samples]

        print(f"  平均检索质量: {sum(qualities)/len(qualities):.3f}")
        print(f"  质量范围: [{min(qualities):.3f}, {max(qualities):.3f}]")

        # 分桶统计
        buckets = {
            "优秀 (0.8-1.0)": 0,
            "良好 (0.6-0.8)": 0,
            "中等 (0.4-0.6)": 0,
            "较差 (0.2-0.4)": 0,
            "很差 (0.0-0.2)": 0,
        }

        for q in qualities:
            if q >= 0.8:
                buckets["优秀 (0.8-1.0)"] += 1
            elif q >= 0.6:
                buckets["良好 (0.6-0.8)"] += 1
            elif q >= 0.4:
                buckets["中等 (0.4-0.6)"] += 1
            elif q >= 0.2:
                buckets["较差 (0.2-0.4)"] += 1
            else:
                buckets["很差 (0.0-0.2)"] += 1

        print("\n  质量分布:")
        for name, count in buckets.items():
            pct = count / len(qualities) * 100
            bar = "█" * int(pct / 2)
            print(f"    {name:20s} {count:4d} ({pct:5.1f}%) {bar}")

    def check_sample_type_distribution(self):
        """检查样本类型分布"""
        print("\n【3. 样本类型分布】")

        types = [s.get('sample_type', 'unknown') for s in self.samples]
        type_counts = Counter(types)

        print(f"  共 {len(type_counts)} 种类型:\n")
        for stype, count in type_counts.most_common():
            pct = count / len(types) * 100
            bar = "█" * int(pct / 2)
            print(f"    {stype:30s} {count:4d} ({pct:5.1f}%) {bar}")

        # 检查expected_behavior分布
        behaviors = [s.get('expected_behavior', 'unknown') for s in self.samples]
        behavior_counts = Counter(behaviors)

        print("\n  期望行为分布:")
        for behavior, count in behavior_counts.most_common():
            pct = count / len(behaviors) * 100
            print(f"    {behavior:30s} {count:4d} ({pct:5.1f}%)")

    def check_answer_format(self):
        """检查答案格式"""
        print("\n【4. 答案格式检查】")

        answer_types = {
            "日期": 0,
            "数字": 0,
            "人名": 0,
            "地名": 0,
            "短语": 0,
            "句子": 0,
        }

        date_pattern = r'\b\d{4}\b|\b\d{1,2}\s+\w+\s+\d{4}\b'
        number_pattern = r'^\d+(\.\d+)?$'

        for s in self.samples:
            answer = s.get('golden_answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            answer = str(answer).strip()

            if re.search(date_pattern, answer):
                answer_types["日期"] += 1
            elif re.match(number_pattern, answer):
                answer_types["数字"] += 1
            elif len(answer.split()) == 1:
                answer_types["人名"] += 1  # 简化判断
            elif len(answer.split()) <= 3:
                answer_types["短语"] += 1
            else:
                answer_types["句子"] += 1

        print("  答案类型分布:")
        for atype, count in answer_types.items():
            if count > 0:
                pct = count / len(self.samples) * 100
                print(f"    {atype:10s} {count:4d} ({pct:5.1f}%)")

    def check_potential_issues(self):
        """检查潜在问题"""
        print("\n【5. 潜在问题检查】")

        issues = []

        # 问题1: 检查是否有空问题或答案
        empty_questions = sum(1 for s in self.samples if not s.get('question', '').strip())
        if empty_questions > 0:
            issues.append(f"⚠️  发现 {empty_questions} 个空问题")

        empty_answers = sum(1 for s in self.samples if not s.get('golden_answer'))
        if empty_answers > 0:
            issues.append(f"⚠️  发现 {empty_answers} 个空答案")

        # 问题2: 检查是否有样本没有文档
        no_docs = sum(1 for s in self.samples if not s.get('docs'))
        if no_docs > 0:
            issues.append(f"⚠️  发现 {no_docs} 个样本没有检索文档")

        # 问题3: 检查"不可检索"样本的比例
        not_retrievable = sum(1 for s in self.samples if s.get('sample_type', '').startswith('not_retrievable'))
        not_retrievable_pct = not_retrievable / len(self.samples) * 100

        if not_retrievable_pct > 50:
            issues.append(f"⚠️  不可检索样本比例过高: {not_retrievable_pct:.1f}% (建议 < 50%)")
        elif not_retrievable_pct < 10:
            issues.append(f"ℹ️  不可检索样本比例较低: {not_retrievable_pct:.1f}% (可能无法充分学习承认不知道)")

        # 问题4: 检查检索质量是否有明显偏差
        qualities = [s.get('retrieval_quality', 0) for s in self.samples]
        avg_quality = sum(qualities) / len(qualities)

        if avg_quality < 0.3:
            issues.append(f"⚠️  平均检索质量过低: {avg_quality:.3f} (大部分样本检索质量差)")
        elif avg_quality > 0.8:
            issues.append(f"ℹ️  平均检索质量很高: {avg_quality:.3f} (缺少困难样本)")

        # 问题5: 检查重复样本
        questions = [s.get('question', '') for s in self.samples]
        unique_questions = len(set(questions))
        duplicate_rate = (len(questions) - unique_questions) / len(questions) * 100

        if duplicate_rate > 5:
            issues.append(f"⚠️  问题重复率较高: {duplicate_rate:.1f}%")

        # 输出问题
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✓ 未发现明显问题")

    def show_random_samples(self, n: int = 3):
        """展示随机样本"""
        print("\n【6. 随机样本预览】")

        # 分别从不同类型中各选一个
        types_to_show = {}

        for s in self.samples:
            stype = s.get('sample_type', 'unknown')
            if stype not in types_to_show:
                types_to_show[stype] = s

            if len(types_to_show) >= 3:
                break

        for idx, (stype, sample) in enumerate(types_to_show.items(), 1):
            print(f"\n  样本 {idx} [{stype}]:")
            print(f"  质量分数: {sample.get('retrieval_quality', 0):.3f}")
            print(f"  期望行为: {sample.get('expected_behavior', 'unknown')}")

            question = sample.get('question', '')
            print(f"  问题: {question[:100]}{'...' if len(question) > 100 else ''}")

            answer = sample.get('golden_answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            print(f"  答案: {str(answer)[:100]}")

            docs = sample.get('docs', [])
            print(f"  文档数: {len(docs)}")

            if docs:
                doc_text = docs[0].get('text', docs[0].get('title', ''))
                print(f"  首个文档: {doc_text[:80]}{'...' if len(doc_text) > 80 else ''}")

    def generate_recommendation(self):
        """生成训练建议"""
        print("\n" + "="*70)
        print("【训练建议】")
        print("="*70)

        qualities = [s.get('retrieval_quality', 0) for s in self.samples]
        avg_quality = sum(qualities) / len(qualities)

        not_retrievable = sum(1 for s in self.samples if s.get('retrieval_quality', 1) < 0.2)
        not_retrievable_pct = not_retrievable / len(self.samples) * 100

        print(f"\n当前数据集特征:")
        print(f"  - 平均检索质量: {avg_quality:.3f}")
        print(f"  - 不可检索样本占比: {not_retrievable_pct:.1f}%")

        print(f"\n适用场景:")

        if avg_quality >= 0.7 and not_retrievable_pct <= 15:
            print("  ✓ 适合训练初期 (Step 200-400)")
            print("  ✓ 重点: 修复推理能力")
            print("  ✓ 目标: 让模型学会正确使用检索到的信息")

        elif 0.5 <= avg_quality < 0.7 and 20 <= not_retrievable_pct <= 45:
            print("  ✓ 适合训练中期 (Step 400-700)")
            print("  ✓ 重点: 学习识别检索失败")
            print("  ✓ 目标: 让模型学会在检索失败时承认不知道")

        elif avg_quality < 0.5 or not_retrievable_pct > 40:
            print("  ✓ 适合训练后期 (Step 700+)")
            print("  ✓ 重点: 平衡与泛化")
            print("  ✓ 目标: 在诚实与能力之间找到平衡")

        print("\n推荐的reward设计:")
        print("  - 如果 retrieval_quality >= 0.7 且答对: +1.0")
        print("  - 如果 retrieval_quality >= 0.7 但答错: -0.5 (惩罚'有信息但用不好')")
        print("  - 如果 retrieval_quality < 0.2 且承认不知道: +0.8")
        print("  - 如果 retrieval_quality < 0.2 但乱答: -0.5")
        print("  - 避免过度使用'不知道': 如果出现>2次, -0.2 per extra")


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python check_sample_quality.py <sample_file.jsonl>")
        print("\n示例:")
        print("  python check_sample_quality.py prepared_samples/phase1_train.jsonl")
        sys.exit(1)

    sample_file = sys.argv[1]

    if not Path(sample_file).exists():
        print(f"错误: 文件不存在 {sample_file}")
        sys.exit(1)

    checker = SampleQualityChecker(sample_file)
    checker.check_all()
    checker.generate_recommendation()


if __name__ == "__main__":
    main()