#!/usr/bin/env python3
"""
BadCase 错误类型分析工具

分类标准：
1. 没有给出答案：模型没有生成 <answer> 标签
2. 检索错误：检索文档中不包含正确答案
3. 推理错误：检索到了正确答案，但模型提取/理解错误
"""

import json
import re
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BadCaseAnalysis:
    """BadCase 分析结果"""
    question: str
    golden_answer: str
    extracted_answer: str
    data_source: str
    num_searches: int
    error_type: str  # "no_answer", "retrieval_error", "reasoning_error"
    explanation: str
    full_trajectory: str


def normalize_answer(text: str) -> str:
    """标准化答案用于匹配"""
    if not text:
        return ""
    text = text.lower().strip()
    # 移除标点和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_answer_in_text(golden_answers: List[str], text: str) -> Tuple[bool, str]:
    """
    检查正确答案是否在文本中

    Returns:
        (found, matched_answer)
    """
    text_normalized = normalize_answer(text)

    for answer in golden_answers:
        answer_normalized = normalize_answer(answer)

        # 完全匹配
        if answer_normalized in text_normalized:
            return True, answer

        # 分词后匹配（处理多词答案）
        answer_words = answer_normalized.split()
        if len(answer_words) > 1:
            # 检查所有关键词是否都出现
            if all(word in text_normalized for word in answer_words):
                return True, answer

    return False, ""


def parse_golden_answer(golden_answer_str: str) -> List[str]:
    """解析 golden_answer 字符串"""
    # 尝试解析 {'target': array([...])} 格式
    if "array([" in golden_answer_str:
        match = re.search(r"array\(\[(.*?)\]", golden_answer_str, re.DOTALL)
        if match:
            content = match.group(1)
            # 提取所有引号内的内容
            answers = re.findall(r"'([^']*)'", content)
            if not answers:
                answers = re.findall(r'"([^"]*)"', content)
            return [a.strip() for a in answers if a.strip()]

    # 如果是简单字符串，直接返回
    return [golden_answer_str.strip()]


def extract_retrieved_docs(trajectory: str) -> List[str]:
    """提取所有检索到的文档内容"""
    docs = []
    pattern = r'<information>(.*?)</information>'
    matches = re.findall(pattern, trajectory, re.DOTALL)

    for match in matches:
        docs.append(match.strip())

    return docs


def check_query_format_errors(trajectory: str) -> Tuple[bool, List[str]]:
    """
    检查是否存在查询格式错误

    Returns:
        (has_error, error_examples)
    """
    errors = []

    # 检查错误格式 1: <search query="..."> </search>
    pattern1 = r'<search\s+query="([^"]*)">\s*</search>'
    matches1 = re.findall(pattern1, trajectory, re.IGNORECASE)
    if matches1:
        for match in matches1[:3]:  # 最多展示 3 个
            errors.append(f'格式错误: <search query="{match}"> </search>')

    # 检查错误格式 2: <search></search> (空查询)
    pattern2 = r'<search>\s*</search>'
    if re.search(pattern2, trajectory):
        errors.append('格式错误: <search></search> (空查询)')

    # 检查错误格式 3: <search>...(没有闭合标签)
    pattern3 = r'<search>([^<]*?)(?:<think>|<answer>|$)'
    matches3 = re.findall(pattern3, trajectory, re.DOTALL)
    for match in matches3:
        if '</search>' not in match and len(match.strip()) > 0:
            query_preview = match.strip()[:50]
            errors.append(f'格式错误: <search>{query_preview}... (未闭合)')
            break  # 只展示第一个

    # 检查错误格式 4: 查询太长或包含特殊结构（可能是误把整段话放进去）
    pattern4 = r'<search>(.*?)</search>'
    matches4 = re.findall(pattern4, trajectory, re.DOTALL)
    for match in matches4:
        if len(match) > 200:  # 查询超过 200 字符，异常
            errors.append(f'查询过长: {len(match)} 字符 (正常应 < 50)')
            break

    return len(errors) > 0, errors


def classify_badcase(result: Dict) -> BadCaseAnalysis:
    """
    分类 BadCase 的错误类型

    分类逻辑：
    0. 首先检查查询格式错误 → "query_format_error"
    1. extracted_answer 为空：
       1.1 检索到了正确答案 → "no_answer_with_correct_docs"（最关键！）
       1.2 没检索到正确答案 → "no_answer_retrieval_error"
    2. extracted_answer 不为空：
       2.1 检索文档中不包含正确答案 → "retrieval_error"（检索错误）
       2.2 检索文档中包含正确答案 → "reasoning_error"（推理错误）
    """
    question = result.get('question', '')
    golden_answer_str = result.get('golden_answer', '')
    extracted_answer = result.get('extracted_answer', '')
    data_source = result.get('data_source', 'unknown')
    num_searches = result.get('num_searches', 0)
    full_trajectory = result.get('full_trajectory', '') or result.get('model_answer', '')

    # 解析正确答案
    golden_answers = parse_golden_answer(golden_answer_str)

    # 提取检索文档
    retrieved_docs = extract_retrieved_docs(full_trajectory)
    all_retrieved_text = "\n".join(retrieved_docs)

    # 检查正确答案是否在检索文档中
    found_in_docs, matched_answer = check_answer_in_text(golden_answers, all_retrieved_text)

    # 类型 0：查询格式错误（优先检查）
    has_query_error, query_errors = check_query_format_errors(full_trajectory)
    if has_query_error:
        error_detail = "; ".join(query_errors)
        return BadCaseAnalysis(
            question=question,
            golden_answer=golden_answer_str,
            extracted_answer=extracted_answer,
            data_source=data_source,
            num_searches=num_searches,
            error_type="query_format_error",
            explanation=f"查询格式错误导致搜索失败: {error_detail}",
            full_trajectory=full_trajectory
        )

    # 类型 1：没有给出答案（细分为两种情况）
    if not extracted_answer or extracted_answer.strip() == "":
        if found_in_docs:
            # 1.1 检索到了正确答案，但模型没生成 <answer>
            return BadCaseAnalysis(
                question=question,
                golden_answer=golden_answer_str,
                extracted_answer=extracted_answer,
                data_source=data_source,
                num_searches=num_searches,
                error_type="no_answer_with_correct_docs",
                explanation=f"模型进行了 {num_searches} 次搜索，检索到了包含正确答案 '{matched_answer}' 的文档，但没有生成 <answer> 标签",
                full_trajectory=full_trajectory
            )
        else:
            # 1.2 检索没找到正确答案，也没生成 <answer>
            return BadCaseAnalysis(
                question=question,
                golden_answer=golden_answer_str,
                extracted_answer=extracted_answer,
                data_source=data_source,
                num_searches=num_searches,
                error_type="no_answer_retrieval_error",
                explanation=f"模型进行了 {num_searches} 次搜索，检索文档中不包含正确答案 '{golden_answers[0]}'，且没有生成 <answer> 标签",
                full_trajectory=full_trajectory
            )

    # 类型 2：给出了答案，但检索错误
    if not found_in_docs:
        return BadCaseAnalysis(
            question=question,
            golden_answer=golden_answer_str,
            extracted_answer=extracted_answer,
            data_source=data_source,
            num_searches=num_searches,
            error_type="retrieval_error",
            explanation=f"模型进行了 {num_searches} 次搜索，但检索到的文档中不包含正确答案 '{golden_answers[0]}'，模型回答了 '{extracted_answer}'",
            full_trajectory=full_trajectory
        )

    # 类型 3：给出了答案，检索也正确，但推理错误
    return BadCaseAnalysis(
        question=question,
        golden_answer=golden_answer_str,
        extracted_answer=extracted_answer,
        data_source=data_source,
        num_searches=num_searches,
        error_type="reasoning_error",
        explanation=f"检索到的文档中包含正确答案 '{matched_answer}'，但模型提取成了 '{extracted_answer}'",
        full_trajectory=full_trajectory
    )


def analyze_badcases(json_path: str, output_path: str = None, sample_size: int = 10):
    """分析 BadCase 并生成报告"""

    # 读取 JSON
    print(f"读取评估结果: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])

    # 筛选错误样本
    badcases = [r for r in results if not r.get('is_correct', False)]

    print(f"总样本数: {len(results)}")
    print(f"错误样本数: {len(badcases)}")
    print(f"\n开始分析 BadCase 类型...\n")

    # 分类统计
    error_types = defaultdict(list)
    source_error_types = defaultdict(lambda: defaultdict(int))

    for result in badcases:
        analysis = classify_badcase(result)
        error_types[analysis.error_type].append(analysis)
        source_error_types[analysis.data_source][analysis.error_type] += 1

    # 打印统计结果
    print("=" * 100)
    print("BadCase 错误类型统计")
    print("=" * 100)
    print()

    total = len(badcases)

    print(f"{'错误类型':<30} {'数量':>10} {'占比':>10} {'说明':<40}")
    print("-" * 100)

    type_info = {
        "query_format_error": "🔴【严重】查询格式错误（导致搜索失败）",
        "no_answer_with_correct_docs": "🟠【重点】检索到正确答案，但没生成 <answer>",
        "no_answer_retrieval_error": "🟡 检索错误 + 没生成 <answer>",
        "retrieval_error": "🟢 检索错误（给出了错误答案）",
        "reasoning_error": "🔵 推理错误（检索到了但理解/提取错误）"
    }

    for error_type in ["query_format_error", "no_answer_with_correct_docs", "no_answer_retrieval_error", "retrieval_error", "reasoning_error"]:
        count = len(error_types[error_type])
        percentage = count / total * 100
        info = type_info.get(error_type, "")
        print(f"{error_type:<30} {count:>10} {percentage:>9.1f}% {info:<40}")

    print()
    print("=" * 100)
    print("按数据源统计错误类型")
    print("=" * 100)
    print()

    print(f"{'数据源':<20} {'总错误':>10} {'格式错':>10} {'检索到无答案':>14} {'检索错无答案':>14} {'检索错':>10} {'推理错':>10}")
    print("-" * 120)

    for source in sorted(source_error_types.keys()):
        stats = source_error_types[source]
        total_errors = sum(stats.values())
        query_err = stats.get('query_format_error', 0)
        no_ans_correct = stats.get('no_answer_with_correct_docs', 0)
        no_ans_retrieval = stats.get('no_answer_retrieval_error', 0)
        retrieval_err = stats.get('retrieval_error', 0)
        reasoning_err = stats.get('reasoning_error', 0)

        print(f"{source:<20} {total_errors:>10} {query_err:>10} {no_ans_correct:>14} {no_ans_retrieval:>14} {retrieval_err:>10} {reasoning_err:>10}")

    # 保存详细报告
    if output_path is None:
        output_path = json_path.replace('.json', '_error_analysis.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("BadCase 错误类型详细分析\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"总样本数: {len(results)}\n")
        f.write(f"错误样本数: {len(badcases)}\n")
        f.write(f"准确率: {(len(results) - len(badcases)) / len(results) * 100:.2f}%\n\n")

        # 按错误类型分组展示
        for error_type in ["query_format_error", "no_answer_with_correct_docs", "no_answer_retrieval_error", "retrieval_error", "reasoning_error"]:
            cases = error_types[error_type]
            if not cases:
                continue

            f.write("\n" + "=" * 100 + "\n")
            f.write(f"{error_type.upper()}: {type_info[error_type]}\n")
            f.write(f"共 {len(cases)} 个样本 ({len(cases)/total*100:.1f}%)\n")
            f.write("=" * 100 + "\n\n")

            # 特别标注重点关注的类型
            if error_type == "query_format_error":
                f.write("🔴 严重问题：模型生成了错误格式的查询，导致检索失败！\n")
                f.write("    常见错误：\n")
                f.write("    1. <search query=\"...\"> </search> (XML 属性格式，应该是 <search>...</search>)\n")
                f.write("    2. <search></search> (空查询)\n")
                f.write("    3. <search>... (没有闭合标签)\n")
                f.write("    4. 查询过长（超过 200 字符）\n")
                f.write("    解决方案：\n")
                f.write("    - 检查 Prompt 是否给出了正确的格式示例\n")
                f.write("    - 训练数据中是否有格式错误的样本\n")
                f.write("    - 考虑增加格式约束的 reward\n\n")

            elif error_type == "no_answer_with_correct_docs":
                f.write("⚠️  重点关注：这些样本检索到了正确答案，但模型没有生成 <answer> 标签！\n")
                f.write("    可能原因：\n")
                f.write("    1. 达到 max_turns 限制，被强制停止\n")
                f.write("    2. 模型判断信息不足，想继续搜索但不能了\n")
                f.write("    3. 模型陷入循环搜索，忘记给答案\n")
                f.write("    4. Prompt 或训练问题，模型没学会何时给答案\n\n")

            # 每种类型展示前 N 个样本
            for i, case in enumerate(cases[:sample_size], 1):
                f.write(f"\n{'─' * 100}\n")
                f.write(f"示例 {i}/{min(sample_size, len(cases))}\n")
                f.write(f"{'─' * 100}\n\n")

                f.write(f"问题: {case.question}\n")
                f.write(f"数据源: {case.data_source}\n")
                f.write(f"标准答案: {case.golden_answer}\n")
                f.write(f"模型答案: {case.extracted_answer}\n")
                f.write(f"搜索次数: {case.num_searches}\n")
                f.write(f"错误说明: {case.explanation}\n")

                f.write(f"\n完整轨迹:\n")
                f.write("─" * 100 + "\n")
                f.write(case.full_trajectory[:2000])  # 限制长度
                if len(case.full_trajectory) > 2000:
                    f.write("\n... (已截断)")
                f.write("\n" + "─" * 100 + "\n")

    print()
    print("=" * 100)
    print(f"✓ 详细分析已保存到: {output_path}")
    print(f"  每种错误类型展示前 {sample_size} 个样本")
    print("=" * 100)

    # 返回统计信息
    return {
        'total': total,
        'query_format_error': len(error_types['query_format_error']),
        'no_answer_with_correct_docs': len(error_types['no_answer_with_correct_docs']),
        'no_answer_retrieval_error': len(error_types['no_answer_retrieval_error']),
        'retrieval_error': len(error_types['retrieval_error']),
        'reasoning_error': len(error_types['reasoning_error']),
        'by_source': dict(source_error_types)
    }


def main():
    parser = argparse.ArgumentParser(description="分析 BadCase 错误类型")
    parser.add_argument('json_path', type=str, help='评估结果 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径')
    parser.add_argument('--sample_size', '-n', type=int, default=10,
                       help='每种错误类型展示的样本数量（默认 10）')

    args = parser.parse_args()

    analyze_badcases(args.json_path, args.output, args.sample_size)


if __name__ == '__main__':
    main()
