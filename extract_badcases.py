#!/usr/bin/env python3
"""
从评估结果 JSON 文件中提取 BadCase 并生成详细分析报告
"""

import json
import argparse
from pathlib import Path


def extract_badcases(json_path: str, output_path: str = None):
    """从 JSON 提取 BadCase"""

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])

    # 筛选错误样本
    badcases = [r for r in results if not r.get('is_correct', False)]

    print(f"总样本数: {len(results)}")
    print(f"错误样本数: {len(badcases)}")
    print(f"准确率: {(len(results) - len(badcases)) / len(results) * 100:.2f}%")

    # 确定输出路径
    if output_path is None:
        output_path = json_path.replace('.json', '_badcases.txt')

    # 写入 BadCase 分析
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"BadCase 分析报告 (共 {len(badcases)} 个错误)\n")
        f.write(f"准确率: {(len(results) - len(badcases)) / len(results) * 100:.2f}%\n")
        f.write("=" * 100 + "\n\n")

        for idx, r in enumerate(badcases, 1):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"BadCase #{idx}/{len(badcases)}\n")
            f.write(f"{'=' * 100}\n\n")

            f.write(f"问题: {r.get('question', 'N/A')}\n")
            f.write(f"数据源: {r.get('data_source', 'N/A')}\n")
            f.write(f"标准答案: {r.get('golden_answer', 'N/A')}\n")
            f.write(f"模型答案: {r.get('extracted_answer', 'N/A')}\n")
            f.write(f"搜索次数: {r.get('num_searches', 0)}\n")
            f.write(f"答案数量: {r.get('num_answers', 0)}\n")
            f.write(f"是否改变答案: {'是' if r.get('answer_changed', False) else '否'}\n")

            if r.get('answer_changed', False) and r.get('all_answers'):
                f.write(f"所有答案: {r.get('all_answers')}\n")

            f.write(f"响应时间: {r.get('response_time', 0):.2f}s\n")

            f.write(f"\n完整轨迹:\n")
            f.write("-" * 100 + "\n")
            trajectory = r.get('full_trajectory') or r.get('model_answer', 'N/A')
            f.write(trajectory)
            f.write("\n" + "-" * 100 + "\n")

    print(f"\n✓ BadCase 分析已保存到: {output_path}")

    # 按数据源统计
    print("\n【BadCase 按数据源分布】")
    source_stats = {}
    for r in badcases:
        source = r.get('data_source', 'unknown')
        source_stats[source] = source_stats.get(source, 0) + 1

    for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} 个")

    # 按搜索次数统计
    print("\n【BadCase 按搜索次数分布】")
    search_stats = {}
    for r in badcases:
        num_searches = r.get('num_searches', 0)
        search_stats[num_searches] = search_stats.get(num_searches, 0) + 1

    for num, count in sorted(search_stats.items()):
        print(f"  {num} 次搜索: {count} 个")

    # 按是否改变答案统计
    print("\n【BadCase 按答案变化统计】")
    changed = sum(1 for r in badcases if r.get('answer_changed', False))
    unchanged = len(badcases) - changed
    print(f"  改变答案: {changed} 个")
    print(f"  未改变答案: {unchanged} 个")


def main():
    parser = argparse.ArgumentParser(description="从评估结果 JSON 提取 BadCase")
    parser.add_argument('json_path', type=str, help='评估结果 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径（默认：自动生成）')

    args = parser.parse_args()

    extract_badcases(args.json_path, args.output)


if __name__ == '__main__':
    main()
