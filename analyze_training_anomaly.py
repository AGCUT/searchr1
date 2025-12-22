#!/usr/bin/env python3
"""
分析 Step 200-300 训练异常
检查是否由数据问题导致训练失效
"""

import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def parse_training_log(log_file):
    """解析训练日志，提取关键指标"""
    metrics = defaultdict(list)

    with open(log_file, 'r', encoding='utf-8') as f:
        current_step = None

        for line in f:
            # 提取 step
            step_match = re.search(r'epoch (\d+), step (\d+)', line)
            if step_match:
                current_step = int(step_match.group(2))
                continue

            # 提取打印的序列（包含 ground truth 和生成结果）
            if current_step and ('Question:' in line or 'Answer:' in line):
                metrics['generated_sequences'].append({
                    'step': current_step,
                    'content': line.strip()
                })

    return metrics

def analyze_wandb_metrics(wandb_export_csv):
    """
    分析从 WandB 导出的 CSV 数据

    如何导出：
    1. 打开 WandB 实验页面
    2. 点击右上角 "Export" -> "Export data as CSV"
    3. 选择所有指标，下载 CSV
    """
    df = pd.read_csv(wandb_export_csv)

    # 过滤 step 200-300
    df_anomaly = df[(df['_step'] >= 200) & (df['_step'] <= 300)]

    # 关键指标统计
    metrics_to_check = [
        'critic/rewards/mean',
        'critic/advantages/mean',
        'critic/advantages/max',
        'critic/advantages/min',
        'response_length/mean',
        'response_length/clip_ratio',
        'actor/grad_norm',
        'critic/kl',
        'critic/score/mean',
    ]

    print("=" * 80)
    print("Step 200-300 异常指标分析")
    print("=" * 80)

    for metric in metrics_to_check:
        if metric in df.columns:
            values = df_anomaly[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  均值: {values.mean():.4f}")
                print(f"  标准差: {values.std():.4f}")
                print(f"  最大值: {values.max():.4f}")
                print(f"  最小值: {values.min():.4f}")

                # 检测异常
                if 'grad_norm' in metric and values.max() > 10:
                    print(f"  ⚠️  梯度爆炸！max={values.max():.2f}")

                if 'kl' in metric and values.max() > 1.0:
                    print(f"  ⚠️  KL 散度过大！max={values.max():.4f}")

                if 'clip_ratio' in metric and values.mean() > 0.5:
                    print(f"  ⚠️  大量序列被截断！mean={values.mean():.2%}")

                if 'advantages/max' in metric and values.max() > 10:
                    print(f"  ⚠️  极端 advantage 值！max={values.max():.2f}")

    # 绘制关键指标趋势
    plot_metrics_trend(df, metrics_to_check)

    return df_anomaly

def plot_metrics_trend(df, metrics):
    """绘制指标趋势图"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics[:9]):
        if metric in df.columns:
            ax = axes[idx]

            # 绘制整体趋势
            df_plot = df[['_step', metric]].dropna()
            ax.plot(df_plot['_step'], df_plot[metric], alpha=0.5, label='全部')

            # 高亮 step 200-300
            df_anomaly = df_plot[(df_plot['_step'] >= 200) & (df_plot['_step'] <= 300)]
            ax.plot(df_anomaly['_step'], df_anomaly[metric],
                   color='red', linewidth=2, label='Step 200-300')

            # 标记 step 200 和 300
            ax.axvline(x=200, color='green', linestyle='--', alpha=0.5, label='Step 200')
            ax.axvline(x=300, color='orange', linestyle='--', alpha=0.5, label='Step 300')

            ax.set_xlabel('Step')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('step_200_300_anomaly_analysis.png', dpi=150)
    print("\n✓ 趋势图已保存到: step_200_300_anomaly_analysis.png")

def check_data_quality(data_file):
    """
    检查训练数据质量

    参数:
        data_file: 训练数据文件路径（parquet 格式）
    """
    import pyarrow.parquet as pq

    print("\n" + "=" * 80)
    print("训练数据质量检查")
    print("=" * 80)

    # 读取数据
    table = pq.read_table(data_file)
    df = table.to_pandas()

    print(f"\n总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 检查 prompt 长度分布
    if 'prompt' in df.columns:
        df['prompt_length'] = df['prompt'].apply(lambda x: len(str(x).split()))

        print(f"\nPrompt 长度统计:")
        print(f"  均值: {df['prompt_length'].mean():.1f} words")
        print(f"  中位数: {df['prompt_length'].median():.1f} words")
        print(f"  最大值: {df['prompt_length'].max()} words")
        print(f"  最小值: {df['prompt_length'].min()} words")
        print(f"  标准差: {df['prompt_length'].std():.1f}")

        # 检查异常长度的样本
        long_prompts = df[df['prompt_length'] > 1000]
        if len(long_prompts) > 0:
            print(f"\n⚠️  发现 {len(long_prompts)} 个超长 prompt (>1000 words)")
            print(f"  占比: {len(long_prompts)/len(df)*100:.2f}%")

    # 检查 ground truth
    if 'ground_truth' in df.columns:
        df['has_answer'] = df['ground_truth'].notna()
        print(f"\n有答案的样本: {df['has_answer'].sum()} / {len(df)} ({df['has_answer'].mean()*100:.1f}%)")

        missing = df[~df['has_answer']]
        if len(missing) > 0:
            print(f"⚠️  发现 {len(missing)} 个没有答案的样本")

    # 检查数据源分布
    if 'data_source' in df.columns:
        print(f"\n数据源分布:")
        for source, count in df['data_source'].value_counts().items():
            print(f"  {source}: {count} ({count/len(df)*100:.1f}%)")

    return df

def estimate_step_200_300_samples(batch_size=512, dataloader_size=331):
    """
    估算 step 200-300 对应的数据样本

    参数:
        batch_size: 每个 batch 的样本数
        dataloader_size: 每个 epoch 的 batch 数量
    """
    print("\n" + "=" * 80)
    print("Step 200-300 样本范围估算")
    print("=" * 80)

    # Step 200 对应的样本
    step_200_batch_idx = 199  # step 从 1 开始，所以 step 200 是第 200 个 batch
    step_200_sample_start = step_200_batch_idx * batch_size
    step_200_sample_end = (step_200_batch_idx + 1) * batch_size

    # Step 300 对应的样本
    step_300_batch_idx = 299
    step_300_sample_start = step_300_batch_idx * batch_size
    step_300_sample_end = (step_300_batch_idx + 1) * batch_size

    print(f"\n每个 batch: {batch_size} 个样本")
    print(f"每个 epoch: {dataloader_size} 个 batch")
    print(f"总样本数（估算）: {dataloader_size * batch_size} 个")

    print(f"\nStep 200 (batch {step_200_batch_idx}):")
    print(f"  样本范围: [{step_200_sample_start}, {step_200_sample_end})")

    print(f"\nStep 300 (batch {step_300_batch_idx}):")
    print(f"  样本范围: [{step_300_sample_start}, {step_300_sample_end})")

    print(f"\nStep 200-300 区间:")
    print(f"  样本范围: [{step_200_sample_start}, {step_300_sample_end})")
    print(f"  总样本数: {step_300_sample_end - step_200_sample_start}")

    return (step_200_sample_start, step_300_sample_end)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分析训练异常')
    parser.add_argument('--log', type=str,
                       default='nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log',
                       help='训练日志文件')
    parser.add_argument('--wandb-csv', type=str,
                       help='从 WandB 导出的 CSV 文件')
    parser.add_argument('--data', type=str,
                       help='训练数据文件（parquet）')

    args = parser.parse_args()

    # 1. 估算样本范围
    sample_range = estimate_step_200_300_samples()

    # 2. 分析 WandB 指标
    if args.wandb_csv and Path(args.wandb_csv).exists():
        print(f"\n正在分析 WandB 数据: {args.wandb_csv}")
        df_anomaly = analyze_wandb_metrics(args.wandb_csv)
    else:
        print("\n⚠️  未提供 WandB CSV 文件，跳过指标分析")
        print("提示: 从 WandB 导出 CSV 后使用 --wandb-csv 参数")

    # 3. 检查数据质量
    if args.data and Path(args.data).exists():
        print(f"\n正在检查数据质量: {args.data}")
        df_data = check_data_quality(args.data)

        # 提取 step 200-300 对应的样本
        start_idx, end_idx = sample_range
        print(f"\n提取 step 200-300 对应的样本:")
        df_suspicious = df_data.iloc[start_idx:end_idx]
        df_suspicious.to_csv('step_200_300_samples.csv', index=False)
        print(f"✓ 已保存到: step_200_300_samples.csv")
    else:
        print("\n⚠️  未提供数据文件，跳过数据质量检查")
        print("提示: 使用 --data 参数指定训练数据文件")

    # 4. 解析训练日志（可选）
    if args.log and Path(args.log).exists():
        print(f"\n正在解析训练日志: {args.log}")
        metrics = parse_training_log(args.log)
        print(f"✓ 解析完成")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n下一步建议:")
    print("1. 检查生成的图表: step_200_300_anomaly_analysis.png")
    print("2. 检查可疑样本: step_200_300_samples.csv")
    print("3. 在 WandB 上查看详细指标变化")

if __name__ == "__main__":
    main()
