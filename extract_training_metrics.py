# -*- coding: utf-8 -*-
"""
训练日志分析工具
从 veRL 训练日志中提取关键指标，帮助诊断训练问题

用法：
    python extract_training_metrics.py <log_file_path>

例如：
    python extract_training_metrics.py /usr/yuque/guo/searchr1/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log
"""

import re
import sys
import json
from collections import defaultdict
from pathlib import Path


def extract_metrics_from_log(log_path: str) -> dict:
    """
    从训练日志中提取关键指标

    参数:
        log_path: 日志文件路径

    返回:
        包含各个step指标的字典
    """

    # 存储提取的指标
    metrics = defaultdict(list)

    # 常见的指标模式（根据 veRL 的日志格式）
    patterns = {
        # 步数/轮次
        'step': [
            r'step[:\s]+(\d+)',
            r'global_step[:\s]+(\d+)',
            r'training_step[:\s]+(\d+)',
        ],
        'epoch': [
            r'epoch[:\s]+(\d+)',
        ],

        # 奖励相关
        'reward': [
            r'reward[:\s]+([-\d.]+)',
            r'mean_reward[:\s]+([-\d.]+)',
            r'avg_reward[:\s]+([-\d.]+)',
            r"'reward'[:\s]+([-\d.]+)",
        ],
        'reward_mean': [
            r'reward/mean[:\s]+([-\d.]+)',
            r'reward_mean[:\s]+([-\d.]+)',
        ],
        'reward_std': [
            r'reward/std[:\s]+([-\d.]+)',
            r'reward_std[:\s]+([-\d.]+)',
        ],

        # KL 散度
        'kl': [
            r'kl[:\s]+([-\d.]+)',
            r'kl_loss[:\s]+([-\d.]+)',
            r'kl_div[:\s]+([-\d.]+)',
            r"'kl'[:\s]+([-\d.]+)",
            r'approx_kl[:\s]+([-\d.]+)',
        ],
        'kl_coef': [
            r'kl_coef[:\s]+([-\d.]+)',
        ],

        # 策略损失
        'policy_loss': [
            r'policy_loss[:\s]+([-\d.]+)',
            r'actor_loss[:\s]+([-\d.]+)',
            r'pg_loss[:\s]+([-\d.]+)',
            r"'policy_loss'[:\s]+([-\d.]+)",
        ],

        # 价值损失
        'value_loss': [
            r'value_loss[:\s]+([-\d.]+)',
            r'critic_loss[:\s]+([-\d.]+)',
            r'vf_loss[:\s]+([-\d.]+)',
            r"'value_loss'[:\s]+([-\d.]+)",
        ],

        # 熵
        'entropy': [
            r'entropy[:\s]+([-\d.]+)',
            r'entropy_loss[:\s]+([-\d.]+)',
            r"'entropy'[:\s]+([-\d.]+)",
        ],

        # Clip 相关
        'clip_frac': [
            r'clip_frac[:\s]+([-\d.]+)',
            r'clip_fraction[:\s]+([-\d.]+)',
            r'clipfrac[:\s]+([-\d.]+)',
        ],

        # 优势函数
        'advantage': [
            r'advantage[:\s]+([-\d.]+)',
            r'adv_mean[:\s]+([-\d.]+)',
            r'advantages[:\s]+([-\d.]+)',
        ],

        # 验证/测试指标
        'val_reward': [
            r'val[/_]reward[:\s]+([-\d.]+)',
            r'test[/_]reward[:\s]+([-\d.]+)',
            r'eval[/_]reward[:\s]+([-\d.]+)',
        ],
        'val_score': [
            r'val[/_]score[:\s]+([-\d.]+)',
            r'test[/_]score[:\s]+([-\d.]+)',
            r'em_score[:\s]+([-\d.]+)',
            r'accuracy[:\s]+([-\d.]+)',
        ],

        # 学习率
        'lr': [
            r'lr[:\s]+([-\d.e]+)',
            r'learning_rate[:\s]+([-\d.e]+)',
            r'actor_lr[:\s]+([-\d.e]+)',
        ],

        # 响应长度
        'response_length': [
            r'response_length[:\s]+([-\d.]+)',
            r'gen_len[:\s]+([-\d.]+)',
            r'avg_length[:\s]+([-\d.]+)',
        ],
    }

    # 当前步数（用于关联指标）
    current_step = None
    current_epoch = None

    print(f"正在读取日志文件: {log_path}")
    print("这可能需要一些时间...")

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = 0
            for line in f:
                line_count += 1
                line_lower = line.lower()

                # 尝试提取步数
                for pattern in patterns['step']:
                    match = re.search(pattern, line_lower)
                    if match:
                        current_step = int(match.group(1))
                        break

                # 尝试提取epoch
                for pattern in patterns['epoch']:
                    match = re.search(pattern, line_lower)
                    if match:
                        current_epoch = int(match.group(1))
                        break

                # 提取其他指标
                for metric_name, metric_patterns in patterns.items():
                    if metric_name in ['step', 'epoch']:
                        continue

                    for pattern in metric_patterns:
                        match = re.search(pattern, line_lower)
                        if match:
                            try:
                                value = float(match.group(1))
                                metrics[metric_name].append({
                                    'step': current_step,
                                    'epoch': current_epoch,
                                    'value': value,
                                    'line': line_count
                                })
                            except ValueError:
                                pass
                            break

                # 每10万行打印进度
                if line_count % 100000 == 0:
                    print(f"  已处理 {line_count} 行...")

        print(f"总共处理了 {line_count} 行")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_path}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件时出错 - {e}")
        sys.exit(1)

    return dict(metrics)


def analyze_metrics(metrics: dict) -> dict:
    """
    分析提取的指标，计算统计信息
    """
    analysis = {}

    for metric_name, values in metrics.items():
        if not values:
            continue

        # 按步数排序
        sorted_values = sorted(values, key=lambda x: (x['step'] or 0))

        # 提取数值
        nums = [v['value'] for v in sorted_values]
        steps = [v['step'] for v in sorted_values]

        # 计算统计信息
        analysis[metric_name] = {
            'count': len(nums),
            'min': min(nums),
            'max': max(nums),
            'mean': sum(nums) / len(nums),
            'first_value': nums[0] if nums else None,
            'last_value': nums[-1] if nums else None,
            'first_step': steps[0] if steps else None,
            'last_step': steps[-1] if steps else None,
            # 保存所有数据点用于详细分析
            'all_values': sorted_values
        }

    return analysis


def print_summary(analysis: dict):
    """
    打印分析摘要
    """
    print("\n" + "=" * 70)
    print("训练指标摘要")
    print("=" * 70)

    # 关键指标的显示顺序和说明
    key_metrics = [
        ('reward', '奖励', '应该上升或稳定'),
        ('reward_mean', '平均奖励', '应该上升或稳定'),
        ('val_reward', '验证奖励', '应该上升或稳定'),
        ('val_score', '验证分数', '应该上升或稳定'),
        ('kl', 'KL散度', '应该 < 0.1，不要爆炸'),
        ('policy_loss', '策略损失', '应该下降或稳定'),
        ('value_loss', '价值损失', '应该下降或稳定'),
        ('entropy', '策略熵', '不应该持续下降到0'),
        ('clip_frac', 'Clip比例', '应该 < 0.3'),
        ('advantage', '优势函数', '均值应接近0'),
        ('lr', '学习率', '参考值'),
        ('response_length', '响应长度', '参考值'),
    ]

    for metric_key, metric_name, description in key_metrics:
        if metric_key in analysis:
            data = analysis[metric_key]
            print(f"\n{metric_name} ({metric_key}):")
            print(f"  说明: {description}")
            print(f"  数据点数: {data['count']}")
            print(f"  范围: [{data['min']:.6f}, {data['max']:.6f}]")
            print(f"  均值: {data['mean']:.6f}")
            print(f"  起始值 (step {data['first_step']}): {data['first_value']:.6f}")
            print(f"  最终值 (step {data['last_step']}): {data['last_value']:.6f}")

            # 计算变化趋势
            if data['first_value'] and data['last_value']:
                change = data['last_value'] - data['first_value']
                change_pct = (change / abs(data['first_value'])) * 100 if data['first_value'] != 0 else 0
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"  变化趋势: {trend} {change:+.6f} ({change_pct:+.1f}%)")


def print_step_comparison(analysis: dict, steps_to_compare: list):
    """
    打印特定步数的指标对比
    """
    print("\n" + "=" * 70)
    print(f"步数对比: {steps_to_compare}")
    print("=" * 70)

    key_metrics = ['reward', 'reward_mean', 'kl', 'policy_loss', 'value_loss',
                   'entropy', 'clip_frac', 'val_reward', 'val_score']

    # 表头
    header = f"{'指标':<20}"
    for step in steps_to_compare:
        header += f"{'Step '+str(step):>15}"
    print(header)
    print("-" * (20 + 15 * len(steps_to_compare)))

    for metric_key in key_metrics:
        if metric_key not in analysis:
            continue

        data = analysis[metric_key]
        all_values = data['all_values']

        row = f"{metric_key:<20}"
        for target_step in steps_to_compare:
            # 找到最接近目标步数的值
            closest = min(all_values, key=lambda x: abs((x['step'] or 0) - target_step))
            if closest and abs((closest['step'] or 0) - target_step) < 50:
                row += f"{closest['value']:>15.6f}"
            else:
                row += f"{'N/A':>15}"

        print(row)


def save_detailed_report(analysis: dict, output_path: str):
    """
    保存详细报告到文件
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("详细训练指标报告")
    report_lines.append("=" * 70)

    for metric_name, data in analysis.items():
        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"指标: {metric_name}")
        report_lines.append(f"{'='*50}")
        report_lines.append(f"Step\t\tValue")
        report_lines.append("-" * 30)

        for item in data['all_values']:
            step = item['step'] if item['step'] is not None else 'N/A'
            report_lines.append(f"{step}\t\t{item['value']:.6f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n详细报告已保存到: {output_path}")


def export_to_csv(analysis: dict, output_path: str):
    """
    导出指标到CSV文件，方便进一步分析
    """
    # 收集所有步数
    all_steps = set()
    for data in analysis.values():
        for item in data['all_values']:
            if item['step'] is not None:
                all_steps.add(item['step'])

    all_steps = sorted(all_steps)

    # 创建CSV内容
    metrics_names = list(analysis.keys())
    header = ['step'] + metrics_names

    rows = []
    for step in all_steps:
        row = [str(step)]
        for metric_name in metrics_names:
            # 找到这个步数对应的值
            data = analysis[metric_name]
            value = None
            for item in data['all_values']:
                if item['step'] == step:
                    value = item['value']
                    break
            row.append(str(value) if value is not None else '')
        rows.append(','.join(row))

    csv_content = ','.join(header) + '\n' + '\n'.join(rows)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)

    print(f"CSV文件已保存到: {output_path}")


def diagnose_issues(analysis: dict):
    """
    自动诊断可能的问题
    """
    print("\n" + "=" * 70)
    print("自动诊断")
    print("=" * 70)

    issues = []
    suggestions = []

    # 1. 检查 KL 散度
    if 'kl' in analysis:
        kl_data = analysis['kl']
        if kl_data['last_value'] > 0.1:
            issues.append(f"⚠️ KL散度过高: {kl_data['last_value']:.4f} (建议 < 0.1)")
            suggestions.append("→ 增大 kl_coef 或减小学习率")
        if kl_data['last_value'] > kl_data['first_value'] * 5:
            issues.append(f"⚠️ KL散度增长过快: {kl_data['first_value']:.4f} → {kl_data['last_value']:.4f}")
            suggestions.append("→ 使用自适应KL控制器或增大kl_coef")

    # 2. 检查熵
    if 'entropy' in analysis:
        ent_data = analysis['entropy']
        if ent_data['last_value'] < ent_data['first_value'] * 0.1:
            issues.append(f"⚠️ 策略熵下降过多: {ent_data['first_value']:.4f} → {ent_data['last_value']:.4f}")
            suggestions.append("→ 增大 entropy_coef 以鼓励探索")

    # 3. 检查 Clip 比例
    if 'clip_frac' in analysis:
        clip_data = analysis['clip_frac']
        if clip_data['mean'] > 0.3:
            issues.append(f"⚠️ Clip比例过高: 平均 {clip_data['mean']:.4f} (建议 < 0.3)")
            suggestions.append("→ 减小学习率或增大clip_ratio")

    # 4. 检查奖励趋势
    reward_key = 'reward' if 'reward' in analysis else 'reward_mean' if 'reward_mean' in analysis else None
    if reward_key:
        reward_data = analysis[reward_key]
        # 检查后期是否下降
        all_values = reward_data['all_values']
        if len(all_values) > 10:
            mid_point = len(all_values) // 2
            first_half_avg = sum(v['value'] for v in all_values[:mid_point]) / mid_point
            second_half_avg = sum(v['value'] for v in all_values[mid_point:]) / (len(all_values) - mid_point)

            if second_half_avg < first_half_avg * 0.9:
                issues.append(f"⚠️ 奖励在后期下降: 前半段平均 {first_half_avg:.4f}, 后半段平均 {second_half_avg:.4f}")
                suggestions.append("→ 考虑早停或减小学习率")

    # 5. 检查价值损失
    if 'value_loss' in analysis:
        vl_data = analysis['value_loss']
        if vl_data['last_value'] > vl_data['first_value'] * 2:
            issues.append(f"⚠️ 价值损失上升: {vl_data['first_value']:.4f} → {vl_data['last_value']:.4f}")
            suggestions.append("→ 减少ppo_epochs或增大cliprange_value")

    # 输出诊断结果
    if issues:
        print("\n发现以下问题:")
        for issue in issues:
            print(f"  {issue}")

        print("\n建议的解决方案:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("\n✓ 未发现明显问题")

    print("\n" + "-" * 70)
    print("注意: 以上诊断仅供参考，请结合WandB图表和实际生成样本进行综合判断")


def main():
    # 默认日志路径
    default_log_path = "/usr/yuque/guo/searchr1/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log"

    # 获取日志路径
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = default_log_path
        print(f"使用默认日志路径: {log_path}")

    # 提取指标
    metrics = extract_metrics_from_log(log_path)

    if not metrics:
        print("警告: 未能从日志中提取到任何指标")
        print("可能的原因:")
        print("1. 日志格式与预期不符")
        print("2. 日志文件为空或损坏")
        return

    print(f"\n成功提取了 {len(metrics)} 种指标")

    # 分析指标
    analysis = analyze_metrics(metrics)

    # 打印摘要
    print_summary(analysis)

    # 打印特定步数对比（200, 300, 以及检测到的最大步数）
    if analysis:
        # 找出所有检测到的步数
        all_steps = set()
        for data in analysis.values():
            for item in data['all_values']:
                if item['step'] is not None:
                    all_steps.add(item['step'])

        if all_steps:
            max_step = max(all_steps)
            # 选择几个关键点进行对比
            compare_steps = [100, 200, 300, 400, 500]
            compare_steps = [s for s in compare_steps if s <= max_step]
            if max_step not in compare_steps:
                compare_steps.append(max_step)

            print_step_comparison(analysis, sorted(compare_steps))

    # 自动诊断
    diagnose_issues(analysis)

    # 保存详细报告
    output_dir = Path(log_path).parent
    report_path = output_dir / "training_metrics_report.txt"
    csv_path = output_dir / "training_metrics.csv"

    save_detailed_report(analysis, str(report_path))
    export_to_csv(analysis, str(csv_path))

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"详细报告: {report_path}")
    print(f"CSV数据: {csv_path}")
    print("\n你可以用以下命令查看CSV数据:")
    print(f"  cat {csv_path}")
    print("\n或者导入到Excel/Python进行进一步分析")


if __name__ == "__main__":
    main()