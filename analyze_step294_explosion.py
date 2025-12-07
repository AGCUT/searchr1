#!/usr/bin/env python3
"""
深度分析 Step 294 梯度爆炸事件
"""

import wandb
import pandas as pd
import numpy as np

# 初始化 WandB API
api = wandb.Api()
RUN_PATH = "2630305490-nanjing-university/Search-R1-NQ-HotpotQA/kc8wjt1j"

print("正在从 WandB 获取数据...")
run = api.run(RUN_PATH)
history = run.history()

# 关注 step 280-305 的详细数据
critical_period = history[history['global_step'].between(280, 305)].copy()
critical_period = critical_period.sort_values('global_step')

print("\n" + "="*80)
print("Step 280-305 详细分析（梯度爆炸前后）")
print("="*80)

# 我们关心的指标
metrics = [
    'global_step',
    'actor/grad_norm',
    'critic/grad_norm',
    'actor/lr',
    'train/kl',
    'train/clip_frac',
    'actor/loss/policy_loss',
    'actor/loss/value_loss',
    'train/advantage_mean',
    'train/advantage_std',
    'train/reward_mean',
    'train/reward_std',
    'actor/approx_kl',
]

# 筛选存在的列
available = [m for m in metrics if m in critical_period.columns]

# 显示每一步的数据
for idx, row in critical_period.iterrows():
    step = int(row['global_step'])
    print(f"\n{'='*40}")
    print(f"Step {step}")
    print(f"{'='*40}")

    for metric in available:
        if pd.notna(row[metric]):
            value = row[metric]

            # 特殊标记异常值
            marker = ""
            if metric == 'actor/grad_norm':
                if value > 100:
                    marker = " ⚠️ 异常高！"
                elif value > 20:
                    marker = " ⚠️ 偏高"
            elif metric == 'train/kl':
                if value > 0.1:
                    marker = " ⚠️ KL过大！"
                elif value > 0.05:
                    marker = " ⚠️ KL偏高"
            elif metric == 'train/clip_frac':
                if value > 0.3:
                    marker = " ⚠️ 裁剪过多！"
                elif value > 0.2:
                    marker = " ⚠️ 裁剪较多"

            # 格式化输出
            if isinstance(value, (int, np.integer)):
                print(f"  {metric:30s}: {value}{marker}")
            elif isinstance(value, (float, np.floating)):
                if abs(value) < 0.001:
                    print(f"  {metric:30s}: {value:.6e}{marker}")
                else:
                    print(f"  {metric:30s}: {value:.6f}{marker}")

# 寻找异常模式
print("\n" + "="*80)
print("异常模式分析")
print("="*80)

if 'actor/grad_norm' in available:
    grad_norms = critical_period['actor/grad_norm'].dropna()

    print(f"\n梯度范数统计:")
    print(f"  最小值: {grad_norms.min():.2f}")
    print(f"  最大值: {grad_norms.max():.2f}")
    print(f"  平均值: {grad_norms.mean():.2f}")
    print(f"  标准差: {grad_norms.std():.2f}")

    # 找到爆炸点
    explosion_steps = grad_norms[grad_norms > 1000]
    if not explosion_steps.empty:
        print(f"\n  发现梯度爆炸 (>1000) 的步数:")
        for idx in explosion_steps.index:
            step = int(critical_period.loc[idx, 'global_step'])
            grad = critical_period.loc[idx, 'actor/grad_norm']
            print(f"    Step {step}: grad_norm = {grad:.2f}")

    # 计算步间增长率
    print(f"\n  步间增长率:")
    for i in range(len(grad_norms) - 1):
        curr_step = critical_period.iloc[i]['global_step']
        next_step = critical_period.iloc[i+1]['global_step']
        curr_grad = grad_norms.iloc[i]
        next_grad = grad_norms.iloc[i+1]

        if curr_grad > 0:
            growth_rate = (next_grad - curr_grad) / curr_grad * 100
            if abs(growth_rate) > 100:  # 增长超过100%
                print(f"    Step {int(curr_step)}→{int(next_step)}: {growth_rate:+.1f}% "
                      f"({curr_grad:.2f} → {next_grad:.2f}) ⚠️")

# 分析 KL 散度
if 'train/kl' in available:
    print(f"\n\nKL 散度分析:")
    kl_values = critical_period['train/kl'].dropna()
    print(f"  最小值: {kl_values.min():.6f}")
    print(f"  最大值: {kl_values.max():.6f}")
    print(f"  平均值: {kl_values.mean():.6f}")

    high_kl = kl_values[kl_values > 0.05]
    if not high_kl.empty:
        print(f"\n  发现高 KL 散度 (>0.05) 的步数:")
        for idx in high_kl.index:
            step = int(critical_period.loc[idx, 'global_step'])
            kl = critical_period.loc[idx, 'train/kl']
            print(f"    Step {step}: KL = {kl:.6f}")

# 分析 clip_frac
if 'train/clip_frac' in available:
    print(f"\n\nPPO 裁剪比例分析:")
    clip_values = critical_period['train/clip_frac'].dropna()
    print(f"  最小值: {clip_values.min():.4f}")
    print(f"  最大值: {clip_values.max():.4f}")
    print(f"  平均值: {clip_values.mean():.4f}")
    print(f"\n  说明: clip_frac 越高，说明策略更新被 PPO 裁剪得越多")
    print(f"       正常范围: 0.0-0.3，超过 0.3 说明策略更新过于激进")

# 分析 reward
if 'train/reward_mean' in available:
    print(f"\n\nReward 分析:")
    reward_mean = critical_period['train/reward_mean'].dropna()
    reward_std = critical_period['train/reward_std'].dropna() if 'train/reward_std' in available else None

    print(f"  Reward 均值范围: {reward_mean.min():.4f} ~ {reward_mean.max():.4f}")
    if reward_std is not None:
        print(f"  Reward 标准差范围: {reward_std.min():.4f} ~ {reward_std.max():.4f}")

        high_std = reward_std[reward_std > reward_std.mean() + 2*reward_std.std()]
        if not high_std.empty:
            print(f"\n  发现异常高的 Reward 方差:")
            for idx in high_std.index:
                step = int(critical_period.loc[idx, 'global_step'])
                std = critical_period.loc[idx, 'train/reward_std']
                print(f"    Step {step}: reward_std = {std:.4f}")

# 关联分析
print("\n" + "="*80)
print("关联分析：梯度爆炸与其他指标的关系")
print("="*80)

if 'actor/grad_norm' in available:
    # 找到梯度最大的那一步
    max_grad_idx = critical_period['actor/grad_norm'].idxmax()
    explosion_step = int(critical_period.loc[max_grad_idx, 'global_step'])

    print(f"\n梯度爆炸发生在 Step {explosion_step}")
    print(f"该步骤的所有指标:")
    print("-" * 40)

    for metric in available:
        if pd.notna(critical_period.loc[max_grad_idx, metric]):
            value = critical_period.loc[max_grad_idx, metric]
            if isinstance(value, (float, np.floating)):
                if abs(value) < 0.001:
                    print(f"  {metric:30s}: {value:.6e}")
                else:
                    print(f"  {metric:30s}: {value:.6f}")
            else:
                print(f"  {metric:30s}: {value}")

# 保存详细数据
output_file = "D:/search-r1/step294_explosion_details.csv"
critical_period.to_csv(output_file, index=False)
print(f"\n详细数据已保存到: {output_file}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)