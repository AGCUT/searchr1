# Search-R1 项目笔记（简历 & 面试专用）

## 一、项目概述

### 项目定位

Search-R1 是一个**端到端强化学习训练框架**，核心目标是让大语言模型学会：
- **何时搜索**：判断当前是否需要外部知识
- **搜索什么**：生成有效的检索查询
- **如何推理**：整合多轮检索结果进行推理
- **何时回答**：判断信息是否充足并给出最终答案

### 技术栈

| 组件 | 技术选型 |
|------|---------|
| 基座模型 | Qwen2.5-3B-Instruct |
| RL 框架 | veRL (PPO) |
| 推理加速 | vLLM |
| 检索器 | BM25 (Pyserini) / BM25 + Rerank |
| 知识库 | Wikipedia 2018 (~2100万文档) |
| 实验追踪 | Weights & Biases |

---

## 二、核心技术点

### 1. 端到端强化学习训练

与传统 RAG 的区别：

| 方面 | 传统 RAG | Search-R1 (本项目) |
|------|---------|-------------------|
| 检索决策 | 规则触发/固定流程 | 模型自主学习何时检索 |
| 查询生成 | 直接用问题或规则改写 | 模型学习生成最优查询 |
| 训练信号 | 检索相似度 | **最终答案正确性** |
| 优化目标 | 分模块优化 | **端到端优化整体效果** |

**核心创新**：将检索决策融入强化学习的 action space，用最终答案的正确性（EM reward）反向优化整个「推理-检索-推理」过程。

### 2. 多轮交互式推理

模型输出格式：
```
<think> 推理过程... </think>
<search> 检索查询 </search>
<information> 检索结果（环境返回） </information>
<think> 继续推理... </think>
<answer> 最终答案 </answer>
```

训练模型学会：
- 在 `<think>` 中进行 Chain-of-Thought 推理
- 在 `<search>` 中生成高质量查询
- 分析 `<information>` 中的检索结果
- 在 `<answer>` 中给出精确答案

### 3. PPO 训练配置与调优

核心超参数设计：

| 参数 | 设置 | 原因 |
|------|------|------|
| Actor LR | 2e-7 ~ 5e-7 | 策略更新需保守，避免崩溃 |
| Critic LR | 2e-6 ~ 5e-6 | 10倍于Actor，快速学习value |
| KL Coef | 0.005 ~ 0.03 | 约束策略变化幅度 |
| Grad Clip | 0.3 ~ 1.0 | 防止梯度爆炸 |
| Batch Size | 504 (需被GPU数整除) | PPO 稳定性要求 |

**关键经验**：Actor LR 和 Critic LR 的比例（通常 1:10）比绝对值更重要；KL 惩罚与 LR 是对抗关系，需要平衡。

### 4. 检索系统设计

支持多种检索策略：

```
BM25 检索（当前使用）:
Query → BM25 索引 → Top-3 文档 → 返回结果

BM25 + Rerank（可选升级）:
Query → BM25 召回 Top-10 → Cross-Encoder 重排 → Top-3 文档
```

**检索器选择对训练的影响**：
- 训练和推理应使用相同的检索器
- 模型会适应特定检索器返回结果的"风格"
- 换检索器相当于改变环境，可能需要重新适应

---

## 三、训练数据

### 数据集组成

| 数据集 | 类型 | 特点 |
|--------|------|------|
| NQ (Natural Questions) | 单跳问答 | Google 真实搜索日志 |
| HotpotQA | 多跳推理 | 需要整合多文档信息 |
| PopQA | 事实问答 | 包含实体流行度信息 |

### 数据格式

```json
{
  "question": "In what country is Mangalore University?",
  "golden_answers": ["India", "Bharat", "Republic of India", ...],
  "data_source": "popqa",
  "prompt": [{"role": "user", "content": "Answer the given question..."}]
}
```

### 训练配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Batch Size | 504 | 需被 GPU 数量整除 |
| Eval Frequency | 50 步 | 及时发现问题 |
| Save Frequency | 50 步 | 保存最佳 checkpoint |
| Max Turns | 4 | 最多 4 轮检索 |
| Top-k | 3 | 每次返回 3 条文档 |

---

## 四、奖励函数设计

### 当前奖励：Exact Match (EM)

```python
reward = 1.0 if answer in golden_answers else 0.0
```

**特点**：
- 简单直接
- 支持多个标准答案别名
- 端到端优化最终效果

### 高级奖励设计（可扩展）

```python
def compute_reward(trajectory, golden_answer):
    reward = 0.0

    # 1. 基础分：答案正确性
    if is_correct(trajectory, golden_answer):
        reward += 1.0

    # 2. 搜索行为奖惩
    num_searches = count_searches(trajectory)
    if num_searches == 0:
        reward -= 0.2  # 惩罚不搜索就回答
    elif num_searches > 3:
        reward -= 0.05 * (num_searches - 3)  # 惩罚过多搜索

    # 3. 格式规范奖励
    if has_valid_format(trajectory):
        reward += 0.1

    return reward
```

### 更改 Reward 的高效方法

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| Critic Warmup | 让 Critic 先适应新 reward | 小幅修改 reward |
| GRPO | 不需要 Critic，用组内相对奖励 | 频繁改 reward |
| 渐进式调整 | 逐步增加 reward 复杂度 | 大幅修改 reward |

---

## 五、BadCase 分析与解决

### BadCase 分类

| 类型 | 现象 | 解决方案 |
|------|------|---------|
| 不搜索就答 | 直接从记忆回答 | reward 惩罚 / prompt 约束 |
| 查询质量差 | 生成无效查询词 | 升级到 Rerank / 查询奖励 |
| 推理错误 | 搜到了但理解错 | 增大 KL 惩罚 |
| 搜索过多 | 达到 max_turns 限制 | 效率惩罚 |
| 知识库缺失 | 答案不在语料库中 | 扩充知识库 |

### 系统性解决思路

1. **分析 BadCase 分布**：确定主要问题类型
2. **针对性调整**：修改 reward / prompt / 检索器
3. **迭代验证**：重新训练并对比效果

---

## 六、与前沿技术的关系

### 与 RAG 优化方法的对比

面试常见问题："如何提升 RAG 效果？"

| 方法 | 描述 | 本项目实现 |
|------|------|-----------|
| 意图识别 + 查询规划 | 大模型规划检索 | 模型生成 `<search>` 查询 |
| 混合检索 + 精排 | BM25 + Dense + Rerank | 已有 BM25 + Rerank |
| DPO 训练规划器 | 收集正负样本训练 | PPO 在线学习（更强） |
| GRPO + 回复奖励 | 用最终效果优化检索 | **正是本项目核心** |

### 本项目的技术优势

> **核心亮点**：将检索和推理的监督信号从"文本相似度"转变为"最终回答正确性"，实现真正的端到端优化。

这与最新的 RAG 优化思路（GRPO + 回复模型奖励）不谋而合，是比较前沿的技术方向。

---

## 七、训练经验总结

### 遇到的问题及解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 梯度爆炸 (Step 294) | LR 过大 / KL 约束不足 | 降低 LR + 增大 KL Coef |
| Batch Size 报错 | 不能被 GPU 数整除 | 512 → 504 (6 GPU) |
| /tmp 磁盘满 | Ray 临时文件 | 设置 RAY_TMPDIR |
| Checkpoint 加载失败 | 路径结构不对 | 确认 actor/critic 目录 |

### 关键调参经验

1. **LR 与 KL 的平衡**：
   - LR 大 + KL 小 → 策略剧变，容易崩溃
   - LR 小 + KL 大 → 几乎不更新，训练太慢
   - 监控 `kl_divergence`：理想范围 0.01 ~ 0.1

2. **从 Checkpoint 恢复**：
   - 设置 `val_before_train=true` 验证 checkpoint 正确性
   - 使用更保守的 LR（原来的 20%~50%）
   - `critic_warmup=0`（已经预热过）

3. **训练稳定性**：
   - 频繁保存 checkpoint（每 50 步）
   - 监控 `grad_norm`：应 < 10，> 100 预示崩溃

---

## 八、简历描述模板

### 中文版

**项目名称**：Search-R1 - 基于强化学习的检索增强推理系统

**项目描述**：
- 基于 veRL 框架实现端到端 PPO 训练，让 Qwen2.5-3B 模型学会在推理过程中自主决策何时检索、检索什么、如何整合信息并给出答案
- 使用 NQ + HotpotQA 混合数据集训练，BM25 检索 Wikipedia 知识库，实现单跳和多跳问答能力
- 设计多维度奖励函数，将监督信号从文本相似度转变为最终答案正确性，实现检索策略和推理能力的联合优化
- 解决 PPO 训练中的梯度爆炸问题，通过调整 LR/KL 比例、梯度裁剪等手段实现稳定训练
- 分析 BadCase 分布，针对"不搜索就答"、"查询质量差"等问题设计针对性优化策略

### 英文版

**Project**: Search-R1 - Reinforcement Learning for Retrieval-Augmented Reasoning

**Description**:
- Implemented end-to-end PPO training using veRL framework, enabling Qwen2.5-3B to autonomously decide when to search, what to search, and how to synthesize retrieved information
- Trained on NQ + HotpotQA mixed dataset with BM25 retrieval over Wikipedia corpus for both single-hop and multi-hop QA
- Designed reward functions that shift supervision from text similarity to final answer correctness, jointly optimizing retrieval strategy and reasoning ability
- Resolved gradient explosion issues through LR/KL ratio tuning and gradient clipping for stable PPO training
- Analyzed BadCase distribution and developed targeted optimization strategies for issues like "answering without search" and "poor query quality"

---

## 九、面试 Q&A 准备

### Q1: PPO 训练的是什么能力？

模型学习四种能力：
1. **何时搜索**：判断是否需要外部知识
2. **搜索什么**：生成有效的检索查询
3. **如何推理**：在 `<think>` 中整合信息
4. **何时回答**：判断信息充足后给出答案

### Q2: 为什么用 EM 作为奖励而不是检索相似度？

因为我们关心的是**最终效果**：
- 传统 RAG 优化检索相似度，但相似不等于有用
- 用 EM 奖励实现端到端优化，让检索真正服务于回答
- 这与最新的 RAG 优化思路（GRPO + 回复奖励）一致

### Q3: 换检索器会影响效果吗？

会。模型在训练时会适应特定检索器的"风格"：
- 训练用 BM25，推理也应该用 BM25
- 如果要升级到 Rerank，建议在新检索器上继续训练几百步

### Q4: 如何解决 BadCase？

1. 先分析 BadCase 类型分布
2. 针对主要问题调整：
   - 不搜索 → reward 惩罚
   - 查询差 → 升级检索器
   - 推理错 → 增大 KL 约束
3. 迭代训练验证效果

### Q5: 更改 reward 后需要重新训练 Critic 吗？

不一定需要从头训练：
- 小幅改动：增加 `critic_warmup=50` 让 Critic 适应
- 大幅改动：考虑用 GRPO（不需要 Critic）
- 或者渐进式调整 reward

---

## 十、后续优化方向

1. **检索升级**：BM25 → BM25 + Rerank → 混合检索
2. **奖励精细化**：加入查询质量、搜索效率等维度
3. **课程学习**：从简单问题逐步过渡到困难问题
4. **模型规模**：扩展到 7B/14B 模型
5. **多轮对话**：迁移到对话式问答数据集
