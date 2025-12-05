# NQ + HotpotQA Training with TriviaQA Evaluation - Linux 4x A800 Guide

完整的官方配置方案，使用 **NQ + HotpotQA** 训练，在 **TriviaQA** 等 7 个数据集上评估。

---

## 📋 官方配置说明

本指南严格参考 Search-R1 官方脚本：
- **数据处理**: `scripts/nq_hotpotqa/data_process.sh`
- **训练脚本**: `scripts/nq_hotpotqa/v0.2/train_ppo.sh`
- **评估脚本**: `scripts/nq_hotpotqa/evaluate.sh`

### 关键区别

| 项目 | 官方配置 | 你的配置 |
|------|---------|---------|
| **训练数据** | NQ + HotpotQA | ✅ NQ + HotpotQA |
| **测试数据** | 7 个数据集 | ✅ 7 个数据集（重点看 TriviaQA） |
| **检索器** | E5 (dense) | **BM25 (sparse)** |
| **GPU 数量** | 8 卡 | **4 卡 (GPU 4,5,6,7)** |
| **模型** | Qwen2.5-7B | **Qwen2.5-3B-Instruct** |

---

## 🚀 快速启动（5 步）

### **第 1 步：环境安装**

```bash
# SSH 登录服务器
ssh your_username@server

# 创建训练环境
conda create -n searchr1 python=3.9 -y
conda activate searchr1
pip install torch==2.4.0
pip install vllm==0.6.3
cd /path/to/Search-R1
pip install -e .
pip install wandb  # 用于训练可视化

# 创建检索环境
conda create -n retriever python=3.10 -y
conda activate retriever
pip install transformers datasets pyserini uvicorn fastapi
conda install openjdk=11 -y  # BM25 需要 Java
```

---

### **第 2 步：下载 BM25 索引**

```bash
# 设置保存路径
export SAVE_PATH=/path/to/wiki_data

# 下载索引
huggingface-cli download PeterJinGo/wiki-18-bm25-index \
    --repo-type dataset \
    --local-dir $SAVE_PATH

# 验证下载
ls -lh $SAVE_PATH
# 应该看到:
#   - bm25/ (目录)
#   - wiki-18.jsonl (文件)
```

---

### **第 3 步：处理数据（自动下载）**

```bash
cd /path/to/Search-R1

# 给脚本添加执行权限
chmod +x process_nq_hotpotqa_data.sh

# 运行数据处理脚本
conda activate searchr1
bash process_nq_hotpotqa_data.sh
```

**这个脚本会做什么？**（参考官方 `data_process.sh`）

1. **下载并处理训练数据**:
   - 从 `RUC-NLPIR/FlashRAG_datasets` 下载 **NQ** 和 **HotpotQA**
   - 合并为 `data/nq_hotpotqa_train/train.parquet`
   - NQ 约 79k + HotpotQA 约 90k = **约 170k 训练样本**

2. **下载并处理测试数据**:
   - 下载 7 个数据集: **NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA, Musique, Bamboogle**
   - 合并为 `data/nq_hotpotqa_train/test.parquet`
   - 总共约 **50k 测试样本**

**预期输出**:
```
============================================
Data Processing Completed!
============================================

Files created:
  1. data/nq_hotpotqa_train/train.parquet (Training: NQ + HotpotQA)
  2. data/nq_hotpotqa_train/test.parquet (Test: 7 datasets)

Training data statistics:
  Total samples: 169837
  Breakdown by dataset:
    - hotpotqa: 90447
    - nq: 79390

Test data statistics:
  Total samples: 51483
  Breakdown by dataset:
    - triviaqa: 11313  ← 你关注的数据集
    - nq: 3610
    - popqa: 14267
    - hotpotqa: 7405
    - 2wikimultihopqa: 12576
    - musique: 2417
    - bamboogle: 125
============================================
```

---

### **第 4 步：启动 BM25 检索服务**

```bash
# 修改检索服务脚本中的路径
nano retrieval_launch_bm25.sh
# 修改第 4 行: file_path=/your/actual/path/to/wiki_data

# 给脚本添加执行权限
chmod +x retrieval_launch_bm25.sh

# 在 tmux 中启动服务
tmux new -s bm25
conda activate retriever
bash retrieval_launch_bm25.sh
# 按 Ctrl+B 然后 D 分离

# 验证服务运行
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "topk": 3}'
```

---

### **第 5 步：启动 4 卡训练**

```bash
# 给训练脚本添加执行权限
chmod +x train_nq_hotpotqa_qwen25_3b_4gpu.sh

# （可选）登录 WandB 以可视化训练
conda activate searchr1
wandb login

# 在 tmux 中启动训练
tmux new -s training
conda activate searchr1
bash train_nq_hotpotqa_qwen25_3b_4gpu.sh
# 按 Ctrl+B 然后 D 分离

# 监控训练进度
tail -f nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log
```

---

## 📊 官方训练配置详解

### 训练参数（参考 v0.2 配置）

```bash
# 数据配置
data.train_batch_size=512          # 总 batch size（4 卡分摊）
data.val_batch_size=256
data.max_prompt_length=4096
data.max_response_length=500
data.max_start_length=2048
data.max_obs_length=500

# Actor 配置
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size=64
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# Critic 配置
critic.optim.lr=1e-5
critic.optim.lr_warmup_steps_ratio=0.015
critic.ppo_micro_batch_size=8

# 训练步数
trainer.total_epochs=15
trainer.total_training_steps=1005    # 官方配置
trainer.save_freq=100
trainer.test_freq=100

# 检索配置
max_turns=4
retriever.topk=3
```

### 显存使用（4 卡 A800）

- **每卡显存**: 约 15-18GB / 80GB
- **总显存**: 约 60-72GB
- **剩余显存**: 约 248GB（充足）

### 训练时间估算

- **单个 epoch**: 约 20-30 分钟（4 卡 A800）
- **总训练时间**: 约 5-7 小时（15 epochs）
- **比单卡快**: 约 3.5-4 倍

---

## 🔍 训练监控

### 方法 1: 查看日志

```bash
# 实时查看训练日志
tail -f nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log

# 查看关键指标
tail -f *.log | grep -E "Epoch|Reward|Loss|EM"
```

### 方法 2: WandB 可视化

```bash
# 登录 WandB
wandb login

# 训练时自动上传到 WandB
# 访问: https://wandb.ai
```

### 方法 3: 监控 GPU

```bash
# 实时监控 GPU 4,5,6,7
watch -n 1 'nvidia-smi -i 4,5,6,7'

# 使用 gpustat（更美观）
pip install gpustat
watch -n 1 'gpustat -i 4,5,6,7'
```

---

## ✅ 训练完成后：评估

### 在 TriviaQA 上评估

```bash
# 给评估脚本添加执行权限
chmod +x eval_triviaqa.sh

# 修改评估脚本中的模型路径
nano eval_triviaqa.sh
# 确保第 15 行指向正确的检查点:
# export BASE_MODEL="verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor"

# 运行评估
conda activate searchr1
bash eval_triviaqa.sh
```

### 查看评估结果

```bash
# 查看完整结果
cat evaluation_results.log

# 提取 TriviaQA 分数
grep -i "triviaqa" evaluation_results.log

# 提取所有数据集分数
grep -E "nq|triviaqa|popqa|hotpotqa|2wikimultihopqa|musique|bamboogle" evaluation_results.log | grep "EM Score"
```

### 预期 TriviaQA 结果

根据官方论文，在 TriviaQA 上的 EM 得分：

| 模型 | TriviaQA EM | 检索器 |
|------|------------|--------|
| Qwen2.5-3B (官方) | ~30-35% | E5 (dense) |
| Qwen2.5-7B (官方) | ~38-45% | E5 (dense) |
| **你的模型** | **~28-35%** (预期) | BM25 (sparse) |

**注意**: 使用 BM25 可能比 E5 稍低 2-5 个百分点，但训练更快且不需要 GPU 检索。

---

## 📈 官方数据集说明

### 训练数据（2 个数据集）

| 数据集 | 样本数 | 类型 | 说明 |
|--------|-------|------|------|
| **NQ** | ~79k | 单跳 QA | Google 搜索查询 |
| **HotpotQA** | ~90k | 多跳 QA | 需要多步推理 |

### 测试数据（7 个数据集）

| 数据集 | 样本数 | 难度 | 说明 |
|--------|-------|------|------|
| **TriviaQA** | ~11k | 中 | 琐事问答，你关注的数据集 |
| NQ | ~3.6k | 中 | 单跳问答 |
| PopQA | ~14k | 易 | 流行问答 |
| HotpotQA | ~7.4k | 难 | 多跳推理 |
| 2WikiMultihopQA | ~12k | 难 | 维基百科多跳 |
| Musique | ~2.4k | 难 | 复杂多跳推理 |
| Bamboogle | ~125 | 极难 | 困难问答 |

---

## 🔧 常见问题排查

### 问题 1: 数据下载失败

**现象**: `ConnectionError` 或 `HTTPError`

**解决**:
```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载数据集（如果自动下载失败）
huggingface-cli download --repo-type dataset \
    PeterJinGo/nq_hotpotqa_train \
    --local-dir ./data/nq_hotpotqa_train
```

### 问题 2: 只有一张 GPU 在工作

**检查**:
```bash
# 确认 GPU 配置
echo $CUDA_VISIBLE_DEVICES
# 应该输出: 4,5,6,7

# 检查训练脚本
grep "n_gpus_per_node" train_nq_hotpotqa_qwen25_3b_4gpu.sh
# 应该是: trainer.n_gpus_per_node=4

# 检查并行模式
grep "tensor_model_parallel_size" train_nq_hotpotqa_qwen25_3b_4gpu.sh
# 应该是: tensor_model_parallel_size=1 (数据并行)
```

### 问题 3: OOM (显存不足)

**解决**:
```bash
# 编辑训练脚本，降低以下参数
nano train_nq_hotpotqa_qwen25_3b_4gpu.sh

# 修改:
data.train_batch_size=256 \          # 从 512 降到 256
actor_rollout_ref.actor.ppo_mini_batch_size=128 \  # 从 256 降到 128
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \  # 从 0.6 降到 0.5
```

### 问题 4: BM25 服务连接失败

**排查**:
```bash
# 检查服务状态
tmux attach -t bm25

# 检查端口
netstat -tuln | grep 8000

# 测试连接
curl http://127.0.0.1:8000/retrieve
```

### 问题 5: Java not found

**解决**:
```bash
conda activate retriever
conda install openjdk=11 -y
java -version
```

---

## 📝 完整命令速查表

| 操作 | 命令 |
|------|------|
| 激活训练环境 | `conda activate searchr1` |
| 激活检索环境 | `conda activate retriever` |
| 处理数据 | `bash process_nq_hotpotqa_data.sh` |
| 启动 BM25 服务 | `tmux new -s bm25; bash retrieval_launch_bm25.sh` |
| 启动训练 | `tmux new -s training; bash train_nq_hotpotqa_qwen25_3b_4gpu.sh` |
| 监控训练 | `tail -f *.log` |
| 监控 GPU | `watch -n 1 nvidia-smi -i 4,5,6,7` |
| 评估模型 | `bash eval_triviaqa.sh` |
| 查看检查点 | `ls -lh verl_checkpoints/` |

---

## 📂 文件结构

```
Search-R1/
├── data/
│   └── nq_hotpotqa_train/
│       ├── train.parquet          (NQ + HotpotQA 训练数据)
│       └── test.parquet            (7 个数据集测试数据)
│
├── verl_checkpoints/
│   └── nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/
│       ├── actor/                  (Actor 模型检查点)
│       ├── critic/                 (Critic 模型检查点)
│       └── step_100/, step_200/... (定期保存的检查点)
│
├── 脚本文件:
├── process_nq_hotpotqa_data.sh    (数据处理)
├── retrieval_launch_bm25.sh       (BM25 服务)
├── train_nq_hotpotqa_qwen25_3b_4gpu.sh  (训练)
├── eval_triviaqa.sh               (评估)
│
└── 日志文件:
    ├── nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em.log  (训练日志)
    └── evaluation_results.log      (评估结果)
```

---

## 🎯 与官方配置的对比

| 配置项 | 官方 | 本方案 | 说明 |
|--------|------|--------|------|
| 训练数据 | NQ + HotpotQA | ✅ 相同 | 约 170k 样本 |
| 测试数据 | 7 个数据集 | ✅ 相同 | 包含 TriviaQA |
| 检索器 | E5 (dense) | ⚠️ BM25 (sparse) | BM25 更快但可能略低 |
| GPU 数量 | 8 卡 | ⚠️ 4 卡 | batch size 减半 |
| 模型 | Qwen2.5-7B | ⚠️ Qwen2.5-3B-Instruct | 更小但训练更快 |
| 训练步数 | 1005 | ✅ 1005 | 相同 |
| 训练轮数 | 15 | ✅ 15 | 相同 |
| 学习率 | 1e-6 | ✅ 1e-6 | 相同 |
| 检索轮次 | 4 | ✅ 4 | 相同 |

---

## 🚀 预期训练流程

### 时间线

```
0:00    - 启动训练
0:05    - 首次验证（val_before_train=true）
0:10    - 开始第 1 个 epoch
0:30    - 完成第 1 个 epoch
1:40    - 完成 Step 100（保存检查点）
3:20    - 完成 Step 200（保存检查点）
5:00    - 完成训练（Step 1005）
5:30    - 运行评估
```

### 训练曲线（预期）

```
Epoch 1:  Reward ~0.25, EM ~25%
Epoch 5:  Reward ~0.35, EM ~35%
Epoch 10: Reward ~0.42, EM ~42%
Epoch 15: Reward ~0.48, EM ~48%  ← NQ+HotpotQA 上的表现
```

### TriviaQA 评估（预期）

```
TriviaQA EM Score: 30-35%  ← 你关注的指标
NQ EM Score: 45-50%
HotpotQA EM Score: 38-43%
Overall Average: ~40%
```

---

## 📚 参考资料

- **官方数据集**: https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
- **官方模型**: https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo
- **论文 v0.2**: https://arxiv.org/abs/2503.09516
- **WandB 日志**: https://wandb.ai/peterjin/Search-R1-v0.2

---

**祝训练顺利！** 🎉

如果遇到任何问题，请参考常见问题部分或查看官方文档。
