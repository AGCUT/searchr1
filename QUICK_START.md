# TriviaQA Training Quick Start Guide

这是一个快速启动指南，帮助你使用 BM25 检索器和 Qwen2.5-3B-Instruct 模型在 TriviaQA 数据集上训练 Search-R1。

---

## 快速启动步骤（5 分钟）

### 前置要求

- ✅ 已安装 conda
- ✅ 有一张至少 16GB 显存的 GPU
- ✅ 至少 100GB 可用磁盘空间

---

## 第 1 步：环境安装（约 10 分钟）

打开命令行，执行以下命令：

```batch
REM 创建主训练环境
conda create -n searchr1 python=3.9 -y
conda activate searchr1
pip install torch==2.4.0
pip install vllm==0.6.3
cd D:\search-r1\Search-R1
pip install -e .

REM 创建检索器环境
conda create -n retriever python=3.10 -y
conda activate retriever
pip install transformers datasets pyserini uvicorn fastapi
```

---

## 第 2 步：下载 BM25 索引（约 20 分钟，取决于网速）

```batch
REM 下载 Wiki-18 BM25 索引和语料库
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir D:\search-r1\wiki_data
```

**如果下载失败**，可以使用镜像站：
```batch
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir D:\search-r1\wiki_data
```

---

## 第 3 步：处理 TriviaQA 数据（约 5 分钟）

```batch
conda activate searchr1
cd D:\search-r1\Search-R1
python scripts\data_process\triviaqa_search.py --local_dir .\data\triviaqa_search
```

**预期输出**：
```
TriviaQA dataset processed and saved to .\data\triviaqa_search
```

---

## 第 4 步：启动 BM25 检索服务（保持运行）

**打开新的命令行窗口**，执行：

```batch
conda activate retriever
cd D:\search-r1\Search-R1
retrieval_launch_bm25.bat
```

**预期输出**：
```
============================================
Starting BM25 Retrieval Server
============================================
...
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**⚠️ 保持这个窗口运行，不要关闭！**

---

## 第 5 步：开始训练（在另一个窗口）

**打开新的命令行窗口**，执行：

```batch
conda activate searchr1
cd D:\search-r1\Search-R1
train_triviaqa_qwen25_3b_ppo.bat
```

训练将开始，你会看到类似输出：
```
============================================
TriviaQA Training with Qwen2.5-3B-Instruct
============================================
...
Epoch 1/10 | Step 1/500 | Reward: 0.25 | KL: 0.001
```

---

## 训练配置说明

### 默认配置

- **模型**: Qwen2.5-3B-Instruct
- **检索器**: BM25 (CPU-based)
- **训练轮数**: 10 epochs
- **训练步数**: 500 steps
- **GPU 内存占用**: 约 12-14GB
- **预计训练时间**: 4-6 小时（单张 RTX 4090）

### 如果遇到显存不足

编辑 `train_triviaqa_qwen25_3b_ppo.bat`，修改以下参数：

```batch
REM 降低 GPU 内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 ^

REM 减小 batch size
data.train_batch_size=128 ^
actor_rollout_ref.actor.ppo_mini_batch_size=64 ^
actor_rollout_ref.actor.ppo_micro_batch_size=16 ^
```

---

## 监控训练进度

### 方法 1: 控制台输出

训练窗口会实时显示训练进度：
```
Epoch 1/10 | Step 10/500 | Reward: 0.35 | Loss: 0.25
```

### 方法 2: WandB（可选）

如果安装了 wandb：
```batch
conda activate searchr1
wandb login
```

然后访问 https://wandb.ai 查看训练曲线。

### 方法 3: 检查检查点

训练期间会自动保存检查点到：
```
D:\search-r1\Search-R1\checkpoints\triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em\
```

---

## 训练完成后

### 评估模型

创建评估脚本 `eval_triviaqa_checkpoint.bat`:

```batch
@echo off
set CHECKPOINT_DIR=./checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em/actor
set DATA_DIR=D:\search-r1\Search-R1\data\triviaqa_search

python -m verl.trainer.main_ppo ^
    data.val_files=%DATA_DIR%\test.parquet ^
    data.val_batch_size=128 ^
    actor_rollout_ref.model.path=%CHECKPOINT_DIR% ^
    +trainer.val_only=true ^
    max_turns=4 ^
    retriever.url="http://127.0.0.1:8000/retrieve" ^
    retriever.topk=3
```

运行评估：
```batch
eval_triviaqa_checkpoint.bat
```

---

## 故障排查

### 问题 1: "No module named 'pyserini'"
```batch
conda activate retriever
pip install pyserini
```

### 问题 2: "Java not found"
Pyserini 需要 Java，请安装 JDK 11 或更高版本：
- 下载：https://www.oracle.com/java/technologies/downloads/
- 或使用：`conda install openjdk -y`

### 问题 3: "Connection refused to http://127.0.0.1:8000"
确保 BM25 检索服务在运行：
```batch
REM 在新窗口
conda activate retriever
cd D:\search-r1\Search-R1
retrieval_launch_bm25.bat
```

### 问题 4: "CUDA out of memory"
降低 GPU 内存使用：
```batch
REM 编辑 train_triviaqa_qwen25_3b_ppo.bat
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 ^
data.train_batch_size=128 ^
```

### 问题 5: "Dataset not found"
重新运行数据处理：
```batch
conda activate searchr1
python scripts\data_process\triviaqa_search.py --local_dir .\data\triviaqa_search
```

---

## 高级配置

### 使用更大的模型

修改 `train_triviaqa_qwen25_3b_ppo.bat`:
```batch
REM 使用 Qwen2.5-7B-Instruct（需要 24GB+ 显存）
set BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### 调整训练步数

```batch
REM 更长的训练
trainer.total_epochs=20 ^
trainer.total_training_steps=1000 ^
```

### 调整检索参数

```batch
REM 返回更多文档
retriever.topk=5 ^

REM 允许更多搜索轮次
max_turns=6 ^
```

---

## 完整命令速查表

| 操作 | 命令 |
|------|------|
| 激活训练环境 | `conda activate searchr1` |
| 激活检索环境 | `conda activate retriever` |
| 启动 BM25 服务 | `retrieval_launch_bm25.bat` |
| 处理数据 | `python scripts\data_process\triviaqa_search.py` |
| 开始训练 | `train_triviaqa_qwen25_3b_ppo.bat` |
| 查看检查点 | `dir checkpoints\triviaqa-*` |

---

## 预期结果

### 训练输出示例

```
Epoch 1/10 | Step 100/500
- Average Reward: 0.45
- KL Divergence: 0.0012
- Actor Loss: 0.23
- Critic Loss: 0.18
- Learning Rate: 9.5e-7
```

### 最终性能（参考）

在 TriviaQA 测试集上的预期结果：
- **EM Score**: 40-50%（取决于训练时长）
- **平均搜索轮次**: 2-3
- **搜索准确率**: 70-80%

---

## 下一步

完成训练后，你可以：

1. **评估模型性能**: 在 TriviaQA 测试集上评估
2. **尝试其他数据集**: NQ、HotpotQA、PopQA
3. **优化检索器**: 尝试 E5 密集检索器
4. **模型调优**: 调整学习率、batch size 等超参数

---

## 需要帮助？

- 📖 完整文档: `TriviaQA_BM25_Qwen_Guide.md`
- 🐛 问题反馈: https://github.com/PeterGriffinJin/Search-R1/issues
- 💬 讨论区: https://github.com/PeterGriffinJin/Search-R1/discussions

---

**祝训练顺利！** 🚀
