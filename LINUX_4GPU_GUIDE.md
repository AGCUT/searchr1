# TriviaQA Training on Linux with 4x A800 GPUs (GPU 4,5,6,7)

完整的 Linux 服务器训练指南，使用 4 张 A800 GPU (编号 4,5,6,7)。

---

## 📋 系统要求

- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+)
- **GPU**: 4x A800 80GB (GPU 4,5,6,7)
- **CUDA**: 11.8 或更高
- **Python**: 3.9 或 3.10
- **磁盘空间**: 至少 200GB

---

## 🚀 快速启动（5 步）

### 第 1 步：环境安装

```bash
# 创建主训练环境
conda create -n searchr1 python=3.9 -y
conda activate searchr1
pip install torch==2.4.0
pip install vllm==0.6.3
cd /path/to/Search-R1  # 修改为你的实际路径
pip install -e .

# 创建检索器环境
conda create -n retriever python=3.10 -y
conda activate retriever
pip install transformers datasets pyserini uvicorn fastapi
```

**检查 CUDA 版本**：
```bash
nvidia-smi
nvcc --version
```

---

### 第 2 步：下载 BM25 索引

```bash
# 设置保存路径（修改为你的实际路径）
export SAVE_PATH=/path/to/wiki_data

# 下载 BM25 索引和语料库
huggingface-cli download PeterJinGo/wiki-18-bm25-index \
    --repo-type dataset \
    --local-dir $SAVE_PATH
```

**如果下载速度慢，使用国内镜像**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download PeterJinGo/wiki-18-bm25-index \
    --repo-type dataset \
    --local-dir $SAVE_PATH
```

下载完成后检查文件：
```bash
ls -lh $SAVE_PATH
# 应该看到:
# - bm25/ (目录)
# - wiki-18.jsonl (文件)
```

---

### 第 3 步：处理 TriviaQA 数据

```bash
conda activate searchr1
cd /path/to/Search-R1

# 运行数据处理脚本
bash process_triviaqa_data.sh
```

**预期输出**：
```
============================================
Processing TriviaQA Dataset
============================================
Output directory: ./data/triviaqa_search
============================================
Loading dataset...
Processing train split...
Processing test split...
============================================
Data processing completed!
Files saved to: ./data/triviaqa_search
- train.parquet
- test.parquet
============================================
```

**验证数据**：
```bash
ls -lh ./data/triviaqa_search/
# 应该看到:
# - train.parquet
# - test.parquet
```

---

### 第 4 步：启动 BM25 检索服务

**修改配置**：
```bash
# 编辑 retrieval_launch_bm25.sh
nano retrieval_launch_bm25.sh

# 修改第 4 行为你的实际路径:
file_path=/your/actual/path/to/wiki_data
```

**启动服务（在 tmux 或 screen 会话中）**：
```bash
# 方法 1: 使用 tmux
tmux new -s bm25
conda activate retriever
bash retrieval_launch_bm25.sh
# 按 Ctrl+B 然后 D 分离会话

# 方法 2: 使用 screen
screen -S bm25
conda activate retriever
bash retrieval_launch_bm25.sh
# 按 Ctrl+A 然后 D 分离会话

# 方法 3: 使用 nohup
conda activate retriever
nohup bash retrieval_launch_bm25.sh > bm25.log 2>&1 &
```

**验证服务运行**：
```bash
# 检查端口
netstat -tuln | grep 8000
# 或
ss -tuln | grep 8000

# 测试 API
curl -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "topk": 3}'
```

---

### 第 5 步：启动 4 卡训练

**修改训练脚本配置**：
```bash
# 编辑 train_triviaqa_qwen25_3b_4gpu.sh
nano train_triviaqa_qwen25_3b_4gpu.sh

# 修改第 11 行的数据路径:
DATA_DIR=/your/actual/path/to/Search-R1/data/triviaqa_search
```

**给脚本添加执行权限**：
```bash
chmod +x train_triviaqa_qwen25_3b_4gpu.sh
chmod +x retrieval_launch_bm25.sh
chmod +x process_triviaqa_data.sh
```

**启动训练**：
```bash
# 确保 BM25 服务正在运行
# 启动训练（推荐在 tmux 中运行）
tmux new -s training
conda activate searchr1
bash train_triviaqa_qwen25_3b_4gpu.sh

# 分离 tmux: Ctrl+B 然后 D
# 重新连接: tmux attach -t training
```

**使用 nohup 后台训练**：
```bash
conda activate searchr1
nohup bash train_triviaqa_qwen25_3b_4gpu.sh > training.log 2>&1 &

# 查看实时日志
tail -f training.log
```

---

## ⚙️ 4 卡训练配置详解

### GPU 分配

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7  # 使用 GPU 4,5,6,7
```

**验证 GPU 可见性**：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -c "import torch; print(torch.cuda.device_count())"
# 应该输出: 4
```

### 关键训练参数（4 卡 A800 优化）

| 参数 | 值 | 说明 |
|-----|-----|------|
| `TRAIN_BATCH_SIZE` | 1024 | 总 batch size（4 卡共享）|
| `PPO_MINI_BATCH_SIZE` | 512 | PPO mini batch |
| `PPO_MICRO_BATCH_SIZE` | 64 | 每卡 micro batch |
| `GPU_MEMORY_UTIL` | 0.7 | A800 80GB 可设高一些 |
| `N_GPUS_PER_NODE` | 4 | 使用 4 张 GPU |
| `ACTOR_LR` | 1e-6 | Actor 学习率 |
| `CRITIC_LR` | 1e-5 | Critic 学习率 |

### 显存使用估算

- **每卡显存占用**: 约 18-22GB
- **总显存占用**: 约 72-88GB
- **A800 80GB**: 完全够用，还有余量

### 如果显存不足

编辑 `train_triviaqa_qwen25_3b_4gpu.sh`，调整这些参数：

```bash
# 降低 batch size
TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=32

# 降低 GPU 内存使用率
GPU_MEMORY_UTIL=0.5
```

---

## 📊 监控训练进度

### 方法 1: 实时查看日志

```bash
# 如果使用 tmux
tmux attach -t training

# 如果使用 nohup
tail -f training.log

# 只看关键指标
tail -f training.log | grep -E "Epoch|Reward|Loss"
```

### 方法 2: 监控 GPU 使用

```bash
# 实时监控所有 GPU
watch -n 1 nvidia-smi

# 只监控 GPU 4,5,6,7
watch -n 1 'nvidia-smi -i 4,5,6,7'

# 使用 gpustat（更美观）
pip install gpustat
watch -n 1 gpustat -i 4,5,6,7
```

### 方法 3: WandB 可视化

```bash
# 登录 WandB
conda activate searchr1
wandb login

# 训练时会自动上传到 WandB
# 访问 https://wandb.ai 查看实时曲线
```

### 方法 4: TensorBoard（可选）

```bash
# 如果日志包含 TensorBoard 格式
tensorboard --logdir ./checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu \
    --bind_all \
    --port 6006

# 在浏览器访问: http://your-server-ip:6006
```

---

## 🔧 常见问题排查

### 问题 1: "CUDA_VISIBLE_DEVICES 不生效"

**解决方案**：
```bash
# 显式设置环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 验证
python -c "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"

# 在训练脚本中再次确认
echo $CUDA_VISIBLE_DEVICES
```

### 问题 2: "无法连接到 BM25 服务"

**排查步骤**：
```bash
# 检查服务是否运行
ps aux | grep retrieval_server

# 检查端口是否监听
netstat -tuln | grep 8000

# 测试连接
curl http://127.0.0.1:8000/retrieve

# 查看服务日志
tail -f bm25.log  # 如果使用 nohup
# 或
tmux attach -t bm25  # 如果使用 tmux
```

### 问题 3: "Java not found (pyserini 需要)"

**解决方案**：
```bash
# 方法 1: 使用 conda 安装
conda activate retriever
conda install openjdk=11 -y

# 方法 2: 系统安装
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-11-jdk -y

# CentOS/RHEL
sudo yum install java-11-openjdk -y

# 验证
java -version
```

### 问题 4: "多卡训练不均衡"

**检查负载**：
```bash
watch -n 1 'nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv'
```

**可能原因**：
- 数据并行配置不正确
- Batch size 设置过小
- vLLM 配置问题

**解决方案**：
```bash
# 确保 tensor_model_parallel_size=1（数据并行模式）
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# 增大 batch size
TRAIN_BATCH_SIZE=1024  # 确保能被 4 整除
```

### 问题 5: "OOM (Out of Memory)"

**解决方案**：
```bash
# 编辑训练脚本，降低这些参数
TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE=32
GPU_MEMORY_UTIL=0.5

# 或者使用梯度累积
actor_rollout_ref.actor.gradient_accumulation_steps=2
```

### 问题 6: "权限问题"

```bash
# 给脚本添加执行权限
chmod +x *.sh

# 检查数据目录权限
ls -ld ./data/triviaqa_search
chmod -R 755 ./data
```

---

## 📈 训练预期结果

### 训练输出示例

```
============================================
TriviaQA Training with Qwen2.5-3B-Instruct
4x A800 GPUs (GPU 4,5,6,7)
============================================
Data directory: /path/to/data/triviaqa_search
Base model: Qwen/Qwen2.5-3B-Instruct
Experiment name: triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu
GPUs: 4,5,6,7
Number of GPUs: 4
============================================

Epoch 1/10 | Step 10/500
- Average Reward: 0.32
- KL Divergence: 0.0015
- Actor Loss: 0.28
- Critic Loss: 0.22
- Learning Rate: 8.5e-7
- GPU Memory (4/5/6/7): 19GB/19GB/20GB/19GB

Epoch 1/10 | Step 50/500
- Average Reward: 0.45
- KL Divergence: 0.0012
- Actor Loss: 0.21
- Critic Loss: 0.18
...
```

### GPU 使用情况

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   4  NVIDIA A800 80GB... On  | 00000000:34:00.0 Off |                    0 |
| N/A   45C    P0   250W / 300W |  19500MiB / 81920MiB |     95%      Default |
|   5  NVIDIA A800 80GB... On  | 00000000:35:00.0 Off |                    0 |
| N/A   46C    P0   252W / 300W |  19600MiB / 81920MiB |     96%      Default |
|   6  NVIDIA A800 80GB... On  | 00000000:36:00.0 Off |                    0 |
| N/A   44C    P0   248W / 300W |  19400MiB / 81920MiB |     94%      Default |
|   7  NVIDIA A800 80GB... On  | 00000000:37:00.0 Off |                    0 |
| N/A   45C    P0   251W / 300W |  19550MiB / 81920MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

### 训练时间估算

- **单个 epoch**: 约 20-30 分钟（4卡 A800）
- **总训练时间**: 约 3-5 小时（10 epochs）
- **比单卡快**: 约 3.5-4 倍

### 最终性能指标

在 TriviaQA 测试集上的预期结果：
- **EM Score**: 45-55%
- **平均搜索轮次**: 2-3
- **搜索成功率**: 75-85%

---

## 🔄 从检查点恢复训练

如果训练中断，可以从检查点恢复：

```bash
# 编辑训练脚本，添加恢复参数
nano train_triviaqa_qwen25_3b_4gpu.sh

# 在 python 命令中添加:
    +trainer.load_checkpoint=./checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu/actor/step_300 \

# 重新启动训练
bash train_triviaqa_qwen25_3b_4gpu.sh
```

---

## 📦 训练完成后

### 1. 检查点位置

```bash
ls -lh ./checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu/

# 应该看到:
# - actor/        (Actor 模型检查点)
# - critic/       (Critic 模型检查点)
# - step_100/
# - step_200/
# - ...
```

### 2. 评估模型

创建评估脚本 `eval_checkpoint.sh`:

```bash
#!/bin/bash
CHECKPOINT_DIR=./checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu/actor
DATA_DIR=./data/triviaqa_search

export CUDA_VISIBLE_DEVICES=4  # 评估只需要 1 张卡

python -m verl.trainer.main_ppo \
    data.val_files=$DATA_DIR/test.parquet \
    data.val_batch_size=128 \
    actor_rollout_ref.model.path=$CHECKPOINT_DIR \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3
```

运行评估：
```bash
chmod +x eval_checkpoint.sh
bash eval_checkpoint.sh
```

### 3. 导出模型

```bash
# 导出为 HuggingFace 格式
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = './checkpoints/triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em-4gpu/actor'
output_path = './models/triviaqa-qwen2.5-3b-final'

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f'Model exported to {output_path}')
"
```

---

## 🎯 高级配置

### 使用更大的模型（Qwen2.5-7B）

```bash
# 编辑训练脚本
nano train_triviaqa_qwen25_3b_4gpu.sh

# 修改模型路径
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

# 可能需要调整显存配置
GPU_MEMORY_UTIL=0.6
PPO_MICRO_BATCH_SIZE=32
```

### 使用 GRPO 而非 PPO

```bash
# 使用 GRPO 训练脚本
cp train_triviaqa_qwen25_3b_4gpu.sh train_triviaqa_qwen25_3b_4gpu_grpo.sh

# 编辑脚本，修改算法配置
algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_kl_loss=true \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.rollout.n_agent=5 \
```

### 启用混合精度训练

```bash
# 添加到训练脚本
actor_rollout_ref.actor.use_fp16=true \
critic.use_fp16=true \
```

---

## 📝 完整命令速查表

| 操作 | 命令 |
|------|------|
| 激活训练环境 | `conda activate searchr1` |
| 激活检索环境 | `conda activate retriever` |
| 下载 BM25 索引 | `huggingface-cli download PeterJinGo/wiki-18-bm25-index` |
| 处理数据 | `bash process_triviaqa_data.sh` |
| 启动 BM25 服务 | `tmux new -s bm25; bash retrieval_launch_bm25.sh` |
| 启动训练 | `tmux new -s training; bash train_triviaqa_qwen25_3b_4gpu.sh` |
| 监控 GPU | `watch -n 1 nvidia-smi -i 4,5,6,7` |
| 查看日志 | `tail -f training.log` |
| 评估模型 | `bash eval_checkpoint.sh` |

---

## 🆘 获取帮助

- 📖 完整指南: `TriviaQA_BM25_Qwen_Guide.md`
- 🐛 问题反馈: https://github.com/PeterGriffinJin/Search-R1/issues
- 💬 社区讨论: https://github.com/PeterGriffinJin/Search-R1/discussions

---

**祝训练顺利！** 🚀

如有问题，请及时检查日志并参考故障排查部分。