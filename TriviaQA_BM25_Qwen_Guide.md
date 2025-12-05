# TriviaQA + BM25 + Qwen2.5-3B-Instruct 完整实施方案

本文档提供了使用 **BM25 检索器** 和 **Qwen2.5-3B-Instruct** 模型在 **TriviaQA** 数据集上进行训练和评估的完整实施方案。

---

## 目录

1. [环境准备](#一环境准备)
2. [下载和准备 BM25 索引](#二下载和准备-bm25-索引)
3. [准备 TriviaQA 数据集](#三准备-triviaqa-数据集)
4. [启动 BM25 检索服务](#四启动-bm25-检索服务)
5. [配置和运行评估/推理](#五配置和运行评估推理)
6. [训练模型](#六训练模型)
7. [推理单个问题](#七推理单个问题快速测试)
8. [关键配置参数说明](#八关键配置参数说明)
9. [完整运行流程总结](#九完整运行流程总结)
10. [预期结果](#十预期结果)
11. [常见问题排查](#十一常见问题排查)

---

## 一、环境准备

### 1.1 创建两个独立的 Conda 环境

```bash
# 环境 1: 主训练环境
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0
pip install vllm==0.6.3
cd D:\search-r1\Search-R1
pip install -e .

# 环境 2: 检索器环境（独立）
conda create -n retriever python=3.10
conda activate retriever
pip install transformers datasets pyserini
pip install uvicorn fastapi
```

**注意**: BM25 检索器不需要 GPU，所以不需要安装 `faiss-gpu`。

---

## 二、下载和准备 BM25 索引

### 2.1 下载 Wiki-18 BM25 索引

```bash
# 设置保存路径
set save_path=D:\search-r1\wiki_data

# 下载 BM25 索引和语料库
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir %save_path%
```

下载完成后，你应该会有：
- `D:\search-r1\wiki_data\bm25\` - BM25 索引文件夹
- `D:\search-r1\wiki_data\wiki-18.jsonl` - 语料库文件

---

## 三、准备 TriviaQA 数据集

### 3.1 创建 TriviaQA 数据处理脚本

在 `D:\search-r1\Search-R1\scripts\data_process\` 目录下创建 `triviaqa_search.py`:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the TriviaQA dataset to parquet format
"""

import re
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/triviaqa_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'triviaqa'

    # Load TriviaQA from FlashRAG
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'triviaqa')

    # TriviaQA 通常只有 test 集，如果需要训练集可以从 train 中取样
    if 'test' in dataset:
        test_dataset = dataset['test']
    elif 'dev' in dataset:
        test_dataset = dataset['dev']
    else:
        test_dataset = dataset['train']

    # 如果有 train 集也处理
    train_dataset = None
    if 'train' in dataset:
        train_dataset = dataset['train']

    # 处理函数
    def make_map_fn(split):
        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 创建目录
    os.makedirs(local_dir, exist_ok=True)

    # 保存测试集
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 如果有训练集也保存
    if train_dataset is not None:
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"TriviaQA dataset processed and saved to {local_dir}")
```

### 3.2 运行数据处理脚本

```bash
conda activate searchr1
cd D:\search-r1\Search-R1

python scripts\data_process\triviaqa_search.py --local_dir .\data\triviaqa_search
```

处理完成后，你将得到：
- `D:\search-r1\Search-R1\data\triviaqa_search\test.parquet`
- `D:\search-r1\Search-R1\data\triviaqa_search\train.parquet` (如果有)

---

## 四、启动 BM25 检索服务

### 4.1 创建 BM25 启动脚本

在 `D:\search-r1\Search-R1\` 目录下创建 `retrieval_launch_bm25.bat`:

```batch
@echo off
set file_path=D:\search-r1\wiki_data
set index_file=%file_path%\bm25
set corpus_file=%file_path%\wiki-18.jsonl
set retriever_name=bm25

python search_r1\search\retrieval_server.py ^
    --index_path %index_file% ^
    --corpus_path %corpus_file% ^
    --topk 3 ^
    --retriever_name %retriever_name%
```

### 4.2 启动检索服务

```bash
# 打开新的命令行窗口
conda activate retriever
cd D:\search-r1\Search-R1
retrieval_launch_bm25.bat
```

服务启动后，会监听在 `http://127.0.0.1:8000/retrieve`

**验证**: 保持这个窗口运行，检索服务将持续提供服务。

---

## 五、配置和运行评估/推理

### 5.1 创建 TriviaQA 评估脚本

在 `D:\search-r1\Search-R1\` 目录下创建 `eval_triviaqa_qwen25_3b.bat`:

```batch
@echo off
REM TriviaQA 评估脚本 - 使用 Qwen2.5-3B-Instruct + BM25

set DATA_DIR=D:\search-r1\Search-R1\data\triviaqa_search
set BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
set EXPERIMENT_NAME=triviaqa-eval-qwen2.5-3b-instruct-bm25
set VLLM_ATTENTION_BACKEND=XFORMERS

REM 设置为评估模式
set PYTHONUNBUFFERED=1

python -m verl.trainer.main_ppo ^
    data.val_files=%DATA_DIR%\test.parquet ^
    data.val_batch_size=128 ^
    data.max_prompt_length=4096 ^
    data.max_response_length=500 ^
    data.max_start_length=2048 ^
    data.max_obs_length=500 ^
    actor_rollout_ref.model.path=%BASE_MODEL% ^
    actor_rollout_ref.rollout.name=vllm ^
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 ^
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 ^
    actor_rollout_ref.rollout.temperature=0.7 ^
    actor_rollout_ref.rollout.top_p=1.0 ^
    max_turns=4 ^
    retriever.url="http://127.0.0.1:8000/retrieve" ^
    retriever.topk=3 ^
    +trainer.val_only=true ^
    +trainer.val_before_train=true ^
    trainer.default_hdfs_dir=null ^
    trainer.default_local_dir=./checkpoints/%EXPERIMENT_NAME% ^
    trainer.project_name=search-r1-triviaqa ^
    trainer.experiment_name=%EXPERIMENT_NAME% ^
    trainer.logger=['console','tracking']
```

### 5.2 运行评估

```bash
# 确保 BM25 检索服务在另一个窗口运行中
conda activate searchr1
cd D:\search-r1\Search-R1
eval_triviaqa_qwen25_3b.bat
```

---

## 六、训练模型

### 6.1 创建训练脚本

在 `D:\search-r1\Search-R1\` 目录下创建 `train_triviaqa_qwen25_3b_ppo.bat`:

```batch
@echo off
REM TriviaQA PPO 训练脚本 - 使用 Qwen2.5-3B-Instruct + BM25

set DATA_DIR=D:\search-r1\Search-R1\data\triviaqa_search
set BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
set EXPERIMENT_NAME=triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em
set VLLM_ATTENTION_BACKEND=XFORMERS

set PYTHONUNBUFFERED=1

python -m verl.trainer.main_ppo ^
    data.train_files=%DATA_DIR%\train.parquet ^
    data.val_files=%DATA_DIR%\test.parquet ^
    data.train_batch_size=256 ^
    data.val_batch_size=128 ^
    data.max_prompt_length=4096 ^
    data.max_response_length=500 ^
    data.max_start_length=2048 ^
    data.max_obs_length=500 ^
    actor_rollout_ref.model.path=%BASE_MODEL% ^
    actor_rollout_ref.actor.optim.lr=1e-6 ^
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 ^
    actor_rollout_ref.actor.ppo_mini_batch_size=128 ^
    actor_rollout_ref.actor.ppo_micro_batch_size=32 ^
    actor_rollout_ref.rollout.name=vllm ^
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 ^
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 ^
    actor_rollout_ref.rollout.temperature=1.0 ^
    actor_rollout_ref.rollout.top_p=1.0 ^
    critic.optim.lr=1e-5 ^
    critic.ppo_micro_batch_size=8 ^
    algorithm.kl_ctrl.kl_coef=0.001 ^
    algorithm.adv_estimator=gae ^
    trainer.total_epochs=10 ^
    trainer.total_training_steps=500 ^
    trainer.save_freq=100 ^
    max_turns=4 ^
    retriever.url="http://127.0.0.1:8000/retrieve" ^
    retriever.topk=3 ^
    trainer.default_hdfs_dir=null ^
    trainer.default_local_dir=./checkpoints/%EXPERIMENT_NAME% ^
    trainer.project_name=search-r1-triviaqa ^
    trainer.experiment_name=%EXPERIMENT_NAME% ^
    trainer.logger=['console','tracking','wandb'] ^
    trainer.n_gpus_per_node=1 ^
    trainer.nnodes=1
```

### 6.2 运行训练

```bash
# 确保 BM25 检索服务在另一个窗口运行中
conda activate searchr1
cd D:\search-r1\Search-R1
train_triviaqa_qwen25_3b_ppo.bat
```

---

## 七、推理单个问题（快速测试）

### 7.1 创建推理脚本

创建 `infer_triviaqa.py`:

```python
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 配置
model_id = "Qwen/Qwen2.5-3B-Instruct"
search_api = "http://127.0.0.1:8000/retrieve"
max_turns = 4
topk = 3

# 加载模型和 tokenizer
print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# TriviaQA 问题示例
question = "What is the capital of Australia?"

# 构建 prompt
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
"""

# 生成
messages = [{"role": "user", "content": prompt}]
conversation = ""

for turn in range(max_turns):
    print(f"\n--- Turn {turn + 1} ---")

    # 生成响应
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"Model: {response}")

    conversation += response

    # 检查是否有搜索请求
    search_match = re.search(r'<search>(.*?)</search>', response)
    if search_match:
        query = search_match.group(1).strip()
        print(f"Searching for: {query}")

        # 调用检索 API
        response_api = requests.post(
            search_api,
            json={"query": query, "topk": topk}
        )

        if response_api.status_code == 200:
            results = response_api.json()
            info_text = "\n\n".join([f"[{i+1}] {doc['contents'][:500]}..."
                                     for i, doc in enumerate(results.get('documents', []))])

            info_message = f"\n<information>\n{info_text}\n</information>\n"
            print(f"Retrieved information")

            # 添加到对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": info_message})
            conversation += info_message
        else:
            print(f"Search failed: {response_api.status_code}")
            break
    else:
        # 没有搜索，检查是否有答案
        answer_match = re.search(r'<answer>(.*?)</answer>', response)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            print(f"\n=== Final Answer ===\n{final_answer}")
            break

print(f"\n=== Full Conversation ===\n{conversation}")
```

### 7.2 运行推理

```bash
conda activate searchr1
cd D:\search-r1\Search-R1
python infer_triviaqa.py
```

---

## 八、关键配置参数说明

### 8.1 数据相关参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| `data.max_prompt_length` | 4096 | 最大 prompt 长度 |
| `data.max_response_length` | 500 | 最大响应长度 |
| `data.max_start_length` | 2048 | 最大起始长度 |
| `data.max_obs_length` | 500 | 最大观察长度（搜索结果） |

### 8.2 检索器参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| `retriever.url` | http://127.0.0.1:8000/retrieve | BM25 服务 URL |
| `retriever.topk` | 3 | 每次检索返回的文档数 |
| `max_turns` | 4 | 最大交互轮次 |

### 8.3 模型生成参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| `temperature` | 0.7-1.0 | 生成温度（评估用 0.7，训练用 1.0） |
| `top_p` | 1.0 | Top-p 采样 |
| `gpu_memory_utilization` | 0.6 | vLLM GPU 内存使用率 |

### 8.4 训练参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| `actor.optim.lr` | 1e-6 | Actor 学习率 |
| `critic.optim.lr` | 1e-5 | Critic 学习率 |
| `kl_coef` | 0.001 | KL 散度系数 |
| `total_epochs` | 10-15 | 总训练轮数 |

---

## 九、完整运行流程总结

```bash
# === 第 1 步：环境准备 ===
conda create -n searchr1 python=3.9
conda activate searchr1
# ... 安装依赖

conda create -n retriever python=3.10
conda activate retriever
# ... 安装依赖

# === 第 2 步：下载 BM25 索引 ===
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir D:\search-r1\wiki_data

# === 第 3 步：处理 TriviaQA 数据 ===
conda activate searchr1
cd D:\search-r1\Search-R1
python scripts\data_process\triviaqa_search.py --local_dir .\data\triviaqa_search

# === 第 4 步：启动 BM25 检索服务（新窗口）===
conda activate retriever
cd D:\search-r1\Search-R1
retrieval_launch_bm25.bat
# 保持运行

# === 第 5 步：运行评估（新窗口）===
conda activate searchr1
cd D:\search-r1\Search-R1
eval_triviaqa_qwen25_3b.bat

# === （可选）第 6 步：训练模型 ===
train_triviaqa_qwen25_3b_ppo.bat
```

---

## 十、预期结果

### 评估输出示例

```
Validation results:
- Total samples: 10000
- Average reward: 0.65
- EM score: 65.0%
- Average turns: 2.3
```

### 检查点位置

- 评估结果: `D:\search-r1\Search-R1\checkpoints\triviaqa-eval-qwen2.5-3b-instruct-bm25\`
- 训练检查点: `D:\search-r1\Search-R1\checkpoints\triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em\`

---

## 十一、常见问题排查

### 问题 1: BM25 服务启动失败
- 检查 `pyserini` 是否正确安装
- 检查索引路径是否正确
- 检查 Java 是否安装（pyserini 依赖 Java）

### 问题 2: 内存不足
- 降低 `gpu_memory_utilization` (从 0.6 到 0.4)
- 使用更小的模型 (Qwen2.5-1.8B)
- 减小 `batch_size`

### 问题 3: TriviaQA 数据集加载失败
- 检查网络连接
- 尝试手动下载并指定本地路径

### 问题 4: VLLM 版本不兼容
- 确保使用 vllm==0.6.3
- 设置 `VLLM_ATTENTION_BACKEND=XFORMERS`

---

## 附录：硬件要求

### 最低配置
- GPU: 单张 16GB 显存（如 RTX 4060 Ti 16GB、V100 16GB）
- CPU: 4核以上
- 内存: 32GB
- 硬盘: 100GB 可用空间

### 推荐配置
- GPU: 单张 24GB 显存（如 RTX 4090、A5000、A100）
- CPU: 8核以上
- 内存: 64GB
- 硬盘: 200GB 可用空间（SSD）

---

**文档版本**: v1.0
**最后更新**: 2025-12-06
**适用模型**: Qwen2.5-3B-Instruct
**检索器**: BM25 (CPU only)
**数据集**: TriviaQA
