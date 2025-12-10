#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3B → 1.5B 模型蒸馏工具

基于你的 Search-R1 PPO 训练配置：
- 教师模型: verl_checkpoints/.../actor/step_200
- 学生模型: /usr/yuque/guo/models/qwen2.5-1.5b-instruct
- 训练数据: /usr/yuque/guo/searchr1/data/nq_hotpotqa_train/train.parquet
- 检索服务: http://127.0.0.1:8000/retrieve

使用方法：
    # 步骤1: 启动检索服务（在另一个终端）
    bash retrieval_launch_bm25.sh

    # 步骤2: 生成蒸馏数据（需要多轮交互）
    python distill_3b_to_1.5b.py generate \
        --teacher_model /usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200 \
        --data /usr/yuque/guo/searchr1/data/nq_hotpotqa_train/train.parquet \
        --output /usr/yuque/guo/searchr1/distill_data.jsonl \
        --retriever_url http://127.0.0.1:8000/retrieve \
        --num_samples 10000

    # 步骤3: 训练1.5B
    python distill_3b_to_1.5b.py train \
        --student_model /usr/yuque/guo/models/qwen2.5-1.5b-instruct \
        --data /usr/yuque/guo/searchr1/distill_data.jsonl \
        --output /usr/yuque/guo/searchr1/checkpoints/qwen2.5-1.5b-distilled
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import re
from tqdm import tqdm
import requests


@dataclass
class DistillConfig:
    """蒸馏配置 - 与你的PPO训练配置对齐"""
    teacher_model_path: str
    student_model_path: str = "/usr/yuque/guo/models/qwen2.5-1.5b-instruct"

    # 数据生成配置（与train_nq_hotpotqa_qwen25_3b_4gpu.sh对齐）
    num_samples: int = 10000
    max_new_tokens: int = 500       # data.max_response_length=500
    max_prompt_length: int = 4096   # data.max_prompt_length=4096
    temperature: float = 1.0        # actor_rollout_ref.rollout.temperature=1
    top_p: float = 1.0              # actor_rollout_ref.rollout.top_p=1.0
    num_return_sequences: int = 4   # 每个问题生成多个响应，选最好的
    max_turns: int = 4              # max_turns=4

    # 检索配置
    retriever_url: str = "http://127.0.0.1:8000/retrieve"
    retriever_topk: int = 3         # retriever.topk=3

    # 筛选配置
    min_response_quality: float = 0.6
    require_correct_answer: bool = True
    require_valid_format: bool = True

    # 训练配置（SFT比PPO用稍大的学习率）
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096


class Retriever:
    """检索器 - 调用你的BM25服务"""

    def __init__(self, url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3):
        self.url = url
        self.topk = topk

    def search(self, query: str) -> List[Dict[str, str]]:
        """调用检索服务"""
        try:
            response = requests.post(
                self.url,
                json={"query": query, "topk": self.topk},
                timeout=30
            )
            if response.status_code == 200:
                results = response.json()
                # 返回格式: [{"title": ..., "text": ...}, ...]
                return results.get("documents", results.get("results", []))
            else:
                print(f"检索失败: {response.status_code}")
                return []
        except Exception as e:
            print(f"检索异常: {e}")
            return []

    def format_docs(self, docs: List[Dict]) -> str:
        """格式化检索结果"""
        if not docs:
            return ""

        formatted = []
        for i, doc in enumerate(docs, 1):
            title = doc.get('title', '')
            text = doc.get('text', doc.get('content', ''))
            formatted.append(f'Doc {i}(Title: "{title}") {text}')

        return "\n".join(formatted)


class DistillDataGenerator:
    """蒸馏数据生成器 - 支持多轮检索交互"""

    def __init__(
        self,
        teacher_model_path: str,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        retriever_topk: int = 3
    ):
        self.teacher_model_path = teacher_model_path
        self.retriever = Retriever(retriever_url, retriever_topk)
        self.model = None
        self.tokenizer = None

    def load_teacher(self):
        """加载教师模型"""
        print(f"加载教师模型: {self.teacher_model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        print("✓ 教师模型加载完成")

    def generate_distill_data(
        self,
        source_data: List[Dict],
        config: DistillConfig
    ) -> List[Dict]:
        """
        生成蒸馏数据

        核心逻辑：
        1. 给教师模型问题
        2. 模型生成 <think>...</think><search>query</search>
        3. 调用检索服务获取文档
        4. 将文档作为 <information>...</information> 返回给模型
        5. 重复直到模型输出 <answer>
        6. 筛选正确的响应
        """
        if self.model is None:
            self.load_teacher()

        distill_data = []
        stats = {"total": 0, "success": 0, "failed_format": 0, "failed_answer": 0}

        print(f"\n开始生成蒸馏数据，共 {len(source_data)} 个样本...")

        for item in tqdm(source_data):
            stats["total"] += 1

            question = item.get('question', item.get('prompt', ''))
            golden_answer = item.get('answer', item.get('reward_model', {}).get('ground_truth', ''))

            if not question or not golden_answer:
                continue

            # 生成多个响应，选最好的
            best_result = None
            best_quality = 0.0

            for attempt in range(config.num_return_sequences):
                result = self._generate_single_response(
                    question=question,
                    golden_answer=golden_answer,
                    max_turns=config.max_turns,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature
                )

                if result and result['quality'] > best_quality:
                    best_result = result
                    best_quality = result['quality']

            # 检查是否满足要求
            if best_result is None:
                stats["failed_format"] += 1
                continue

            if config.require_correct_answer and not best_result['is_correct']:
                stats["failed_answer"] += 1
                continue

            if best_result['quality'] < config.min_response_quality:
                continue

            # 保存成功的样本
            stats["success"] += 1
            distill_data.append({
                'question': question,
                'golden_answer': golden_answer,
                'full_trajectory': best_result['full_trajectory'],
                'teacher_response': best_result['final_response'],
                'extracted_answer': best_result['extracted_answer'],
                'is_correct': best_result['is_correct'],
                'quality': best_result['quality'],
                'num_searches': best_result['num_searches'],
                # 用于SFT的对话格式
                'conversations': [
                    {'role': 'user', 'content': self._build_user_prompt(question)},
                    {'role': 'assistant', 'content': best_result['full_trajectory']}
                ]
            })

            # 定期打印进度
            if stats["total"] % 100 == 0:
                print(f"\n  进度: {stats['total']}/{len(source_data)}, "
                      f"成功: {stats['success']}, "
                      f"格式错误: {stats['failed_format']}, "
                      f"答案错误: {stats['failed_answer']}")

        print(f"\n✓ 生成完成!")
        print(f"  总样本: {stats['total']}")
        print(f"  成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"  格式错误: {stats['failed_format']}")
        print(f"  答案错误: {stats['failed_answer']}")

        return distill_data

    def _generate_single_response(
        self,
        question: str,
        golden_answer: str,
        max_turns: int = 4,
        max_tokens: int = 500,
        temperature: float = 1.0
    ) -> Optional[Dict]:
        """
        生成单个响应（多轮交互）

        模拟真实的 search-r1 环境：
        - 模型输出 <search>query</search> 时，调用检索
        - 将检索结果作为 <information>...</information> 返回
        - 重复直到输出 <answer> 或达到max_turns
        """
        import torch

        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        user_prompt = self._build_user_prompt(question)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        full_trajectory = ""  # 完整的生成轨迹
        num_searches = 0

        for turn in range(max_turns):
            # 构建prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 生成
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            full_trajectory += response

            # 检查是否有 <answer>
            if '<answer>' in response and '</answer>' in response:
                # 完成了
                break

            # 检查是否有 <search>
            search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
            if search_match:
                query = search_match.group(1).strip()
                num_searches += 1

                # 调用检索
                docs = self.retriever.search(query)
                formatted_docs = self.retriever.format_docs(docs)

                # 构造 information
                info_response = f"\n\n<information>{formatted_docs}</information>\n\n"
                full_trajectory += info_response

                # 更新messages，继续生成
                messages.append({"role": "assistant", "content": response + info_response})
            else:
                # 既没有answer也没有search，可能格式错误
                break

        # 评估结果
        return self._evaluate_trajectory(full_trajectory, golden_answer, num_searches)

    def _evaluate_trajectory(
        self,
        trajectory: str,
        golden_answer: str,
        num_searches: int
    ) -> Optional[Dict]:
        """评估生成的轨迹"""

        # 检查格式
        has_think = '<think>' in trajectory and '</think>' in trajectory
        has_answer = '<answer>' in trajectory and '</answer>' in trajectory

        if not has_answer:
            return None

        # 提取答案
        answer_match = re.search(r'<answer>(.*?)</answer>', trajectory, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else ""

        # 检查正确性
        is_correct = self._check_answer_correct(extracted_answer, golden_answer)

        # 计算质量分数
        quality = 0.0
        if has_think:
            quality += 0.2
        if has_answer:
            quality += 0.2
        if is_correct:
            quality += 0.5
        if 0 < num_searches <= 3:
            quality += 0.1  # 适度使用search

        # 检查推理质量
        think_matches = re.findall(r'<think>(.*?)</think>', trajectory, re.DOTALL)
        if think_matches:
            total_reasoning = " ".join(think_matches)
            if len(total_reasoning) > 50:
                quality += 0.05

        return {
            'full_trajectory': trajectory,
            'final_response': trajectory,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'quality': min(1.0, quality),
            'num_searches': num_searches
        }

    def _check_answer_correct(self, extracted: str, golden: str) -> bool:
        """检查答案是否正确"""
        extracted_lower = extracted.lower().strip()

        # 处理多个可能的答案
        if isinstance(golden, list):
            golden_list = [str(g).strip().lower() for g in golden]
        else:
            golden_list = [str(golden).strip().lower()]

        for g in golden_list:
            # 完全包含
            if g in extracted_lower or extracted_lower in g:
                return True
            # 标准化后比较
            if self._normalize(g) == self._normalize(extracted_lower):
                return True

        return False

    def _normalize(self, text: str) -> str:
        """标准化文本"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def _build_user_prompt(self, question: str) -> str:
        """构建用户prompt"""
        return f"""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}"""

    def save_distill_data(self, data: List[Dict], output_path: str):
        """保存蒸馏数据"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✓ 蒸馏数据已保存到: {output_path}")

        # 打印统计
        correct_count = sum(1 for d in data if d.get('is_correct', False))
        avg_searches = sum(d.get('num_searches', 0) for d in data) / len(data) if data else 0
        print(f"  正确率: {correct_count/len(data)*100:.1f}%")
        print(f"  平均检索次数: {avg_searches:.2f}")


class StudentTrainer:
    """学生模型训练器"""

    def __init__(self, student_model_path: str):
        self.student_model_path = student_model_path

    def train(
        self,
        distill_data_path: str,
        output_dir: str,
        config: DistillConfig
    ):
        """训练学生模型"""
        print(f"\n开始训练学生模型...")
        print(f"  学生模型: {self.student_model_path}")
        print(f"  蒸馏数据: {distill_data_path}")
        print(f"  输出目录: {output_dir}")

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq
        )
        from datasets import load_dataset
        import torch

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.student_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载数据
        dataset = load_dataset('json', data_files=distill_data_path, split='train')
        print(f"  数据集大小: {len(dataset)}")

        # 数据预处理
        def preprocess(examples):
            texts = []
            for convs in examples['conversations']:
                text = tokenizer.apply_chat_template(
                    convs,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)

            model_inputs = tokenizer(
                texts,
                max_length=config.max_seq_length,
                truncation=True,
                padding=False
            )
            model_inputs['labels'] = model_inputs['input_ids'].copy()
            return model_inputs

        tokenized_dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            optim="adamw_torch",
            report_to="wandb",
            run_name="qwen2.5-1.5b-distill",
            gradient_checkpointing=True,
        )

        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        # 训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        # 保存
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✓ 训练完成，模型已保存到: {output_dir}")


def load_parquet_data(path: str) -> List[Dict]:
    """加载parquet格式的数据"""
    import pandas as pd

    print(f"加载数据: {path}")
    df = pd.read_parquet(path)
    print(f"  列名: {list(df.columns)}")
    print(f"  样本数: {len(df)}")

    # 转换为list of dict
    data = df.to_dict('records')

    # 查看第一个样本的结构
    if data:
        print(f"  第一个样本的keys: {list(data[0].keys())}")

    return data


def main():
    parser = argparse.ArgumentParser(description="3B → 1.5B 模型蒸馏工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 生成蒸馏数据
    gen_parser = subparsers.add_parser('generate', help='生成蒸馏数据')
    gen_parser.add_argument('--teacher_model', type=str, required=True,
                           help='教师模型路径')
    gen_parser.add_argument('--data', type=str, required=True,
                           help='训练数据路径 (支持 .parquet 和 .jsonl)')
    gen_parser.add_argument('--output', type=str, required=True,
                           help='输出蒸馏数据路径')
    gen_parser.add_argument('--retriever_url', type=str,
                           default='http://127.0.0.1:8000/retrieve',
                           help='检索服务URL')
    gen_parser.add_argument('--num_samples', type=int, default=10000,
                           help='生成样本数')
    gen_parser.add_argument('--num_return', type=int, default=4,
                           help='每个问题生成的响应数')
    gen_parser.add_argument('--max_turns', type=int, default=4,
                           help='最大检索轮数')

    # 训练学生模型
    train_parser = subparsers.add_parser('train', help='训练学生模型')
    train_parser.add_argument('--student_model', type=str,
                             default='/usr/yuque/guo/models/qwen2.5-1.5b-instruct',
                             help='学生模型路径')
    train_parser.add_argument('--data', type=str, required=True,
                             help='蒸馏数据路径')
    train_parser.add_argument('--output', type=str, required=True,
                             help='输出模型路径')
    train_parser.add_argument('--epochs', type=int, default=3,
                             help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=1e-5,
                             help='学习率')
    train_parser.add_argument('--batch_size', type=int, default=4,
                             help='批大小')

    args = parser.parse_args()

    if args.command == 'generate':
        # 加载数据
        if args.data.endswith('.parquet'):
            source_data = load_parquet_data(args.data)
        else:
            source_data = []
            with open(args.data, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        source_data.append(json.loads(line))

        # 限制样本数
        if len(source_data) > args.num_samples:
            source_data = random.sample(source_data, args.num_samples)
            print(f"随机采样 {args.num_samples} 个样本")

        # 配置
        config = DistillConfig(
            teacher_model_path=args.teacher_model,
            retriever_url=args.retriever_url,
            num_return_sequences=args.num_return,
            max_turns=args.max_turns
        )

        # 生成
        generator = DistillDataGenerator(
            args.teacher_model,
            args.retriever_url
        )
        distill_data = generator.generate_distill_data(source_data, config)
        generator.save_distill_data(distill_data, args.output)

    elif args.command == 'train':
        config = DistillConfig(
            teacher_model_path="",
            student_model_path=args.student_model,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size
        )

        trainer = StudentTrainer(args.student_model)
        trainer.train(args.data, args.output, config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
