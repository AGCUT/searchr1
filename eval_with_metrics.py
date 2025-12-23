#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本 - 支持记录 search 次数和回答时间

功能：
1. 加载测试数据集
2. 对每个问题进行推理
3. 记录每个问题的：
   - search 次数
   - 回答时间（秒）
   - 是否正确
   - 模型回答
4. 输出详细统计报告

使用方法：
    python eval_with_metrics.py \
        --model /usr/yuque/guo/searchr1/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200 \
        --data /usr/yuque/guo/searchr1/data/nq_hotpotqa_train/test.parquet \
        --output eval_results.json \
        --retriever_url http://127.0.0.1:8000/retrieve \
        --max_samples 500
"""

import argparse
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import requests
from tqdm import tqdm


@dataclass
class EvalResult:
    """单个样本的评估结果"""
    question: str
    golden_answer: str
    model_answer: str
    extracted_answer: str
    is_correct: bool
    num_searches: int
    response_time: float  # 秒
    full_trajectory: str
    data_source: str = ""  # nq / hotpotqa / triviaqa

    # 多答案相关字段
    all_answers: List[str] = field(default_factory=list)  # 所有答案
    num_answers: int = 0  # 答案数量
    answer_changed: bool = False  # 是否改变了答案


@dataclass
class EvalStats:
    """评估统计"""
    total_samples: int = 0
    correct_samples: int = 0
    accuracy: float = 0.0

    # Search 统计
    total_searches: int = 0
    avg_searches: float = 0.0
    max_searches: int = 0
    min_searches: int = 0
    search_distribution: Dict[int, int] = field(default_factory=dict)  # {0: 10, 1: 50, 2: 30, ...}

    # 时间统计
    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = 0.0

    # 答案一致性统计（新增）
    samples_with_multiple_answers: int = 0  # 有多个答案的样本数
    samples_with_changed_answers: int = 0   # 改变答案的样本数
    answer_change_rate: float = 0.0         # 改变答案的比例
    avg_answers_per_question: float = 0.0   # 平均每题多少个答案
    accuracy_changed_answer: float = 0.0    # 改变答案的题目准确率
    accuracy_single_answer: float = 0.0     # 单答案题目准确率

    # 按数据源统计
    by_source: Dict[str, dict] = field(default_factory=dict)


class Evaluator:
    """评估器"""

    def __init__(
        self,
        model_path: str,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        retriever_topk: int = 3,
        max_turns: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 1024
    ):
        self.model_path = model_path
        self.retriever_url = retriever_url
        self.retriever_topk = retriever_topk
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self.device = None

        # Qwen2.5 的 EOS token IDs
        self.eos_token_ids = [151645, 151643]

    def load_model(self):
        """加载模型"""
        import torch
        import transformers

        print(f"加载模型: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        print(f"✓ 模型加载完成，设备: {self.device}")

    def search(self, query: str) -> str:
        """调用检索服务"""
        try:
            payload = {
                "queries": [query],
                "topk": self.retriever_topk,
                "return_scores": True
            }
            response = requests.post(self.retriever_url, json=payload, timeout=30)

            if response.status_code == 200:
                results = response.json().get('result', [[]])
                return self._format_search_results(results[0] if results else [])
            else:
                return ""
        except Exception as e:
            print(f"检索异常: {e}")
            return ""

    def _format_search_results(self, results: List[Dict]) -> str:
        """格式化检索结果"""
        formatted = []
        for idx, doc_item in enumerate(results):
            # 兼容多种返回格式
            if isinstance(doc_item, str):
                # 直接是字符串
                content = doc_item
            elif isinstance(doc_item, dict):
                # 字典格式
                doc = doc_item.get('document', '')
                if isinstance(doc, dict):
                    # {"document": {"contents": "..."}}
                    content = doc.get('contents', '')
                else:
                    # {"document": "..."} 字符串格式
                    content = doc
            else:
                content = str(doc_item)

            if content:
                parts = content.split("\n")
                title = parts[0].strip('"') if parts else ""
                text = "\n".join(parts[1:]) if len(parts) > 1 else ""
                formatted.append(f"Doc {idx+1}(Title: {title}) {text}")
        return "\n".join(formatted)

    def _build_prompt(self, question: str) -> str:
        """构建 prompt"""
        question = question.strip()
        if question and question[-1] != '?':
            question += '?'

        prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        return prompt

    def _get_query(self, text: str) -> Optional[str]:
        """从文本中提取 search query"""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def _extract_answer(self, text: str) -> Tuple[str, List[str], int, bool]:
        """
        从文本中提取 answer

        返回:
            - 最终答案（最后一个）
            - 所有答案列表
            - 答案数量
            - 是否改变了答案
        """
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)

        if not matches:
            return "", [], 0, False

        # 清理所有答案
        all_answers = [m.strip() for m in matches]

        # 检查是否改变了答案（标准化后比较）
        normalized_answers = [self._normalize_answer(a) for a in all_answers]
        answer_changed = len(set(normalized_answers)) > 1

        # 返回最后一个答案
        return all_answers[-1], all_answers, len(all_answers), answer_changed

    def _normalize_answer(self, text: str) -> str:
        """标准化答案用于比较"""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _check_correct(self, extracted: str, golden: str) -> bool:
        """检查答案是否正确"""
        extracted_lower = extracted.lower().strip()

        if isinstance(golden, list):
            golden_list = [str(g).strip().lower() for g in golden]
        else:
            golden_list = [str(golden).strip().lower()]

        for g in golden_list:
            if g in extracted_lower or extracted_lower in g:
                return True
            # 标准化比较
            g_norm = re.sub(r'[^\w\s]', '', g)
            e_norm = re.sub(r'[^\w\s]', '', extracted_lower)
            if g_norm == e_norm:
                return True

        return False

    def evaluate_single(self, question: str, golden_answer: str) -> EvalResult:
        """评估单个问题"""
        import torch
        import transformers

        start_time = time.time()

        prompt = self._build_prompt(question)
        full_trajectory = ""
        num_searches = 0

        # 定义停止条件
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n"]

        class StopOnSequence(transformers.StoppingCriteria):
            def __init__(self, target_sequences, tokenizer):
                self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
                self.target_lengths = [len(ids) for ids in self.target_ids]

            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[1] < min(self.target_lengths):
                    return False
                for i, target in enumerate(self.target_ids):
                    target_tensor = torch.as_tensor(target, device=input_ids.device)
                    if input_ids.shape[1] >= self.target_lengths[i]:
                        if torch.equal(input_ids[0, -self.target_lengths[i]:], target_tensor):
                            return True
                return False

        stopping_criteria = transformers.StoppingCriteriaList([
            StopOnSequence(target_sequences, self.tokenizer)
        ])

        # 多轮生成
        for turn in range(self.max_turns):
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                )

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_trajectory += output_text

            # 检查是否结束
            if outputs[0][-1].item() in self.eos_token_ids:
                break

            # 检查是否有 search
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            query = self._get_query(full_output)

            if query:
                num_searches += 1
                search_results = self.search(query)
                search_text = f"\n\n<information>{search_results}</information>\n\n"
                full_trajectory += search_text
                prompt += output_text + search_text
            else:
                # 没有 search 也没有结束，可能出问题了
                break

        end_time = time.time()
        response_time = end_time - start_time

        # 提取答案并检查正确性（使用最后一个答案）
        extracted_answer, all_answers, num_answers, answer_changed = self._extract_answer(full_trajectory)
        is_correct = self._check_correct(extracted_answer, golden_answer)

        return EvalResult(
            question=question,
            golden_answer=str(golden_answer),
            model_answer=full_trajectory,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            num_searches=num_searches,
            response_time=response_time,
            full_trajectory=full_trajectory,
            all_answers=all_answers,
            num_answers=num_answers,
            answer_changed=answer_changed
        )

    def evaluate_dataset(
        self,
        data: List[Dict],
        max_samples: Optional[int] = None
    ) -> Tuple[List[EvalResult], EvalStats]:
        """评估整个数据集"""
        if self.model is None:
            self.load_model()

        if max_samples and len(data) > max_samples:
            import random
            data = random.sample(data, max_samples)
            print(f"随机采样 {max_samples} 个样本进行评估")

        results = []
        stats = EvalStats()

        # 按数据源分组统计
        source_stats = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'searches': [], 'times': []
        })

        print(f"\n开始评估，共 {len(data)} 个样本...")

        for item in tqdm(data, desc="评估中"):
            question = item.get('question', item.get('prompt', ''))
            golden_answer = item.get('answer', item.get('reward_model', {}).get('ground_truth', ''))
            data_source = item.get('data_source', item.get('source', 'unknown'))

            if not question or not golden_answer:
                continue

            try:
                result = self.evaluate_single(question, golden_answer)
                result.data_source = data_source
                results.append(result)

                # 更新统计
                stats.total_samples += 1
                if result.is_correct:
                    stats.correct_samples += 1
                stats.total_searches += result.num_searches
                stats.total_time += result.response_time

                # 更新 search 分布
                stats.search_distribution[result.num_searches] = \
                    stats.search_distribution.get(result.num_searches, 0) + 1

                # 按数据源统计
                source_stats[data_source]['total'] += 1
                if result.is_correct:
                    source_stats[data_source]['correct'] += 1
                source_stats[data_source]['searches'].append(result.num_searches)
                source_stats[data_source]['times'].append(result.response_time)

            except Exception as e:
                print(f"\n评估失败: {question[:50]}... 错误: {e}")
                continue

        # 计算最终统计
        if stats.total_samples > 0:
            stats.accuracy = stats.correct_samples / stats.total_samples
            stats.avg_searches = stats.total_searches / stats.total_samples
            stats.avg_time = stats.total_time / stats.total_samples

            all_searches = [r.num_searches for r in results]
            all_times = [r.response_time for r in results]

            stats.max_searches = max(all_searches) if all_searches else 0
            stats.min_searches = min(all_searches) if all_searches else 0
            stats.max_time = max(all_times) if all_times else 0
            stats.min_time = min(all_times) if all_times else 0

            # 答案一致性统计
            total_answers = sum(r.num_answers for r in results)
            stats.avg_answers_per_question = total_answers / stats.total_samples
            stats.samples_with_multiple_answers = sum(1 for r in results if r.num_answers > 1)
            stats.samples_with_changed_answers = sum(1 for r in results if r.answer_changed)
            stats.answer_change_rate = stats.samples_with_changed_answers / stats.total_samples

            # 改变答案 vs 单答案的准确率对比
            changed_results = [r for r in results if r.answer_changed]
            single_results = [r for r in results if r.num_answers == 1]

            if changed_results:
                stats.accuracy_changed_answer = sum(1 for r in changed_results if r.is_correct) / len(changed_results)
            if single_results:
                stats.accuracy_single_answer = sum(1 for r in single_results if r.is_correct) / len(single_results)

            # 按数据源的最终统计
            for source, s in source_stats.items():
                if s['total'] > 0:
                    stats.by_source[source] = {
                        'total': s['total'],
                        'correct': s['correct'],
                        'accuracy': s['correct'] / s['total'],
                        'avg_searches': sum(s['searches']) / len(s['searches']),
                        'avg_time': sum(s['times']) / len(s['times'])
                    }

        return results, stats


def load_data(path: str) -> List[Dict]:
    """加载数据"""
    if path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict('records')
    else:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data


def print_stats(stats: EvalStats):
    """打印统计信息"""
    print("\n" + "=" * 70)
    print("评估结果统计")
    print("=" * 70)

    print(f"\n【整体统计】")
    print(f"  总样本数: {stats.total_samples}")
    print(f"  正确数: {stats.correct_samples}")
    print(f"  准确率: {stats.accuracy * 100:.2f}%")

    print(f"\n【Search 统计】")
    print(f"  总 search 次数: {stats.total_searches}")
    print(f"  平均 search 次数: {stats.avg_searches:.2f}")
    print(f"  最大 search 次数: {stats.max_searches}")
    print(f"  最小 search 次数: {stats.min_searches}")

    print(f"\n  Search 次数分布:")
    for num_search in sorted(stats.search_distribution.keys()):
        count = stats.search_distribution[num_search]
        pct = count / stats.total_samples * 100
        bar = "█" * int(pct / 2)
        print(f"    {num_search} 次: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n【时间统计】")
    print(f"  总时间: {stats.total_time:.2f} 秒")
    print(f"  平均时间: {stats.avg_time:.2f} 秒/样本")
    print(f"  最长时间: {stats.max_time:.2f} 秒")
    print(f"  最短时间: {stats.min_time:.2f} 秒")

    # 答案一致性统计
    print(f"\n【答案一致性统计】")
    print(f"  平均答案数/题: {stats.avg_answers_per_question:.2f}")
    print(f"  多答案样本数: {stats.samples_with_multiple_answers} ({stats.samples_with_multiple_answers/stats.total_samples*100:.1f}%)")
    print(f"  改变答案样本数: {stats.samples_with_changed_answers} ({stats.answer_change_rate*100:.1f}%)")

    if stats.samples_with_changed_answers > 0:
        print(f"\n  准确率对比:")
        print(f"    改变答案的题目: {stats.accuracy_changed_answer*100:.2f}%")
        print(f"    单答案的题目:   {stats.accuracy_single_answer*100:.2f}%")

        # 计算差异
        diff = (stats.accuracy_changed_answer - stats.accuracy_single_answer) * 100
        if diff > 0:
            print(f"    → 改变答案后准确率提高了 {diff:.2f}%")
        else:
            print(f"    → 改变答案后准确率降低了 {abs(diff):.2f}%")

    if stats.by_source:
        print(f"\n【按数据源统计】")
        print(f"  {'数据源':<20} {'样本数':>8} {'正确数':>8} {'准确率':>10} {'平均search':>12} {'平均时间':>12}")
        print("  " + "-" * 70)
        for source, s in sorted(stats.by_source.items()):
            print(f"  {source:<20} {s['total']:>8} {s['correct']:>8} {s['accuracy']*100:>9.2f}% {s['avg_searches']:>12.2f} {s['avg_time']:>11.2f}s")

    print("\n" + "=" * 70)


def save_results(results: List[EvalResult], stats: EvalStats, output_path: str):
    """保存结果"""
    output = {
        'stats': asdict(stats),
        'results': [asdict(r) for r in results]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 结果已保存到: {output_path}")

    # 同时保存一个简洁的 CSV
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("question,golden_answer,extracted_answer,is_correct,num_searches,response_time,data_source\n")
        for r in results:
            q = r.question.replace('"', '""')[:100]
            g = str(r.golden_answer).replace('"', '""')[:50]
            e = r.extracted_answer.replace('"', '""')[:50]
            f.write(f'"{q}","{g}","{e}",{r.is_correct},{r.num_searches},{r.response_time:.2f},{r.data_source}\n')

    print(f"✓ CSV 已保存到: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="评估脚本 - 支持记录 search 次数和回答时间")
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据路径')
    parser.add_argument('--output', type=str, default='eval_results.json', help='输出结果路径')
    parser.add_argument('--retriever_url', type=str, default='http://127.0.0.1:8000/retrieve', help='检索服务URL')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数')
    parser.add_argument('--max_turns', type=int, default=4, help='最大检索轮数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')

    args = parser.parse_args()

    print("=" * 70)
    print("评估配置")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"检索服务: {args.retriever_url}")
    print(f"最大样本数: {args.max_samples or '全部'}")
    print("=" * 70)

    # 加载数据
    data = load_data(args.data)
    print(f"加载了 {len(data)} 个样本")

    # 初始化评估器
    evaluator = Evaluator(
        model_path=args.model,
        retriever_url=args.retriever_url,
        max_turns=args.max_turns,
        temperature=args.temperature
    )

    # 评估
    results, stats = evaluator.evaluate_dataset(data, args.max_samples)

    # 打印统计
    print_stats(stats)

    # 保存结果
    save_results(results, stats, args.output)

    print("\n评估完成!")


if __name__ == "__main__":
    main()
