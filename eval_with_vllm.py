#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 批量评估脚本 - 支持多卡并行推理

相比 eval_with_metrics.py 的优势：
1. 使用 vLLM 进行批量推理，速度提升 5-10 倍
2. 自动利用多卡并行
3. 支持更大的 batch size

使用方法：
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_with_vllm.py \
        --model verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-it-bm25-em/actor/global_step_200 \
        --data data/nq_hotpotqa_train/test_1k.parquet \
        --output eval_results_vllm_bm25_rerank.json \
        --tensor_parallel_size 4
"""

import argparse
import json
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import requests
from tqdm import tqdm
import pandas as pd


@dataclass
class EvalResult:
    """单个样本的评估结果"""
    question: str
    golden_answer: str
    model_answer: str
    extracted_answer: str
    is_correct: bool
    num_searches: int
    response_time: float
    full_trajectory: str
    data_source: str = ""

    all_answers: List[str] = field(default_factory=list)
    num_answers: int = 0
    answer_changed: bool = False


@dataclass
class EvalStats:
    """评估统计"""
    total_samples: int = 0
    correct_samples: int = 0
    accuracy: float = 0.0

    total_searches: int = 0
    avg_searches: float = 0.0
    max_searches: int = 0
    min_searches: int = 0
    search_distribution: Dict[int, int] = field(default_factory=dict)

    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = 0.0

    samples_with_multiple_answers: int = 0
    samples_with_changed_answers: int = 0
    answer_change_rate: float = 0.0
    avg_answers_per_question: float = 0.0
    accuracy_changed_answer: float = 0.0
    accuracy_single_answer: float = 0.0

    by_source: Dict[str, dict] = field(default_factory=dict)


class VLLMEvaluator:
    """vLLM 批量评估器"""

    def __init__(
        self,
        model_path: str,
        retriever_url: str = "http://127.0.0.1:8000/retrieve",
        retriever_topk: int = 3,
        max_turns: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.9
    ):
        self.model_path = model_path
        self.retriever_url = retriever_url
        self.retriever_topk = retriever_topk
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = None
        self.tokenizer = None
        self.eos_token_ids = [151645, 151643]

    def load_model(self):
        """加载 vLLM 模型"""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"使用 vLLM 加载模型: {self.model_path}")
        print(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        print(f"GPU Memory Utilization: {self.gpu_memory_utilization}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16"
        )

        print("✓ vLLM 模型加载完成")

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
        return matches[-1].strip() if matches else None

    def _extract_answer(self, text: str) -> Tuple[str, List[str], int, bool]:
        """提取答案"""
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)

        if not matches:
            return "", [], 0, False

        all_answers = [m.strip() for m in matches]
        normalized_answers = [self._normalize_answer(a) for a in all_answers]
        answer_changed = len(set(normalized_answers)) > 1

        return all_answers[-1], all_answers, len(all_answers), answer_changed

    def _normalize_answer(self, text: str) -> str:
        """标准化答案"""
        text = re.sub(r'\s+', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _check_correct(self, extracted: str, golden: str) -> bool:
        """检查答案是否正确"""
        extracted_lower = extracted.lower().strip()

        # 如果没有提取到答案，直接返回 False
        if not extracted_lower:
            return False

        if isinstance(golden, list):
            golden_list = [str(g).strip().lower() for g in golden]
        else:
            golden_list = [str(golden).strip().lower()]

        for g in golden_list:
            if g in extracted_lower or extracted_lower in g:
                return True
            g_norm = re.sub(r'[^\w\s]', '', g)
            e_norm = re.sub(r'[^\w\s]', '', extracted_lower)
            if g_norm == e_norm:
                return True

        return False

    def evaluate_single(self, question: str, golden_answer: str) -> EvalResult:
        """评估单个问题（多轮交互）"""
        from vllm import SamplingParams

        start_time = time.time()

        prompt = self._build_prompt(question)
        full_trajectory = ""
        num_searches = 0

        # 只在 </search> 停止，不在 </answer> 停止（允许模型生成多个答案）
        stop_sequences = ["</search>"]

        for turn in range(self.max_turns):
            sampling_params = SamplingParams(
                temperature=self.temperature if self.temperature > 0 else 1.0,
                max_tokens=self.max_new_tokens,
                stop=stop_sequences,
                skip_special_tokens=False
            )

            outputs = self.llm.generate([prompt], sampling_params)
            output_text = outputs[0].outputs[0].text
            stop_reason = outputs[0].outputs[0].stop_reason

            # 调试输出：打印原始生成结果
            if turn == 0:
                print(f"\n[DEBUG Turn {turn}] Raw output_text: {repr(output_text[:200])}")
                print(f"[DEBUG Turn {turn}] Stop reason: {stop_reason}")

            # vLLM 的 stop_reason 会是停止序列本身，或者 None (EOS/max_tokens)
            if stop_reason == "</search>" or stop_reason == "stop":
                # vLLM 遇到 </search> 停止，但不包含在 output_text 中，需要补上
                output_text = output_text + "</search>"
            elif '</search>' in output_text:
                # 如果模型在 max_tokens 前生成了 </search>，截断
                output_text = output_text.split('</search>')[0] + '</search>'
            elif '</answer>' in output_text:
                # 在 </answer> 处截断（但不停止循环）
                output_text = output_text.split('</answer>')[0] + '</answer>'

            full_trajectory += output_text

            # 检查是否有 search
            query = self._get_query(output_text)

            if query:
                # 有搜索，调用检索
                num_searches += 1
                search_results = self.search(query)
                search_text = f"\n\n<information>{search_results}</information>\n\n"
                full_trajectory += search_text
                prompt += output_text + search_text
            elif '</answer>' in output_text:
                # 已经给出答案但没有新的搜索，继续到 max_turns
                prompt += output_text
            else:
                # 既没有 search 也没有 answer，或者遇到 EOS，结束
                break

        # 如果达到 max_turns 还没有答案，强制生成一个答案
        if num_searches > 0 and '</answer>' not in full_trajectory:
            print(f"[WARNING] 达到 max_turns={self.max_turns} 但没有答案，强制生成答案...")

            # 添加提示让模型给出最终答案
            prompt += "\n\n<think>Based on the information gathered, I should now provide my final answer.</think>\n<answer>"

            # 生成答案（不设置 stop，让模型生成到 </answer>）
            sampling_params = SamplingParams(
                temperature=self.temperature if self.temperature > 0 else 1.0,
                max_tokens=100,
                stop=["</answer>"],
                skip_special_tokens=False
            )

            outputs = self.llm.generate([prompt], sampling_params)
            answer_text = outputs[0].outputs[0].text

            # 补全答案标签
            full_trajectory += f"\n\n<think>Based on the information gathered, I should now provide my final answer.</think>\n<answer>{answer_text}</answer>"

        end_time = time.time()
        response_time = end_time - start_time

        extracted_answer, all_answers, num_answers, answer_changed = self._extract_answer(full_trajectory)
        is_correct = self._check_correct(extracted_answer, golden_answer)

        # 调试输出：如果没有搜索或答案，打印完整轨迹
        if num_searches == 0 or not extracted_answer:
            print(f"\n[DEBUG] No searches or answers found!")
            print(f"  num_searches: {num_searches}")
            print(f"  extracted_answer: {repr(extracted_answer)}")
            print(f"  full_trajectory: {repr(full_trajectory[:500])}")

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
        if self.llm is None:
            self.load_model()

        if max_samples and len(data) > max_samples:
            import random
            data = random.sample(data, max_samples)
            print(f"随机采样 {max_samples} 个样本进行评估")

        results = []
        stats = EvalStats()

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

                stats.total_samples += 1
                if result.is_correct:
                    stats.correct_samples += 1
                stats.total_searches += result.num_searches
                stats.total_time += result.response_time

                stats.search_distribution[result.num_searches] = \
                    stats.search_distribution.get(result.num_searches, 0) + 1

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

            total_answers = sum(r.num_answers for r in results)
            stats.avg_answers_per_question = total_answers / stats.total_samples
            stats.samples_with_multiple_answers = sum(1 for r in results if r.num_answers > 1)
            stats.samples_with_changed_answers = sum(1 for r in results if r.answer_changed)
            stats.answer_change_rate = stats.samples_with_changed_answers / stats.total_samples

            changed_results = [r for r in results if r.answer_changed]
            single_results = [r for r in results if r.num_answers == 1]

            if changed_results:
                stats.accuracy_changed_answer = sum(1 for r in changed_results if r.is_correct) / len(changed_results)
            if single_results:
                stats.accuracy_single_answer = sum(1 for r in single_results if r.is_correct) / len(single_results)

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

    if stats.total_samples == 0:
        print("  警告: 没有成功评估的样本")
        return

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

    print(f"\n【答案一致性统计】")
    print(f"  平均答案数/题: {stats.avg_answers_per_question:.2f}")
    print(f"  多答案样本数: {stats.samples_with_multiple_answers} ({stats.samples_with_multiple_answers/stats.total_samples*100:.1f}%)")
    print(f"  改变答案样本数: {stats.samples_with_changed_answers} ({stats.answer_change_rate*100:.1f}%)")

    if stats.samples_with_changed_answers > 0:
        print(f"\n  准确率对比:")
        print(f"    改变答案的题目: {stats.accuracy_changed_answer*100:.2f}%")
        print(f"    单答案的题目:   {stats.accuracy_single_answer*100:.2f}%")

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

    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("question,golden_answer,extracted_answer,is_correct,num_searches,response_time,data_source\n")
        for r in results:
            q = r.question.replace('"', '""')[:100]
            g = str(r.golden_answer).replace('"', '""')  # 不再截断
            e = r.extracted_answer.replace('"', '""')  # 不再截断
            f.write(f'"{q}","{g}","{e}",{r.is_correct},{r.num_searches},{r.response_time:.2f},"{r.data_source}"\n')

    print(f"✓ CSV 已保存到: {csv_path}")

    # 保存 BadCase 详细分析
    badcase_path = output_path.replace('.json', '_badcases.txt')
    incorrect_results = [r for r in results if not r.is_correct]

    with open(badcase_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"BadCase 分析报告 (共 {len(incorrect_results)} 个错误)\n")
        f.write("=" * 100 + "\n\n")

        for idx, r in enumerate(incorrect_results, 1):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"BadCase #{idx}/{len(incorrect_results)}\n")
            f.write(f"{'=' * 100}\n\n")

            f.write(f"问题: {r.question}\n")
            f.write(f"数据源: {r.data_source}\n")
            f.write(f"标准答案: {r.golden_answer}\n")
            f.write(f"模型答案: {r.extracted_answer}\n")
            f.write(f"搜索次数: {r.num_searches}\n")
            f.write(f"答案数量: {r.num_answers}\n")
            f.write(f"是否改变答案: {'是' if r.answer_changed else '否'}\n")

            if r.answer_changed:
                f.write(f"所有答案: {r.all_answers}\n")

            f.write(f"\n完整轨迹:\n")
            f.write("-" * 100 + "\n")
            f.write(r.full_trajectory)
            f.write("\n" + "-" * 100 + "\n")

    print(f"✓ BadCase 分析已保存到: {badcase_path}")
    print(f"  共 {len(incorrect_results)} 个错误样本")


def main():
    parser = argparse.ArgumentParser(description="vLLM 批量评估脚本")
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据路径')
    parser.add_argument('--output', type=str, default='eval_results_vllm.json', help='输出结果路径')
    parser.add_argument('--retriever_url', type=str, default='http://127.0.0.1:8000/retrieve', help='检索服务URL')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数')
    parser.add_argument('--max_turns', type=int, default=4, help='最大检索轮数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Tensor Parallel 大小（GPU数量）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU 内存利用率')

    args = parser.parse_args()

    print("=" * 70)
    print("vLLM 批量评估配置")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"检索服务: {args.retriever_url}")
    print(f"最大样本数: {args.max_samples or '全部'}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=" * 70)

    # 加载数据
    data = load_data(args.data)
    print(f"加载了 {len(data)} 个样本")

    # 初始化评估器
    evaluator = VLLMEvaluator(
        model_path=args.model,
        retriever_url=args.retriever_url,
        max_turns=args.max_turns,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
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
