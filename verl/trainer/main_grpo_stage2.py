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
Stage 2 GRPO Training: Efficiency Optimization

在第一阶段 PPO/GRPO 训练完成后，使用本入口进行第二阶段训练：
- 奖励函数：EM + 效率奖励（检索次数越少，奖励越高）
- 目标：在保持答案正确率的前提下，减少不必要的检索次数
"""
import os
os.environ['NCCL_TIMEOUT'] = '1800'

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em_efficiency
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


def _select_rm_score_fn(data_source):
    """选择带效率奖励的评分函数"""
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em_efficiency.compute_score_em
    else:
        raise NotImplementedError


def _select_em_score_fn(data_source):
    """选择纯 EM 评分函数（用于日志记录）"""
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em_efficiency.compute_em_score
    else:
        raise NotImplementedError


class EfficiencyRewardManager:
    """
    第二阶段效率奖励管理器

    奖励计算：
    - 答案正确：EM_score (1.0) + efficiency_score
    - 答案错误：0

    效率奖励计算：
    - efficiency_score = avg_scaled_time - scaled_time[i]
    - 检索次数少于平均 → 正奖励
    - 检索次数多于平均 → 负奖励
    """

    def __init__(self, tokenizer, num_examine=1) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        print("[Stage2-Efficiency] Reward Manager initialized")
        print("[Stage2-Efficiency] Reward = EM + efficiency_bonus")

    def __call__(self, data: DataProto):
        """计算带效率奖励的总奖励"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        batch_size = len(data)

        # 获取检索次数并计算效率奖励
        # 注意：框架中记录的是 valid_search_stats，不是 search_times
        search_times = data.meta_info.get('valid_search_stats', None)
        if search_times is None:
            search_times = data.meta_info.get('search_times', None)

        # 确保 search_times 长度与 batch_size 一致
        if search_times is None:
            search_times = [0] * batch_size
            print(f"[Stage2-Warning] search_times not found in meta_info, using zeros")
        elif len(search_times) != batch_size:
            print(f"[Stage2-Warning] search_times length ({len(search_times)}) != batch_size ({batch_size})")
            # 如果长度不匹配，可能是因为 n_agent > 1，尝试扩展
            if batch_size % len(search_times) == 0:
                repeat_factor = batch_size // len(search_times)
                search_times = [t for t in search_times for _ in range(repeat_factor)]
                print(f"[Stage2-Info] Expanded search_times by factor {repeat_factor}")
            else:
                # 无法对齐，使用 0
                search_times = [0] * batch_size
                print(f"[Stage2-Warning] Cannot align, using zeros")

        # 效率奖励缩放系数（降低效率奖励的影响，防止遗忘）
        efficiency_scale = 0.2  # 效率奖励范围 [-0.2, +0.2]

        if len(search_times) > 0 and max(search_times) > 0:
            # 计算 95 分位数进行归一化
            percentile_95 = np.percentile(search_times, 95)
            percentile_95 = max(percentile_95, 1.0)  # 避免除零

            scaled_times = []
            for t in search_times:
                scaled_time = t * efficiency_scale / percentile_95
                scaled_time = min(scaled_time, efficiency_scale)
                scaled_times.append(scaled_time)

            avg_scaled_time = sum(scaled_times) / len(scaled_times)
        else:
            scaled_times = [0] * len(data)
            avg_scaled_time = 0

        already_print_data_sources = {}
        em_scores = []
        efficiency_scores = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            # 计算效率奖励
            efficiency_score = avg_scaled_time - scaled_times[i]
            efficiency_scores.append(efficiency_score)

            # 计算总奖励（EM + 效率）
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                efficiency_score=efficiency_score
            )

            # 计算纯 EM 分数用于日志
            compute_em_fn = _select_em_score_fn(data_source)
            em_score = compute_em_fn(solution_str=sequences_str, ground_truth=ground_truth)
            em_scores.append(em_score)

            reward_tensor[i, valid_response_length - 1] = score

            # 打印样本
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[Stage2] Total: {score:.4f}, EM: {em_score:.4f}, "
                      f"Efficiency: {efficiency_score:.4f}, Searches: {search_times[i]}")
                print(sequences_str[:500] + "..." if len(sequences_str) > 500 else sequences_str)

        # 记录统计信息
        data.meta_info['em_scores'] = em_scores
        data.meta_info['efficiency_scores'] = efficiency_scores

        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
        avg_eff = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        avg_searches = sum(search_times) / len(search_times) if search_times else 0

        print(f"[Stage2-Batch] Avg EM: {avg_em:.4f}, Avg Efficiency: {avg_eff:.4f}, Avg Searches: {avg_searches:.2f}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'NCCL_TIMEOUT': '1800'
            }
        })

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    import torch
    import numpy as np
    import random

    def set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    seed = config.get('seed', '')
    if seed != '':
        set_seed(seed)

    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download checkpoint
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    # 注意：GRPO 不需要 Critic，所以不检查 critic.strategy
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # GRPO 只需要 ActorRollout 和 RefPolicy，不需要 Critic
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 使用效率奖励管理器
    reward_fn = EfficiencyRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = EfficiencyRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
