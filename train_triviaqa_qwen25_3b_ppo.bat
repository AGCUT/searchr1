@echo off
REM TriviaQA PPO Training Script
REM Model: Qwen2.5-3B-Instruct
REM Retriever: BM25 (CPU-based)

echo ============================================
echo TriviaQA Training with Qwen2.5-3B-Instruct
echo ============================================

set DATA_DIR=D:\search-r1\Search-R1\data\triviaqa_search
set BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
set EXPERIMENT_NAME=triviaqa-search-r1-ppo-qwen2.5-3b-instruct-bm25-em
set VLLM_ATTENTION_BACKEND=XFORMERS
set PYTHONUNBUFFERED=1

echo Data directory: %DATA_DIR%
echo Base model: %BASE_MODEL%
echo Experiment name: %EXPERIMENT_NAME%
echo ============================================

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

echo ============================================
echo Training completed!
echo Checkpoints saved to: ./checkpoints/%EXPERIMENT_NAME%
echo ============================================
