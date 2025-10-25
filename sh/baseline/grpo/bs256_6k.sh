# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x


PROJECT_NAME='xxx'
EXP_NAME='xxx'

MODEL_PATH=xxx


# reward_tag=math-default
reward_tag=math-verify
# reward_tag=symeval

prompt_tag=system-p1

data_train_path=xxx

data_test_aime24="data/AIME24/${reward_tag}/${prompt_tag}/test.parquet"
data_test_aime25="data/AIME25/${reward_tag}/${prompt_tag}/test.parquet"
data_test_amc23="data/AMC23/${reward_tag}/${prompt_tag}/test.parquet"
data_test_math500="data/MATH-500/${reward_tag}/${prompt_tag}/test.parquet"
data_test_gaokao2023en="data/GaoKao2023en/${reward_tag}/${prompt_tag}/test.parquet"
data_test_olympiadbench="data/OlympiadBench/${reward_tag}/${prompt_tag}/test.parquet"
data_test_gsm8k="data/GSM8K/${reward_tag}/${prompt_tag}/test.parquet"
data_test_minerva="data/MinervaMath/${reward_tag}/${prompt_tag}/test.parquet"
data_test_path="['$data_test_aime24', '$data_test_aime25', '$data_test_amc23', '$data_test_math500', '$data_test_gaokao2023en', '$data_test_olympiadbench', '$data_test_gsm8k','$data_test_minerva']"


# ------------------------------------------------------------------------
### train
nnodes=1
n_gpus_per_node=8
vllm_gpu_memory_util=0.8
epoch=100
lr=1e-6
wd=0.0
n_rollout=8
train_temp=1.0
train_batchsize=256
ppo_mini_batchsize=64
micro_batchsize_per_gpu=16

### val
val_batchsize=512

###
save_freq=500
test_freq=10


# ------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_path \
    data.val_files="$data_test_path" \
    data.train_batch_size=${train_batchsize} \
    data.val_batch_size=${val_batchsize} \
    data.max_prompt_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batchsize} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.temperature=${train_temp} \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_memory_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=${wd} \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_epochs=${epoch} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.rollout_data_dir="${EXP_NAME}/rollout_log/training" \
    trainer.validation_data_dir="${EXP_NAME}/rollout_log/validation" \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    trainer.hint_data_dir="${EXP_NAME}/rollout_log/hint" \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME $@



    