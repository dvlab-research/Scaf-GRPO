conda activate scaf-grpo

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1

set -x


model_path=xxx
data_path=xxx
save_root_dir=xxx

n_gpus_per_node=8
gpu_memory_utilization=0.9
tensor_model_parallel_size=1
max_num_batched_tokens=32768
log_prob_micro_batch_size_per_gpu=16

prompt_length=2048
response_length=2048
batch_size=1024
n_samples=8
do_sample=True
temperature=1.0
top_p=1.0
top_k=-1


save_path=${save_root_dir}/prompt-length-${prompt_length}__response-length-${response_length}__batchsize-${batch_size}__n-${n_samples}__temp-${temperature}__topp-${top_p}__topk-${top_k}__do-sample-${do_sample}/generation_output.parquet


## -------------------------- generation --------------------------

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    data.path=$data_path \
    data.batch_size=$batch_size \
    data.prompt_key=prompt \
    data.n_samples=$n_samples \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.do_sample=$do_sample \
    rollout.temperature=$temperature \
    rollout.top_p=$top_p \
    rollout.top_k=$top_k \
    rollout.prompt_length=$prompt_length \
    rollout.response_length=$response_length \
    rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    rollout.gpu_memory_utilization=$gpu_memory_utilization \
    rollout.max_num_batched_tokens=$max_num_batched_tokens \
    rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu

sleep 30

## -------------------------- evaluation --------------------------

python3 -m verl.trainer.main_eval \
    data.path=$save_path \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model
    # data.extra_info_key=extra_info
