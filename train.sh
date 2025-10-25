conda activate scaf-grpo

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="xxx"


bash sh/hint_mix_grpo/bs256_6k_mix.sh


