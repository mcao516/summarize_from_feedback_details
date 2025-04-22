#!/bin/bash

module load python/3.10
source ~/p310/bin/activate

SEED=1
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    MODEL=EleutherAI/pythia-410m-deduped
fi
LR=3e-6
REWARD_MODEL_PATH=/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/models/EleutherAI/pythia-410m-deduped/reward_model_$SEED
SFT_MODEL_PATH=/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/models/EleutherAI/pythia-410m-deduped/sft_model_$SEED
POLICY_MODEL_PATH=/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/models/EleutherAI/pythia-410m-deduped/policy_model_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=64 # smaller fits better on GPU
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU
local_eval_batch_size=16 # smaller fits better on GPU

# 1. you want to make sure gradient_accumulation_steps * local_micro_batch_size = 64
# so you have the same hyperparameters as the paper
# 2. if you are running on a single GPU, you want to make sure 
# gradient_accumulation_steps * local_micro_batch_size = 512 to have the same hyperparameters

# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/sft.py \
#     --base_model=$MODEL \
#     --lr=$LR \
#     --deepspeed \
#     --track \
#     --output_dir=$SFT_MODEL_PATH \
#     --local_micro_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --run_eval \
#     --local_eval_batch_size 16 \
#     --seed=$SEED

# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/reward.py \
#     --base_model=$MODEL \
#     --sft_model_path=$SFT_MODEL_PATH \
#     --lr=$LR \
#     --deepspeed \
#     --run_eval \
#     --track \
#     --output_dir=$REWARD_MODEL_PATH \
#     --local_eval_batch_size=$local_eval_batch_size \
#     --local_micro_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --seed=$SEED

poetry run accelerate launch --config_file deepspeed.yaml \
     summarize_from_feedback_details/ppo.py \
      --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
      --gradient_accumulation_steps=$gradient_accumulation_steps \
      --local_micro_batch_size=$local_micro_batch_size \
      --local_eval_batch_size=$local_eval_batch_size \
      --base_model=$MODEL \
      --sft_model_path=$SFT_MODEL_PATH \
      --reward_model_path=$REWARD_MODEL_PATH \
      --lr=$LR \
      --deepspeed \
      --run_eval \
      --track \
      --output_dir=$POLICY_MODEL_PATH \
      --seed=$SEED \
      --print_sample_output_freq=100 \
      --ppo.kl_coef=0.05 \
      --use_dense_rewards \
      --reward_type "attr_lig"
