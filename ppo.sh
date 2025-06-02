#!/bin/bash

module load python/3.10
source ~/envTLDR/bin/activate

if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    MODEL=EleutherAI/pythia-410m-deduped
fi
LR=3e-6

# Timestamp for the entire batch of runs (if you want a unique timestamp
# for *each individual seed run*, move this line inside the loop)
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

REWARD_MODEL_PATH=$SCRATCH/summarize_from_feedback_details/models/$MODEL/reward_model_2
SFT_MODEL_PATH=$SCRATCH/summarize_from_feedback_details/models/$MODEL/sft_model_2

# Vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=32 # smaller fits better on GPU
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU
local_eval_batch_size=16 # smaller fits better on GPU

# Define the seeds to iterate over
SEEDS_TO_RUN=(1) # Or use: SEEDS_TO_RUN=($(seq 1 5))

echo "Starting batch of runs with timestamp: $BATCH_TIMESTAMP"
echo "Target model: $MODEL"
echo "Learning rate: $LR"
echo "SFT Model: $SFT_MODEL_PATH"
echo "Reward Model: $REWARD_MODEL_PATH"
echo "Iterating over seeds: ${SEEDS_TO_RUN[*]}"
echo "-----------------------------------------------------"

for CURRENT_SEED in "${SEEDS_TO_RUN[@]}"
do
    echo "#####################################################"
    echo "## Processing SEED: $CURRENT_SEED"
    echo "#####################################################"

    SEED=$CURRENT_SEED
    POLICY_MODEL_PATH=$SCRATCH/summarize_from_feedback_details/models/$MODEL/policy_model_${SEED}_${BATCH_TIMESTAMP}

    echo "Policy model for SEED $SEED will be saved to: $POLICY_MODEL_PATH"

    # --total_episodes 116722
    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/ppo.py \
            --local_rollout_forward_batch_size $local_rollout_forward_batch_size \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --local_micro_batch_size $local_micro_batch_size \
            --local_eval_batch_size $local_eval_batch_size \
            --total_episodes 116722 \
            --base_model $MODEL \
            --sft_model_path $SFT_MODEL_PATH \
            --reward_model_path $REWARD_MODEL_PATH \
            --lr $LR \
            --deepspeed \
            --run_eval \
            --track \
            --output_dir $POLICY_MODEL_PATH \
            --save_frequency 200 \
            --seed $SEED \
            --print_sample_output_freq 100 \
            --ppo.kl_coef 0.05 \
            --wandb_entity "caden" \
            --wandb_project_name "summarize_from_feedback" \
            --exp_name "sparse" \
            --reward_type "sparse" \

    echo "## Finished processing SEED: $CURRENT_SEED"
    echo "-----------------------------------------------------"
done

echo "All runs completed for seeds: ${SEEDS_TO_RUN[*]}."