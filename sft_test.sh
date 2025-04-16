module load python/3.10
# source ~/envTLDR/bin/activate

SEED=1
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
LR=3e-6
REWARD_MODEL_PATH=$SCRATCH/TLDR/models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=$SCRATCH/TLDR/models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=$SCRATCH/TLDR/models/$MODEL/policy_model_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=64 # smaller fits better on GPU
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU
local_eval_batch_size=16 # smaller fits better on GPU

# 1. you want to make sure gradient_accumulation_steps * local_micro_batch_size = 64
# so you have the same hyperparameters as the paper
# 2. if you are running on a single GPU, you want to make sure 
# gradient_accumulation_steps * local_micro_batch_size = 512 to have the same hyperparameters

# --local_eval_batch_size 4 \
poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/sft.py \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --local_micro_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --run_eval \
    --seed=$SEED \