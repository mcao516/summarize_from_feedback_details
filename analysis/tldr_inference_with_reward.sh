#!/bin/bash
module load python/3.10
source ~/envTLDR/bin/activate

# accelerate launch tldr_inference_with_reward.py \
#     --policy_model_path "/home/mila/c/caomeng/scratch/caden_shuyuan_share/abc_policy_model_1" \
#     --reward_model_path "/home/mila/c/caomeng/scratch/caden_shuyuan_share/reward_model_1" \
#     --query_dataset "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144" \
#     --dataset_split "test" \
#     --output_json_path "results/generated_test_summaries_with_scores.json" \
#     --response_length 53 \
#     --batch_size 4 \
#     --temperature 0.01

#!/bin/bash

# --- Configuration ---
# Path to your Python evaluation script
PYTHON_SCRIPT="tldr_inference_with_reward.py"

# Path to the trained policy model checkpoint directory
# (This is the directory where your fine-tuned model was saved, e.g., "models/ppo_model")
POLICY_MODEL_PATH="/home/mila/c/caomeng/scratch/caden_shuyuan_share/uniform_policy_model_1800"

# (Optional) Path to the reward model directory for scoring
# If you don't want to use a reward model for scoring, leave this empty or comment it out.
REWARD_MODEL_PATH="/home/mila/c/caomeng/scratch/caden_shuyuan_share/reward_model_1" # Example: "models/reward_model_archive"
# REWARD_MODEL_PATH="" # Uncomment this line if no reward model is used

# Dataset to use for evaluation (from Hugging Face Hub or local path)
QUERY_DATASET="vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"

# Dataset split for evaluation (e.g., "test", "validation")
DATASET_SPLIT="test"

# Path to save the output JSON file
OUTPUT_JSON_PATH="evaluation_outputs/uniform_generated_test_summaries_with_scores.json"

# Whether to use CUDA if available (true/false)
USE_CUDA="true" # or "false"

# Evaluation batch size
EVAL_BATCH_SIZE=8

# Other optional arguments for the Python script can be added here
# For example:
# RESPONSE_LENGTH=60
# TEMPERATURE=0.1
# SEED=42

# --- Construct the command ---
CMD="python ${PYTHON_SCRIPT}"
CMD="${CMD} --policy-model-checkpoint-path ${POLICY_MODEL_PATH}"
CMD="${CMD} --query-dataset ${QUERY_DATASET}"
CMD="${CMD} --dataset-split ${DATASET_SPLIT}"
CMD="${CMD} --output-json-path ${OUTPUT_JSON_PATH}"
CMD="${CMD} --local-eval-batch-size ${EVAL_BATCH_SIZE}"

if [ "${USE_CUDA}" = "true" ]; then
  CMD="${CMD} --cuda"
else
  CMD="${CMD} --no-cuda" # tyro typically handles boolean flags like this or --cuda False
fi

# Add reward model path if it's set
if [ -n "${REWARD_MODEL_PATH}" ]; then
  CMD="${CMD} --reward-model-path ${REWARD_MODEL_PATH}"
fi

# Add any other optional arguments here
# if [ -n "${RESPONSE_LENGTH}" ]; then
#   CMD="${CMD} --response-length ${RESPONSE_LENGTH}"
# fi
# if [ -n "${TEMPERATURE}" ]; then
#   CMD="${CMD} --temperature ${TEMPERATURE}"
# fi
# if [ -n "${SEED}" ]; then
#   CMD="${CMD} --seed ${SEED}"
# fi


# --- Run the command ---
echo "Running evaluation with the following command:"
echo "${CMD}"
echo # Newline for readability

# Execute the command
${CMD}

# Check exit status
if [ $? -eq 0 ]; then
  echo "Evaluation script completed successfully."
else
  echo "Evaluation script failed. Check output for errors."
fi