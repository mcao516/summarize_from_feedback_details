#!/bin/bash
module load python/3.10
source ~/envTLDR/bin/activate

python tldr_inference.py \
    --model-path /home/mila/c/caomeng/scratch/caden_shuyuan_share/abc_policy_model_1 \
    --query-dataset-name "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144" \
    --dataset-split "test" \
    --output-file "generated_test_summaries.json" \
    --response-length 53 \
    --temperature 0.01 \
    --batch-size 8