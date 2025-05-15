import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional
import json

import numpy as np
# import pandas as pd # Not strictly needed for this version
import torch
import torch.nn as nn
# import torch.nn.functional as F # Not used directly in this version
import tyro
# from accelerate import Accelerator # REMOVED
# from accelerate.state import AcceleratorState # REMOVED
# from accelerate.utils import broadcast, gather_object # REMOVED
from datasets import load_dataset # Dataset not strictly needed but can be useful
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

# Helper functions to keep
def exact_div(a, b): # Not used in this eval script, but kept for completeness from original
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    # Ensure bools is on the correct device for arange
    arange_device = bools.device
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=arange_device)
    return torch.min(zero_or_index, dim=-1).values

# Reused Model classes
class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: Optional[PretrainedConfig] = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        if base_config is None:
            try:
                base_config = AutoConfig.from_pretrained(base_model)
            except Exception:
                class MinimalConfig: pass
                base_config = MinimalConfig()
                base_config.hidden_size = hidden_size
        self.base_config_dict = base_config.to_dict() if hasattr(base_config, 'to_dict') else {}
        self.hidden_size = getattr(base_config, 'hidden_size', hidden_size)
        self.bias = bias

class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.config.base_model = "/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/models/EleutherAI/pythia-1b-deduped/sft_model_1"
        base_config_obj = AutoConfig.from_pretrained(config.base_model, **config.base_config_dict)

        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=base_config_obj,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        backbone_kwargs = {k: v for k, v in kwargs.items() if k in self.lm_backbone.forward.__code__.co_varnames}
        output = self.lm_backbone(**backbone_kwargs)
        if hasattr(output, "last_hidden_state"):
            hidden_states = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        reward = self.scalar_head(hidden_states) - self.config.bias
        return reward


@dataclass
class EvalArgs:
    exp_name: str = field(default_factory=lambda: os.path.basename(__file__)[:-len(".py")] + "_eval" if __file__ else "eval_script_eval")
    seed: int = 1
    cuda: bool = True
    run_name: Optional[str] = None
    load_from_cache_file: bool = False # Not directly used in this eval script
    print_sample_output_freq: int = 10
    base_model: str = "EleutherAI/pythia-160m"
    policy_model_checkpoint_path: str = "models/ppo_model"
    reward_model_path: Optional[str] = None
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    dataset_split: str = "test"
    response_length: int = 53
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    temperature: float = 0.01
    local_eval_batch_size: int = 8
    output_json_path: str = "evaluation_results.json"
    # offload: bool = False # Removed, was for Deepspeed

def parse_eval_args() -> EvalArgs: # Removed Accelerator from return
    args = tyro.cli(EvalArgs)
    # Setup run name if not provided
    if args.run_name is None:
        time_int = int(time.time())
        args.run_name = f"{args.exp_name}_{args.seed}_{time_int}"
    return args

def generate(lm_backbone, queries, tokenizer, generation_config, device):
    """Generate responses using the policy model."""
    context_length = queries.shape[1]
    # Ensure queries are on the correct device
    queries = queries.to(device)
    attention_mask = (queries != tokenizer.pad_token_id).to(device)
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)

    # Ensure model is on the same device as inputs
    lm_backbone = lm_backbone.to(device)

    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def truncate_response(args: EvalArgs, tokenizer: AutoTokenizer, responses: torch.Tensor) -> torch.Tensor:
    """Truncate responses at the first occurrence of `args.truncate_token_id`."""
    if args.truncate_token_id is None:
        raise ValueError("truncate_token_id is not set in args")

    # Ensure responses tensor is on the same device where arange will be created
    arange_device = responses.device
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    token_indices = torch.arange(responses.shape[1], device=arange_device).view(*new_size)
    mask_beyond_trunc = token_indices > trunc_idxs
    postprocessed_responses = torch.masked_fill(responses, mask_beyond_trunc, tokenizer.pad_token_id)
    return postprocessed_responses


def get_score_from_reward_model(
    reward_model: ScalarModel,
    query_responses: torch.Tensor,
    tokenizer: AutoTokenizer,
    context_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculates scalar scores from a reward model for `query_responses`.
    """
    query_responses = query_responses.to(device)
    reward_model = reward_model.to(device)

    attention_mask = (query_responses != tokenizer.pad_token_id).to(device)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    with torch.no_grad():
        reward_output_logits = reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    _seq_lens_in_response_part = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id)
    score_indices = _seq_lens_in_response_part - 1 + context_length
    score_indices = torch.clamp(score_indices, min=context_length, max=query_responses.shape[1]-1)

    scores = reward_output_logits[
        torch.arange(reward_output_logits.size(0), device=device), score_indices # Ensure arange is on device
    ].squeeze(-1)
    return scores


if __name__ == "__main__":
    args = parse_eval_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Setup logging and reproducibility
    # local_seed = args.seed # No process index to add
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed) # For multi-GPU if used without accelerator (though this script is single-GPU focused now)
    # torch.backends.cudnn.deterministic = True # Can slow down, optional

    console = Console(force_terminal=True)
    pprint(asdict(args))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model_checkpoint_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    if args.truncate_token == "eos":
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token, but truncate_token is 'eos'.")
        args.truncate_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(f"Unsupported truncate_token: {args.truncate_token}. Must be 'eos'.")

    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=-1,
        temperature=max(args.temperature, 1e-7),
        top_k=0.0 if args.temperature > 0 else None,
        top_p=1.0 if args.temperature > 0 else None,
        do_sample=True if args.temperature > 0 else False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Load Policy Model
    print(f"Loading policy model from {args.policy_model_checkpoint_path}...")
    policy_model_config = AutoConfig.from_pretrained(args.policy_model_checkpoint_path, trust_remote_code=True)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_checkpoint_path,
        config=policy_model_config,
        trust_remote_code=True
    )
    disable_dropout(policy_model)
    policy_model.eval()
    policy_model = policy_model.to(device)

    # Load Reward Model (optional)
    reward_model = None
    if args.reward_model_path:
        print(f"Loading reward model from {args.reward_model_path}...")
        try:
            reward_model = ScalarModel.from_pretrained(
                args.reward_model_path,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed to load ScalarModel directly: {e}. Check path and model type.")
            raise
        disable_dropout(reward_model)
        reward_model.eval()
        reward_model = reward_model.to(device)
    
    # Load and prepare dataset
    print(f"Loading dataset {args.query_dataset} split {args.dataset_split}...")
    eval_dataset = load_dataset(args.query_dataset, split=args.dataset_split, cache_dir=".cache/huggingface/datasets")
    if "query_token" not in eval_dataset.column_names:
        raise ValueError(f"'query_token' column not found in dataset. Available columns: {eval_dataset.column_names}")
    eval_dataset = eval_dataset.with_format("torch", columns=["query_token"])
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.local_eval_batch_size,
        shuffle=False
    )

    all_results = []
    processed_batches = 0
    
    print("Starting evaluation...")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        queries = batch["query_token"].to(device) # Move batch data to device
        context_length = queries.shape[1]

        # Generate responses
        query_responses = generate(
            policy_model, # Already on device
            queries,
            tokenizer,
            generation_config,
            device # Pass device to generate
        )
        responses_before_truncation = query_responses[:, context_length:]
        
        # Post-process responses (truncate at EOS and pad)
        postprocessed_responses = truncate_response(args, tokenizer, responses_before_truncation)
        postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)

        batch_scores_list = None
        if reward_model:
            scores_tensor = get_score_from_reward_model(
                reward_model, # Already on device
                postprocessed_query_responses,
                tokenizer,
                context_length,
                device # Pass device to scoring function
            )
            batch_scores_list = scores_tensor.cpu().numpy().tolist()

        # Decode for JSON output
        decoded_queries = tokenizer.batch_decode(queries.cpu(), skip_special_tokens=True)
        decoded_summaries = tokenizer.batch_decode(postprocessed_responses.cpu(), skip_special_tokens=True)
        
        batch_results = []
        for i in range(len(decoded_queries)):
            doc = decoded_queries[i]
            summ = decoded_summaries[i]
            score_val = batch_scores_list[i] if batch_scores_list is not None else None
            batch_results.append({"document": doc, "summary": summ, "score": score_val})
        all_results.extend(batch_results)

        processed_batches += 1
        if processed_batches % args.print_sample_output_freq == 0 and len(decoded_queries) > 0:
            console.print(f"\n--- Sample from Batch {processed_batches} ---")
            console.print(f"  Query: {decoded_queries[0][:200]}...")
            console.print(f"  Summary: {decoded_summaries[0]}")
            if batch_scores_list is not None:
                console.print(f"  Score: {batch_scores_list[0]:.4f}")
            console.print("--- End Sample ---")

    # Save to JSON
    output_dir = os.path.dirname(args.output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
            
    with open(args.output_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation results saved to {args.output_json_path}")
    print(f"Total samples evaluated and saved: {len(all_results)}")

    print("Evaluation finished.")