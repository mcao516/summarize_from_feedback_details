import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional
import json

import numpy as np
import torch
import torch.nn as nn
import tyro
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

# Helper functions (kept from original)
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
    arange_device = bools.device
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=arange_device)
    return torch.min(zero_or_index, dim=-1).values

# Reused Model classes (ScalarModelConfig, ScalarModel - unchanged)
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
        # Ensure base_config_obj is serializable for from_config
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
class ScoreArgs:
    exp_name: str = field(default_factory=lambda: os.path.basename(__file__)[:-len(".py")] + "_score" if __file__ else "score_script_score")
    seed: int = 1
    cuda: bool = True
    run_name: Optional[str] = None
    print_sample_output_freq: int = 10
    
    base_model: Optional[str] = None # If reward model config doesn't specify, or for tokenizer
    reward_model_path: str = "models/reward_model_pythia160m" # Mandatory
    input_json_path: str = "generated_summaries.json" # Path to JSON with "document" and "summary"
    
    max_query_length: Optional[int] = None # Max length for tokenized query
    max_summary_length: Optional[int] = None # Max length for tokenized summary
    
    local_eval_batch_size: int = 8 # Renamed for clarity, was local_batch_size
    output_json_path: str = "scored_results.json"

    score_field_name: str = "score" 


def parse_score_args() -> ScoreArgs:
    args = tyro.cli(ScoreArgs)
    if args.run_name is None:
        time_int = int(time.time())
        args.run_name = f"{args.exp_name}_{args.seed}_{time_int}"
    if not args.reward_model_path:
        raise ValueError("reward_model_path must be specified.")
    return args


# get_score_from_reward_model - unchanged from original eval script
def get_score_from_reward_model(
    reward_model: ScalarModel,
    query_responses: torch.Tensor, # This is input_ids
    attention_mask: torch.Tensor, # Pass attention_mask explicitly
    tokenizer: AutoTokenizer,
    context_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculates scalar scores from a reward model for `query_responses`.
    `query_responses` are assumed to be `query_tokens + summary_tokens`.
    `context_length` is the length of the `query_tokens` part.
    """
    query_responses = query_responses.to(device)
    attention_mask = attention_mask.to(device) # Ensure mask is on device
    reward_model = reward_model.to(device)

    # The original script did:
    # attention_mask = (query_responses != tokenizer.pad_token_id).to(device)
    # input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    # Here, we receive query_responses as input_ids and a pre-computed attention_mask.

    with torch.no_grad():
        reward_output_logits = reward_model(
            input_ids=query_responses, # Use the correctly padded input_ids
            attention_mask=attention_mask,
        )

    # Find the last non-pad token in the response part
    # query_responses[:, context_length:] is the response part
    # (query_responses[:, context_length:] == tokenizer.pad_token_id) marks padding tokens in response
    response_part = query_responses[:, context_length:]
    # Need to consider the attention mask for the response part to correctly find the end
    response_attention_mask = attention_mask[:, context_length:]
    
    # The first pad token after context_length, OR end of sequence if no pad
    # A simpler way: sum of attention mask for the response part gives its actual length
    _seq_lens_in_response_part = torch.sum(response_attention_mask, dim=1)
    
    # If response part is empty (e.g. all pad), seq_len is 0. Score index should be context_length.
    # If response part has length L, score index is context_length + L - 1.
    score_indices = context_length + _seq_lens_in_response_part - 1
    
    # Clamp score_indices to be at least context_length (for empty responses)
    # and at most query_responses.shape[1]-1 (for full responses without padding)
    score_indices = torch.clamp(score_indices, min=context_length, max=query_responses.shape[1] - 1)
    
    # Handle cases where _seq_lens_in_response_part is 0 (empty response).
    # In this case, score_indices becomes context_length - 1. We want context_length.
    # This means we take the score from the last token of the query.
    # Or, if we want to score an "empty" response as low, this needs adjustment.
    # The original code clamps min to context_length.
    # If response length is 0, score_indices = context_length + 0 -1 = context_length -1.
    # Clamp(context_length-1, min=context_length) makes it context_length.
    # This means if summary is empty, it takes score from the last token of query.
    # This seems fine as per original logic.

    scores = reward_output_logits[
        torch.arange(reward_output_logits.size(0), device=device), score_indices
    ].squeeze(-1)
    return scores


# Custom Dataset for loading JSON data
class JsonDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading or parsing JSON from {data_path}: {e}")

        if not isinstance(self.data, list):
            raise ValueError(f"JSON file {data_path} must contain a list of objects.")
        if not self.data:
            raise ValueError(f"JSON file {data_path} is empty.")
        if not all(isinstance(item, dict) and "document" in item and "summary" in item for item in self.data):
            raise ValueError("Each item in the JSON list must be a dictionary with 'document' and 'summary' keys.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Return the full item to preserve other fields if any
        return {"document": item["document"], "summary": item["summary"], "original_item": item}


if __name__ == "__main__":
    args = parse_score_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    console = Console(force_terminal=True)
    pprint(asdict(args))

    # Load Reward Model Config to determine base model for tokenizer
    try:
        reward_config = ScalarModelConfig.from_pretrained(args.reward_model_path)
        tokenizer_model_name = reward_config.base_model
        print(f"Using tokenizer from reward model's base: {tokenizer_model_name}")
    except Exception as e:
        print(f"Could not load ScalarModelConfig from {args.reward_model_path}: {e}")
        if args.base_model:
            tokenizer_model_name = args.base_model
            print(f"Falling back to args.base_model for tokenizer: {tokenizer_model_name}")
        else:
            raise ValueError("Cannot determine tokenizer. Please specify args.base_model or ensure reward_model_path contains a valid ScalarModelConfig.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        padding_side="right", # Important for how context_length is used
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # E.g., GPT2
    
    # Determine model_max_length for tokenizer
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 4096: # Safety for large values
        try:
            model_config = AutoConfig.from_pretrained(tokenizer_model_name)
            tokenizer.model_max_length = model_config.max_position_embeddings
            print(f"Set tokenizer.model_max_length to {tokenizer.model_max_length} from model config.")
        except Exception:
            print(f"Warning: Could not determine tokenizer.model_max_length from {tokenizer_model_name} config. Using 1024.")
            tokenizer.model_max_length = 1024
    
    max_total_len = tokenizer.model_max_length

    # Determine max_query_length and max_summary_length
    if args.max_query_length is None:
        # Default: allocate roughly half to query, ensuring some space for summary
        # Subtract a small buffer for special tokens if they are added outside these lengths.
        # For simplicity, let's assume BOS/EOS are part of the tokenized query/summary.
        mql = max_total_len // 2 
    else:
        mql = args.max_query_length
    
    if args.max_summary_length is None:
        # Remaining length for summary
        msl = max_total_len - mql 
    else:
        msl = args.max_summary_length

    if mql + msl > max_total_len:
        print(f"Warning: max_query_length ({mql}) + max_summary_length ({msl}) > tokenizer.model_max_length ({max_total_len}).")
        print(f"Adjusting max_summary_length to {max_total_len - mql}")
        msl = max_total_len - mql
    
    if mql <=0 or msl <=0:
        raise ValueError(f"Max query ({mql}) or summary ({msl}) length is non-positive. Check tokenizer.model_max_length ({max_total_len}) and provided lengths.")

    print(f"Effective max_query_length: {mql}, max_summary_length: {msl}, total_max_len: {max_total_len}")

    # Load Reward Model
    print(f"Loading reward model from {args.reward_model_path}...")
    try:
        reward_model = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load ScalarModel from {args.reward_model_path}: {e}")
        raise
    disable_dropout(reward_model)
    reward_model.eval()
    reward_model = reward_model.to(device)
    
    # Load dataset from JSON
    print(f"Loading data from {args.input_json_path}...")
    score_dataset = JsonDataset(args.input_json_path)
    
    # Manual batching, similar to original script's dataloader loop
    all_scored_results = []
    
    print("Starting scoring...")
    for i in tqdm(range(0, len(score_dataset), args.local_eval_batch_size), desc="Scoring Batches"):
        batch_data_items = [score_dataset[j] for j in range(i, min(i + args.local_eval_batch_size, len(score_dataset)))]
        
        queries_str_batch = [item['document'] for item in batch_data_items]
        summaries_str_batch = []
        for item in batch_data_items:
            if item['summary'].endswith(tokenizer.eos_token):
                summaries_str_batch.append(item['summary'])
            else:
                summaries_str_batch.append(item['summary'] + tokenizer.eos_token)

        # Tokenize queries
        # Add special tokens (e.g. BOS) for queries.
        # Padding to longest in batch, truncate to mql.
        tokenized_queries = tokenizer(
            queries_str_batch,
            padding="longest",
            padding_side="left",
            truncation=True,
            max_length=mql,
            return_tensors="pt",
        )
        query_input_ids = tokenized_queries.input_ids
        query_attention_mask = tokenized_queries.attention_mask
        
        context_length = query_input_ids.shape[1] # Length of the (padded) query part

        # Tokenize summaries
        # Do NOT add special tokens like BOS if query already has it and they are concatenated.
        # EOS might be added by tokenizer or desired. For scoring, often it's not strictly needed if using attention mask.
        # Let's be consistent with typical Causal LM tokenization: query ends, summary begins.
        tokenized_summaries = tokenizer(
            summaries_str_batch,
            padding="longest",
            padding_side="right",
            truncation=True,
            max_length=msl,
            return_tensors="pt",
        )
        summary_input_ids = tokenized_summaries.input_ids
        summary_attention_mask = tokenized_summaries.attention_mask

        # Concatenate query and summary tokens
        # query_input_ids: [batch, q_len]
        # summary_input_ids: [batch, s_len]
        # result: [batch, q_len + s_len]
        concatenated_input_ids = torch.cat((query_input_ids, summary_input_ids), dim=1)
        concatenated_attention_mask = torch.cat((query_attention_mask, summary_attention_mask), dim=1)

        # Ensure total length does not exceed model_max_length
        if concatenated_input_ids.shape[1] > max_total_len:
            concatenated_input_ids = concatenated_input_ids[:, :max_total_len]
            concatenated_attention_mask = concatenated_attention_mask[:, :max_total_len]

        scores_tensor = get_score_from_reward_model(
            reward_model,
            concatenated_input_ids,
            concatenated_attention_mask, # Pass the combined attention mask
            tokenizer,
            context_length, # This is q_len
            device,
        )
        batch_scores_list = scores_tensor.cpu().numpy().tolist()

        for idx, original_item_data in enumerate(batch_data_items):
            doc = original_item_data['document']
            summ = original_item_data['summary']
            score_val = batch_scores_list[idx]
            
            # Preserve original fields and add score
            scored_item = original_item_data["original_item"].copy()
            scored_item[args.score_field_name] = score_val
            all_scored_results.append(scored_item)

        if (i // args.local_eval_batch_size + 1) % args.print_sample_output_freq == 0 and len(queries_str_batch) > 0:
            console.print(f"\n--- Sample from Batch {i // args.local_eval_batch_size + 1} ---")
            console.print(f"  Query: {queries_str_batch[0][:200]}...")
            console.print(f"  Summary: {summaries_str_batch[0][:200]}...")
            console.print(f"  Score: {batch_scores_list[0]:.4f}")
            console.print("--- End Sample ---")

    # Save to JSON
    output_dir = os.path.dirname(args.output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_scored_results, f, indent=2, ensure_ascii=False)
    print(f"\nScored results saved to {args.output_json_path}")
    print(f"Total samples scored and saved: {len(all_scored_results)}")

    print("Scoring finished.")