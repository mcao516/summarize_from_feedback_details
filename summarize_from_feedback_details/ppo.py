import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

import shap
from summarize_from_feedback_details.shap_reward import get_shap_rewards
from summarize_from_feedback_details.shap_utils import parse_sentence

torch.set_printoptions(precision=4, sci_mode=False)
api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False
    kl_coef: float = 0.05


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 64
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = True
    """whether to penalize responses that do not contain `truncate_token_id`"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = ""
    """the path to the reward model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/ppo_model"
    """Where to save the model"""

    # mine
    reward_type: str = "sparse"
    """Reward type"""
    sparse_weight: float = 1.0
    """Weight of sparse rewards"""
    dense_weight: float = 1.0
    """Weight of dense rewards"""
    save_frequency: int = 0
    """Save checkpoint every N updates (PPO epochs). 0 means only at the end. Recommended value: 50, 100, or 200"""
    ppo: PpoHParams = field(default_factory=PpoHParams)


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps * args.nminibatches
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
    args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
    if args.ppo.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_updates = args.total_episodes // args.batch_size
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        return_lm_hidden_states = kwargs.pop("return_lm_hidden_states", False)

        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias

        if return_lm_hidden_states:
            return reward, output
        else:
            return reward


def get_reward(
    model,
    query_responses,
    tokenizer,
    context_length,
    reward_type="sparse",
):
    """
    Args:
        query_responses: [curr_batch_size, query_responses_len]

    Returns:
        reward_logits: [curr_batch_size, query_responses_len, 1]
        score: [curr_batch_size]
        sequence_lengths: [curr_batch_size]
    """
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits, rm_hidden_states = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=False,
        return_lm_hidden_states=True,
    )
    # reward_logits: [batch_size, query_responses_len, 1]
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    seq_rewards = reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1)
    if reward_type == "sparse":
        return (
            reward_logits,
            seq_rewards,
            sequence_lengths,
        )
    elif reward_type == "dummy_dense":
        batch_size = reward_logits.size(0)
        response_max_length = query_responses.size(1) - context_length
        dense_rewards = torch.zeros(batch_size, response_max_length).to(reward_logits)
        return (
            dense_rewards,
            seq_rewards,
            sequence_lengths,
        )
    elif reward_type == "abc":
        raise NotImplementedError
    elif reward_type == "shap_token":
        batch_size = reward_logits.size(0)
        response_max_length = query_responses.size(1) - context_length
        dense_rewards = np.zeros((batch_size, response_max_length))
        contain_eos_token = torch.any(query_responses[:, context_length:] == tokenizer.eos_token_id, dim=-1)
        for i in range(batch_size):
            if contain_eos_token[i]:
                try:
                    query_str = tokenizer.decode(query_responses[i][:context_length], skip_special_tokens=True)
                    response_str = tokenizer.decode(query_responses[i][context_length:], skip_special_tokens=True)
                    shap_outputs = get_shap_rewards(model, query_str, response_str, tokenizer)
                    m = shap_outputs.values.shape[1]
                    response_length = sequence_lengths[i] - context_length
                    if m != response_length:
                        raise ValueError(f"SHAP values [{i}] has length {m} greater than the response length {response_length}.")
                    dense_rewards[i, :m] = shap_outputs.values
                except Exception as e:
                    print("SHAP Exception: ", e)
            else:
                print(f"Sample [{i}] has no <eos> token. Skip SHAP calculation.")
        dense_rewards = torch.tensor(dense_rewards).to(reward_logits)
        return (
            dense_rewards,
            seq_rewards,
            sequence_lengths,
        )
    elif reward_type == "shap_span":
        batch_size = reward_logits.size(0)
        response_max_length = query_responses.size(1) - context_length
        dense_rewards = np.zeros((batch_size, response_max_length))
        contain_eos_token = torch.any(query_responses[:, context_length:] == tokenizer.eos_token_id, dim=-1)
        for i in range(batch_size):
            if contain_eos_token[i]:
                try:
                    query_str = tokenizer.decode(query_responses[i][:context_length], skip_special_tokens=True)
                    response_str = tokenizer.decode(query_responses[i][context_length:], skip_special_tokens=True)
                    if len(parse_sentence(response_str)['input_ids']) == 1:
                        print(f"Sample [{i}] cannot be parsed. Skip SHAP calculation.")
                        continue
                    
                    shap_outputs = get_shap_rewards(
                        model,
                        query_str,
                        response_str,
                        tokenizer,
                        masker=shap.maskers.Text(parse_sentence, mask_token=" ", collapse_mask_token=True)
                    )
                    shap_values, shap_data = np.squeeze(shap_outputs.values), shap_outputs.data
                    assert "".join(shap_data[0]) == response_str, f"{''.join(shap_data[0])} \n {response_str}"
                    for n in range(1, len(shap_values) + 1):
                        sents = "".join(shap_data[0][:n])
                        idx = len(tokenizer.encode(sents)) - 1
        
                        if idx > len(dense_rewards[i]) - 1:
                            print("idx: {}; response length: {}".format(idx, len(dense_rewards)))
                            dense_rewards[i, -1] = shap_values[n-1]
                        else:
                            dense_rewards[i, idx] = shap_values[n-1]
                except Exception as e:
                    print("SHAP Exception: ", e)
            else:
                print(f"Sample [{i}] has no <eos> token. Skip SHAP calculation.")
        dense_rewards = torch.tensor(dense_rewards).to(reward_logits)
        return (
            dense_rewards,
            seq_rewards,
            sequence_lengths,
        )
    elif reward_type == "attr_lig":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def evaluate(reward_model, policy, tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader, disable=sampling):
            queries = data["query_token"]
            reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((data["query_token"], data["reference_response_token"]), dim=1)
            _, reference_score, _ = get_reward(reward_model, query_reference_responses, tokenizer, queries.shape[1])

            query_responses, _ = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            _, score, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer, queries.shape[1])

            eval_storage["query_token"].extend(queries)
            eval_storage["reference_response_token"].extend(reference_response_token)
            eval_storage["reference_score"].append(reference_score)
            eval_storage["postprocessed_response_token"].extend(postprocessed_responses)
            eval_storage["score"].append(score)
            if sampling:
                break

    eval_storage["query"] = tokenizer.batch_decode(eval_storage["query_token"], skip_special_tokens=True)
    eval_storage["reference_response"] = tokenizer.batch_decode(eval_storage["reference_response_token"])
    eval_storage["postprocessed_response"] = tokenizer.batch_decode(
        eval_storage["postprocessed_response_token"], skip_special_tokens=True
    )
    eval_score = torch.cat(eval_storage["score"]).float().cpu().numpy().tolist()
    eval_reference_score = torch.cat(eval_storage["reference_score"]).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage["query"]),
            "postprocessed_response": gather_object(eval_storage["postprocessed_response"]),
            "reference_responses": gather_object(eval_storage["reference_response"]),
            "scores": gather_object(eval_score),
            "reference_scores": gather_object(eval_reference_score),
        }
    )
    return eval_storage, eval_df


def save_checkpoint(
    checkpoint_dir: str,
    accelerator: Accelerator,
    unwrapped_policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    args_for_hub_push: Args, # Pass the main Args object
    is_final_save: bool = False,
):
    """Saves the model and tokenizer to the specified directory."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        if is_final_save and args_for_hub_push.push_to_hub:
            tokenizer.push_to_hub(repo_id=args_for_hub_push.hf_repo_id, revision=args_for_hub_push.hf_repo_revision)
            accelerator.print(f"Tokenizer pushed to hub: {args_for_hub_push.hf_repo_id}/{args_for_hub_push.hf_repo_revision}")

    accelerator.wait_for_everyone() # Wait for all processes to reach this point

    if accelerator.is_main_process:
        unwrapped_policy.save_pretrained(
            checkpoint_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(unwrapped_policy),
            safe_serialization=False, # Set to True if preferred and compatible
        )
        if is_final_save and args_for_hub_push.push_to_hub:
            unwrapped_policy.push_to_hub(
                repo_id=args_for_hub_push.hf_repo_id,
                revision=args_for_hub_push.hf_repo_revision,
                safe_serialization=False # Set to True if preferred
            )
            accelerator.print(f"Model pushed to hub: {args_for_hub_push.hf_repo_url}")

    accelerator.wait_for_everyone() # Ensure saving is complete on main process before others continue


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    dataset = load_dataset(args.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True, drop_last=True)
    eval_dataloaders = {}
    for split in ["validation", "test"]:
        eval_dataset = load_dataset(args.query_dataset, split=split)
        eval_dataset = eval_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
        eval_dataloaders[split] = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=False,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model_config = AutoConfig.from_pretrained(args.base_model)
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if not args.reward_model_path:
        critic: PreTrainedModel = ScalarModel(scalar_model_config)
        reward_model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        critic: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
        reward_model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    for module in [policy, ref_policy, critic, reward_model]:
        disable_dropout(module)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    model = PolicyAndValueWrapper(policy, critic)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if args.offload or args.base_model == "EleutherAI/pythia-6.9b-deduped":
            deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
            eval_ds_config["zero_optimization"] = {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {"device": "cpu"},
            }
        accelerator.print(f"{eval_ds_config=}")
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        reward_model.eval()
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        ref_policy.eval()
    else:
        ref_policy = ref_policy.to(device)
        reward_model = reward_model.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()
    eval_split = list(eval_dataloaders.keys())[0]
    stats_shape = (args.ppo.noptepochs, args.nminibatches, args.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    model.train()
    for update in range(1, args.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - (update - 1.0) / args.num_updates
        lrnow = frac * args.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            eval_storage, eval_df = evaluate(
                reward_model,
                accelerator.unwrap_model(model).policy,
                tokenizer,
                eval_dataloaders[eval_split],
                validation_generation_config,
                sampling=True,
            )
            validation_score = eval_storage["score"][0]
            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                if accelerator.is_main_process:
                    eval_ds = Dataset.from_pandas(eval_df)
                    eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset_{global_step}")
                    if args.track:
                        wandb.log({f"samples/{eval_split}_query_responses": wandb.Table(dataframe=eval_df)}, step=update)
            del eval_storage, eval_df
            torch.cuda.empty_cache()

            queries = data["query_token"].to(device)
            context_length = queries.shape[1]
            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            values = []
            scores = []
            dense_scores = []
            sequence_lengths = []
            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]
                # query: [local_rollout_forward_batch_size, max_query_len]
                # query_response: [local_rollout_forward_batch_size, max_query_len + response_len]
                query_response, logits = generate(
                    accelerator.unwrap_model(model).policy,
                    query,
                    tokenizer,
                    generation_config,
                )
                response = query_response[:, context_length:]

                # use the logits during generation directly, instead of using the following
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()
                # ref_logprob: [local_rollout_forward_batch_size, response_len]
                ref_output = forward(ref_policy, query_response, tokenizer)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                postprocessed_response = truncate_response(args, tokenizer, response)

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                # query_response: [local_rollout_forward_batch_size, max_query_len + response_len]
                full_value, _, _ = get_reward(
                    accelerator.unwrap_model(model).critic, query_response, tokenizer, context_length
                )
                value = full_value[:, context_length - 1 : -1].squeeze(-1)
                # score: [local_rollout_forward_batch_size]
                # _, score, _ = get_reward(reward_model, postprocessed_query_response, tokenizer, context_length)
                dense_score, score, _ = get_reward(
                    reward_model,
                    postprocessed_query_response,
                    tokenizer,
                    context_length,
                    reward_type=args.reward_type,
                )

                query_responses.append(query_response)
                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                values.append(value)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                dense_scores.append(dense_score)
            query_responses = torch.cat(query_responses, 0)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            values = torch.cat(values, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)  # [batch_size]
            dense_scores = torch.cat(dense_scores, 0)  # [batch_size, response_max_length]
            del (logprob, ref_logprob, full_value, value, score)
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
            accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            sequence_lengths_p1 = sequence_lengths + 1
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask_p1, 0)

            # 4. compute rewards
            kl = logprobs - ref_logprobs  # [batch_size, response_length]
            non_score_reward = -args.ppo.kl_coef * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            # sequence_lengths: index of the first <EOS> token
            # sequence_lengths_p1: index of the first <PAD> token
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            if args.reward_type != "sparse":
                dense_scores = torch.masked_fill(dense_scores, padding_mask_p1, 0)
                rewards[[actual_start, actual_end]] += args.sparse_weight * scores
                rewards += args.dense_weight * dense_scores
            else:
                rewards[[actual_start, actual_end]] += args.sparse_weight * scores

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()
            accelerator.print("rewards====", rewards[0])
            accelerator.print("advantages====", advantages[0])
            accelerator.print("values====", values[0])
            torch.cuda.empty_cache()

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_return = returns[micro_batch_inds]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds])
                        loss = pg_loss + args.ppo.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # del everything and empty cache
                # fmt: off
                del (
                    output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                    vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                torch.cuda.empty_cache()
            if accelerator.is_main_process:
                console.print(
                    "ppo_epoch_idx",
                    ppo_epoch_idx,
                    "approxkl",
                    approxkl_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_loss",
                    pg_loss_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_clipfrac",
                    pg_clipfrac_stats[: ppo_epoch_idx + 1].mean().item(),
                    "ratio",
                    ratio_stats[: ppo_epoch_idx + 1].mean().item(),
                )

        # Save checkpoint during training
        if args.output_dir and args.save_frequency > 0 and update % args.save_frequency == 0:
            checkpoint_save_dir = f"{args.output_dir}/checkpoint_{update}"
            accelerator.print(f"Saving intermediate checkpoint to {checkpoint_save_dir} at update {update}")
            policy_to_save = accelerator.unwrap_model(model).policy
            save_checkpoint(
                checkpoint_dir=checkpoint_save_dir,
                accelerator=accelerator,
                unwrapped_policy=policy_to_save,
                tokenizer=tokenizer,
                args_for_hub_push=args, # Pass the main args
                is_final_save=False, # This is an intermediate save
            )
            accelerator.print(f"Intermediate checkpoint saved to {checkpoint_save_dir}")

        with torch.no_grad():
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(pg_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", lrnow, update)
            writer.add_scalar("ppo/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("ppo/eps", eps, update)
            accelerator.print("ppo/eps", eps, update)
        del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
        torch.cuda.empty_cache()

    if args.run_eval:
        for eval_split in eval_dataloaders:
            eval_storage, eval_df = evaluate(
                reward_model,
                accelerator.unwrap_model(model).policy,
                tokenizer,
                eval_dataloaders[eval_split],
                validation_generation_config,
                sampling=False,
            )
            if accelerator.is_main_process:
                eval_ds = Dataset.from_pandas(eval_df)
                eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset")
                if args.track:
                    wandb.log({f"eval/{eval_split}_query_responses": wandb.Table(dataframe=eval_df)})

    # save final model
    if args.output_dir:
        accelerator.print(f"Saving final model to {args.output_dir}")
        policy_to_save = accelerator.unwrap_model(model).policy
        save_checkpoint(
            checkpoint_dir=args.output_dir, # Final save goes to the main output_dir
            accelerator=accelerator,
            unwrapped_policy=policy_to_save,
            tokenizer=tokenizer,
            args_for_hub_push=args, # Pass the main args
            is_final_save=True,
        )
        accelerator.print(f"Final model saved to {args.output_dir}")
        if args.push_to_hub and accelerator.is_main_process: # Message already printed in save_checkpoint
             accelerator.print(f"🚀 Final model and tokenizer (if configured) have been pushed to the Hugging Face Hub.")
