import collections
import os
import random
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Literal, Optional

import evaluate as hf_evaluate
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi

api = HfApi()
rouge = hf_evaluate.load("rouge")


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
    gradient_accumulation_steps: int = 16
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 4
    """per rank eval batch size"""

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
    temperature: float = 0.01
    """the sampling temperature"""

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
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given for the final model)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset for the final model)"""
    output_dir: str = "models/sft_model"
    """Where to save the model and checkpoints"""
    save_frequency_updates: int = 0
    """Frequency of saving checkpoints in terms of optimizer updates. After N optimizer updates, a checkpoint is saved. 0 or negative to disable intermediate saving."""


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
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
        if args.hf_repo_revision is None:  # auto-generate one for the final model, usually based on run_name
            args.hf_repo_revision = args.run_name
        # This URL is for the final model with its specific base revision (e.g., run_name)
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


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
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


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
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )


def save_model_checkpoint(
    args: Args,
    accelerator: Accelerator,
    tokenizer,
    model_prepared: PreTrainedModel, # This is the accelerator.prepared model
    optimizer_step: int,
    is_final_save: bool = False,
):
    if not args.output_dir:
        accelerator.print("No output_dir specified, skipping save.")
        return

    # Determine save directory and HF revision
    save_directory = args.output_dir
    # `args.hf_repo_revision` is the one set in parse_args, typically based on run_name for the final model
    # or a user-defined base revision.
    current_hf_revision = args.hf_repo_revision 

    if not is_final_save:
        checkpoint_name = f"checkpoint-{optimizer_step}"
        save_directory = os.path.join(args.output_dir, checkpoint_name)
        if args.hf_repo_revision: # If a base revision is set (e.g. run_name or user-defined for the run)
            current_hf_revision = f"{args.hf_repo_revision}-{checkpoint_name}"
        # If args.hf_repo_revision was None (e.g. push_to_hub is false, or it wasn't set),
        # current_hf_revision remains None for checkpoints, preventing push attempts later if it's None.
    
    accelerator.print(f"=== Saving {'final model' if is_final_save else f'checkpoint at optimizer step {optimizer_step}'} to {save_directory} ===")
    os.makedirs(save_directory, exist_ok=True)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_directory)
        if args.push_to_hub and args.hf_repo_id and current_hf_revision:
            tokenizer_hub_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{current_hf_revision}"
            accelerator.print(f"Pushing tokenizer to HF Hub: {args.hf_repo_id}, revision: {current_hf_revision}")
            accelerator.print(f"Tokenizer will be available at: {tokenizer_hub_url}")
            try:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=current_hf_revision)
            except Exception as e:
                accelerator.print(f"Failed to push tokenizer: {e}")
        elif args.push_to_hub and (not args.hf_repo_id or not current_hf_revision):
            accelerator.print(f"Skipping tokenizer push to hub: hf_repo_id ('{args.hf_repo_id}') or current_hf_revision ('{current_hf_revision}') is not configured correctly.")

    unwrapped_model: PreTrainedModel = accelerator.unwrap_model(model_prepared)
    accelerator.wait_for_everyone() # Ensure all processes are ready before saving, especially for sharded models

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            save_directory,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model_prepared), # Get state_dict from prepared model
            safe_serialization=False,
        )
        if args.push_to_hub and args.hf_repo_id and current_hf_revision:
            model_hub_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{current_hf_revision}"
            accelerator.print(f"Pushing model to HF Hub: {args.hf_repo_id}, revision: {current_hf_revision}")
            accelerator.print(f"Model will be available at: {model_hub_url}")
            try:
                unwrapped_model.push_to_hub(repo_id=args.hf_repo_id, revision=current_hf_revision, safe_serialization=False)
                accelerator.print(f"🔥 Successfully pushed to {model_hub_url}")
                # args.hf_repo_url is specifically for the final model's main revision (e.g., run_name)
                if is_final_save and args.hf_repo_url and current_hf_revision == args.hf_repo_revision:
                     accelerator.print(f"This is the final model. Main repo URL: {args.hf_repo_url}")
            except Exception as e:
                accelerator.print(f"Failed to push model: {e}")
        elif args.push_to_hub and (not args.hf_repo_id or not current_hf_revision):
            accelerator.print(f"Skipping model push to hub: hf_repo_id ('{args.hf_repo_id}') or current_hf_revision ('{current_hf_revision}') is not configured correctly.")


def evaluate(args: Args, accelerator, tokenizer, model, dataloader, generation_config):
    model.eval()
    rouge_scores = collections.defaultdict(list)
    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    all_losses = []
    unwrapped = accelerator.unwrap_model(model)
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"]
            reference_responses = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((queries, reference_responses), dim=1)
            output = forward(model, query_reference_responses, tokenizer)
            labels = query_reference_responses.masked_fill(query_reference_responses == tokenizer.pad_token_id, -1)
            lm_logits = output.logits
            # hand-rolled transformer loss: Shift so that tokens < n predict n
            # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1,
            )
            loss = accelerator.gather(loss)
            all_losses.append(loss)

            generated_responses = generate(
                unwrapped,
                queries,
                tokenizer,
                generation_config,
            )
            responses = generated_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            decode_queries = tokenizer.batch_decode(queries)
            decode_reference_responses = tokenizer.batch_decode(
                reference_responses,
                skip_special_tokens=True,
            )
            decode_responses = tokenizer.batch_decode(
                postprocessed_responses,
                skip_special_tokens=True,
            )
            rouge_score = rouge.compute(predictions=decode_responses, references=decode_reference_responses)
            decode_queries = gather_object(decode_queries)
            decode_responses = gather_object(decode_responses)
            decode_reference_responses = gather_object(decode_reference_responses)
            rouge_scores["rouge1"].append(np.mean(gather_object([rouge_score["rouge1"]])))
            rouge_scores["rouge2"].append(np.mean(gather_object([rouge_score["rouge2"]])))
            rouge_scores["rougeL"].append(np.mean(gather_object([rouge_score["rougeL"]])))
            all_decode_queries.extend(decode_queries)
            all_decode_responses.extend(decode_responses)
            all_decode_reference_responses.extend(decode_reference_responses)
    return (
        pd.DataFrame(
            {
                "query": all_decode_queries,
                "response": all_decode_responses,
                "reference": all_decode_reference_responses,
            }
        ),
        rouge_scores,
        all_losses,
    )


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    dataset = load_dataset(args.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size, shuffle=True)
    eval_dataloaders = {}
    for split in ["validation", "test"]:
        eval_dataset = load_dataset(args.query_dataset, split=split)
        eval_dataset = eval_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
        eval_dataloaders[split] = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)
    args.total_episodes = len(dataset)
    # If args.num_updates is not given, it's calculated based on total_episodes and batch_size
    # This represents the number of optimizer steps per epoch if training on the full dataset.
    if args.num_updates is None:
        args.num_updates = args.total_episodes // args.batch_size

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
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
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
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=model_config,
        trust_remote_code=True,
    )
    disable_dropout(model)
    model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    
    # Total number of optimizer steps for the scheduler
    # args.num_updates is number of optimizer steps per epoch if not overridden.
    # If args.num_updates was set by user, it's likely the total number of optimizer steps for the whole training.
    # The original code uses `args.num_updates * args.num_train_epochs`.
    # Let's clarify: If user sets args.num_updates, it might be total updates.
    # If args.num_updates is calculated, it's per epoch.
    # The current scheduler line `num_training_steps=args.num_updates * args.num_train_epochs` is correct if
    # args.num_updates is per-epoch updates. If user specifies it as total, then num_train_epochs should be 1 or this logic adjusted.
    # For this change, we keep it as is.
    num_total_scheduler_steps = args.num_updates * args.num_train_epochs

    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=num_total_scheduler_steps,
    )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    # Accelerator prepares model, optimizer, dataloader(s), and scheduler
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)

    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training model===")
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0 # Original global_step: micro-batch counter across all processes for one epoch. Re-set per epoch in original? No, seems cumulative.
    update = 0 # Original update: micro-batch counter for this rank, cumulative across epochs. Used for logging.
    
    optimizer_steps_completed = 0 # Counter for actual optimizer steps

    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            update += 1 # This is the micro-batch step for the current rank, used in original logging
            global_step += args.micro_batch_size # Total micro-batches processed across all ranks
            
            query_responses = data["query_reference_response_token"]
            with accelerator.accumulate(model):
                output = forward(model, query_responses, tokenizer)
                # mask out gradient effects on response padding tokens
                labels = query_responses.masked_fill(query_responses == tokenizer.pad_token_id, -1)
                lm_logits = output.logits
                # hand-rolled transformer loss: Shift so that tokens < n predict n
                # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Loss stats for current micro-batch, for logging the average accumulated loss
            loss_stats[gradient_accumulation_idx] = loss.detach() # Use .detach()
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            
            # This block is executed when an actual optimizer step has been performed
            # (i.e., after `args.gradient_accumulation_steps` micro-batches have been processed by `accelerator.accumulate`)
            # The original condition `update > 1 and (update - 1) % args.gradient_accumulation_steps == 0`
            # is implicitly handled by `accelerator.accumulate` context manager for when `optimizer.step()` runs.
            # We need to know when an optimizer step actually happens.
            # `accelerator.sync_gradients` is True when gradients have been synced and optimizer can step.
            # This check happens *inside* the accumulate block usually.
            # A simpler way is to check if the micro-batch `update` counter means an accumulation cycle is complete.
            if update % args.gradient_accumulation_steps == 0 : # An optimizer step was just performed for this rank
                scheduler.step() # Original position of scheduler step
                
                # START: Added code for checkpointing
                optimizer_steps_completed += 1
                # END: Added code for checkpointing

                # Original logging logic (uses `update` as step, which is micro-batch count for this rank)
                writer.add_scalar("train/sft/loss", accelerator.gather(loss_stats).mean().item(), update)
                writer.add_scalar("train/sft/lr", scheduler.get_last_lr()[0], update)
                if update % (args.gradient_accumulation_steps * 10) == 0: # Print less frequently
                    accelerator.print(
                        f"Epoch: {epoch}, Update (micro-batch): {update}, Opt.Step: {optimizer_steps_completed}, "
                        f"Avg Acc Loss: {accelerator.gather(loss_stats).mean().item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                
                # START: Added code for checkpointing
                if args.save_frequency_updates > 0 and \
                   optimizer_steps_completed > 0 and \
                   optimizer_steps_completed % args.save_frequency_updates == 0:
                    save_model_checkpoint(
                        args, accelerator, tokenizer, model, # model is the prepared model
                        optimizer_step=optimizer_steps_completed,
                        is_final_save=False
                    )
                # END: Added code for checkpointing
            
            # Optional: Early exit if total number of optimizer steps is reached
            if num_total_scheduler_steps > 0 and optimizer_steps_completed >= num_total_scheduler_steps:
                accelerator.print(f"Reached target number of optimizer steps ({num_total_scheduler_steps}). Stopping training.")
                break # break from inner dataloader loop
        
        if num_total_scheduler_steps > 0 and optimizer_steps_completed >= num_total_scheduler_steps:
            break # break from outer epoch loop

    # save final model
    if args.output_dir and args.num_train_epochs > 0 and optimizer_steps_completed > 0:
        accelerator.print("===saving final model===")
        save_model_checkpoint(
            args,
            accelerator,
            tokenizer,
            model,
            optimizer_step=optimizer_steps_completed, # Pass the total optimizer steps
            is_final_save=True,
        )
    elif args.output_dir:
        accelerator.print("Skipping final model save: No training epochs completed or no optimizer steps performed.")

    if args.run_eval:
        accelerator.print("===evaluating model===")
        for eval_split in eval_dataloaders:
            eval_df, rouge_scores, all_eval_losses = evaluate(
                args, accelerator, tokenizer, model, eval_dataloaders[eval_split], generation_config
            )
            if accelerator.is_main_process:
                # Original eval table naming (no step number, assumes one final eval)
                eval_df_path = os.path.join(f"runs/{args.run_name}", f"{eval_split}_table.csv")
                os.makedirs(os.path.dirname(eval_df_path), exist_ok=True)
                eval_df.to_csv(eval_df_path)

                if args.track:
                    # Original WandB logging for eval table (uses `update` which is total micro-batches on rank 0)
                    wandb.log({f"eval/{eval_split}_query_responses": wandb.Table(dataframe=eval_df)}, step=update)
            for k, v in rouge_scores.items():
                rouge_metric = torch.tensor(v, device=device)
                # Gather the list of means, then compute mean of means
                gathered_means = accelerator.gather(rouge_metric)
                final_rouge_mean = gathered_means.mean().item()
                # Original writer logging for eval (uses `update`)
                writer.add_scalar(f"{eval_split}/sft/rouge/{k}", final_rouge_mean, update)
                accelerator.print(f"{eval_split}/sft/rouge/{k}: {final_rouge_mean} (logged at micro-batch step {update})")
            # Original writer logging for eval loss (uses `update`)
            writer.add_scalar(f"{eval_split}/sft/loss", torch.stack(all_eval_losses).mean().item(), update)
