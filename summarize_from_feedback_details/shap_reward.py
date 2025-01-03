import re
import shap
import torch


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_shap_reward(model, query_response, tokenizer, context_length):
    """
    Args:
        query_response: [query_responses_len]
    """
    query_text_clean = tokenizer.decode(query_response[:context_length], skip_special_tokens=True)
    response_text_clean = tokenizer.decode(query_response[context_length:], skip_special_tokens=True)

    if torch.any(query_response[context_length:] == tokenizer.eos_token_id, dim=-1):
        response_text_clean += '<|endoftext|>'

    def f(x):
        inputs = []
        for _x in x:
            concatenated = query_text_clean + " " + _x
            inputs.append(concatenated)
    
        with torch.no_grad():
            input_ids = tokenizer(inputs, padding="longest", return_tensors="pt")["input_ids"].to("cuda")
            attention_mask = input_ids != tokenizer.pad_token_id
            reward_logits = model(
                input_ids=torch.masked_fill(input_ids, ~attention_mask, 0),
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            sequence_lengths = first_true_indices(input_ids == tokenizer.pad_token_id) - 1
            output = reward_logits[torch.arange(len(inputs), device=reward_logits.device), sequence_lengths].squeeze(-1)

        return output.detach().cpu().float().numpy()

    masker = tokenizer
    explainer = shap.Explainer(f, masker, algorithm="auto")

    shap_values = explainer([response_text_clean]) 

    return shap_values