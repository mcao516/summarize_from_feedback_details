import re
import shap
import torch


def parse_sentence(paragraph, return_offsets_mapping=True):
    """
    Parse a paragraph into spans based on delimiters and return offset mappings.
    
    Args:
        paragraph (str): The input English paragraph.
    
    Returns:
        spans (list): A list of spans split by the delimiters.
        offset_mapping (list): A list of tuples indicating the start and end position of each span.
    """
    # Regex pattern for the delimiters
    pattern = r"[.,;?!]"
    
    spans = []
    offset_mapping = []
    start = 0
    
    for match in re.finditer(pattern, paragraph):
        end = match.end()
        span = paragraph[start:end].strip()
        if span:  # Only add non-empty spans
            spans.append(span)
            offset_mapping.append((start, end))
        start = end
    
    # Add the last span if there's any text left after the final delimiter
    if start < len(paragraph):
        spans.append(paragraph[start:].strip())
        offset_mapping.append((start, len(paragraph)))
    
    if return_offsets_mapping:
        return {'input_ids': spans, 'offset_mapping': offset_mapping}
    else:
        return {'input_ids': spans}


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_shap_rewards(model, query_response, tokenizer, context_length, masker=None):
    """
    Args:
        query_response: [query_responses_len]
    """
    query_text_clean = tokenizer.decode(query_response[:context_length], skip_special_tokens=True)
    response_text_clean = tokenizer.decode(query_response[context_length:], skip_special_tokens=True)

    # if torch.any(query_response[context_length:] == tokenizer.eos_token_id, dim=-1):
    #     response_text_clean += '<|endoftext|>'

    def f(x):
        inputs = []
        for _x in x:
            if len(_x) > 0 and _x[0] == " ":
                concatenated = query_text_clean + _x + "<|endoftext|>"
            else:
                concatenated = query_text_clean + " " + _x + "<|endoftext|>"
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

    masker = tokenizer if not masker else masker
    explainer = shap.Explainer(f, masker, algorithm="auto")

    shap_values = explainer([response_text_clean]) 

    return shap_values