import torch
from captum.attr import (
    LayerGradientShap,
    LayerIntegratedGradients,
    LayerDeepLift,
    LayerDeepLiftShap,
    TokenReferenceBase,
    LayerFeatureAblation,
    LayerLRP,
)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    # reward_logits: [batch_size, query_responses_len, 1]
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )


def get_attr_rewards(model, query_response, tokenizer, context_length, n_steps=300, internal_batch_size=50):
    """
    Args:
        query_response: [query_responses_len]

    Returns:
        attributions: 
    """
    doc_ids, summary_ids = query_response[:context_length], query_response[context_length:]
    doc_ids, summary_ids = doc_ids.unsqueeze(0), summary_ids.unsqueeze(0)

    # def f(input_ids):
    #     if doc_ids.shape[0] == input_ids.shape[0]:
    #         ids = torch.cat((doc_ids, input_ids), 1)
    #     else:
    #         doc_ids_expanded = doc_ids.repeat(input_ids.shape[0], 1)
    #         ids = torch.cat((doc_ids_expanded, input_ids), 1)
    #     _, score, _ = get_reward(reward_model, ids, tokenizer, doc_ids.shape[1])
    #     return score

    def f(input_ids, document_ids, dim=1):
        ids = torch.cat((document_ids, input_ids), dim)
        _, score, _ = get_reward(model, ids, tokenizer, doc_ids.shape[1])
        return score
    
    attr_method = LayerIntegratedGradients(f, model.lm_backbone.embed_in)

    model.zero_grad() 

    # generate reference ids
    seq_len = summary_ids.shape[1]
    token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
    reference_indices = token_reference.generate_reference(seq_len, device=summary_ids.device).unsqueeze(0)
    reference_indices = reference_indices.repeat(summary_ids.shape[0], 1)

    attributions, delta = attr_method.attribute(
        summary_ids,
        reference_indices,
        additional_forward_args=(doc_ids, 1),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    attributions = attributions[:, context_length:, :].detach()

    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions