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


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    # return torch.tensor([logit[1] - logit[0] for logit in out.logits], device=query_responses.device)
    return out.logits[:, 1] - out.logits[:, 0]

def get_attr_rewards(model, query_response, tokenizer, context_length, n_steps=300, internal_batch_size=50):
    """
    Args:
        query_response: [query_responses_len]

    Returns:
        attributions: 
    """
    doc_ids, summary_ids = query_response[:context_length], query_response[context_length:]
    doc_ids, summary_ids = doc_ids.unsqueeze(0), summary_ids.unsqueeze(0)

    def f(input_ids, document_ids, dim=1):
        ids = torch.cat((document_ids, input_ids), dim)
        score = get_reward(model, ids, tokenizer, doc_ids.shape[1])
        return score
    
    attr_method = LayerIntegratedGradients(f, model.transformer.wte)

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