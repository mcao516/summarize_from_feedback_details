import re


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
        span = paragraph[start:end]
        if span:  # Only add non-empty spans
            spans.append(span)
            offset_mapping.append((start, end))
        start = end
    
    # Add the last span if there's any text left after the final delimiter
    if start < len(paragraph):
        spans.append(paragraph[start:])
        offset_mapping.append((start, len(paragraph)))
    
    if return_offsets_mapping:
        return {'input_ids': spans, 'offset_mapping': offset_mapping}
    else:
        return {'input_ids': spans}