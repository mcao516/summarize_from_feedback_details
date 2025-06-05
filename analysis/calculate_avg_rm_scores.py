import json
import argparse
import math
from typing import Tuple, Optional


def calculate_score_statistics(file_path: str, field_name: str = "score") -> Tuple[Optional[float], Optional[float]]:
    """
    Reads a JSON file, calculates the mean and standard deviation of numerical values
    from the specified field.
    Assumes the file contains a list of dictionaries.
    Returns (mean, std_dev) or (None, None) if critical errors occur (e.g., file not found, JSON error).
    Returns (0.0, 0.0) if no valid scores are found in the data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        return None, None

    if not isinstance(data, list):
        print(f"Error: JSON content in '{file_path}' is not a list of items.")
        return None, None
    
    print(f"Total items in the list from '{file_path}': {len(data)}")

    scores = []
    for item in data:
        if isinstance(item, dict) and field_name in item:
            try:
                scores.append(float(item[field_name]))
            except (ValueError, TypeError):
                print(f"Warning: Skipping invalid '{field_name}' value '{item.get(field_name)}' in item: {item}")
        else:
            print(f"Warning: Skipping item because it's not a dictionary or lacks the '{field_name}' field: {item}")

    print(f"Number of valid '{field_name}' values found for calculation: {len(scores)}")

    if not scores:
        print(f"No valid '{field_name}' values found to calculate statistics.")
        return 0.0, 0.0 # Mean and StdDev are 0 if no data

    mean = sum(scores) / len(scores)
    
    if len(scores) < 2:
        # Standard deviation is 0 if there's only one data point (or zero, covered above)
        std_dev = 0.0
    else:
        # Calculate population standard deviation
        sum_of_squared_differences = sum([(s - mean) ** 2 for s in scores])
        variance = sum_of_squared_differences / len(scores)
        std_dev = math.sqrt(variance)

    return mean, std_dev


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation of scores from a JSON file.")
    parser.add_argument("file_to_process", type=str, help="Path to the JSON file to process.")
    parser.add_argument(
        "--field_name", 
        type=str, 
        default="score", 
        help="Name of the field in JSON objects containing the numerical value (default: 'score')."
    )
    args = parser.parse_args()

    mean_score, std_dev_score = calculate_score_statistics(args.file_to_process, args.field_name)

    if mean_score is not None and std_dev_score is not None:
        print(f"\nStatistics for field '{args.field_name}' from file '{args.file_to_process}':")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Standard Deviation: {std_dev_score:.4f}")
    else:
        print(f"\nCould not calculate statistics for '{args.file_to_process}'. Please see error messages above.")