import argparse
import json
import random
import os
import time
from google import genai
from google.genai import types
from openai import OpenAI
import concurrent.futures
import threading


def load_json_dataset(file_path):
    """Loads a JSON dataset from the given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of objects.")
        for item in data:
            if not isinstance(item, dict) or "document" not in item or "summary" not in item:
                raise ValueError("Each item in the JSON list must be a dictionary with 'document' and 'summary' keys.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except ValueError as ve:
        print(f"Error: Data format incorrect in {file_path}: {ve}")
        return None


def call_gemini_api(prompt_text, gemini_client, model_name, retries, delay, temperature, max_output_tokens):
    """Calls the Gemini API with retry logic."""
    if not gemini_client:
        print("Error: Gemini client not initialized. Check API Key or initialization.")
        return None
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            print(f"Gemini API call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached for Gemini. Skipping this item.")
                return None


def call_openai_api(prompt_text, openai_client, model_name, retries, delay, temperature, max_output_tokens):
    """Calls the OpenAI API with retry logic."""
    if not openai_client:
        print("Error: OpenAI client not initialized. Check API Key or initialization.")
        return None
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of text summaries."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=max_output_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached for OpenAI. Skipping this item.")
                return None


def extract_decision(llm_response_text):
    """Extracts the A, B, or C decision from the LLM's response."""
    if not llm_response_text:
        return None
    try:
        lines = [line.strip() for line in llm_response_text.strip().split('\n') if line.strip()]
        if not lines:
            print(f"Warning: LLM response was empty or only whitespace.")
            return None

        last_line_decision = lines[-1].upper()
        if last_line_decision in ['A', 'B', 'C']:
            return last_line_decision

        if last_line_decision.endswith(('A', 'B', 'C')):
            char = last_line_decision[-1]
            if char in ['A', 'B', 'C']:
                 return char

        print(f"Warning: Could not parse decision ('A', 'B', 'C') from last line: '{last_line_decision}'. Trying fallback.")
        for line in reversed(lines):
            cleaned_line_upper = line.upper()
            if cleaned_line_upper.startswith("DECISION: A") or cleaned_line_upper == "A": return 'A'
            if cleaned_line_upper.startswith("DECISION: B") or cleaned_line_upper == "B": return 'B'
            if cleaned_line_upper.startswith("DECISION: C") or cleaned_line_upper == "C": return 'C'

            if line.startswith("**Overall Decision:**"): # Matches the prompt example
                decision_part = cleaned_line_upper.replace("**OVERALL DECISION:**", "").strip()
                if decision_part == 'A': return 'A'
                if decision_part == 'B': return 'B'
                if decision_part == 'C': return 'C'

                if "SUMMARY A IS BETTER" in cleaned_line_upper or "A IS BETTER" in cleaned_line_upper: return 'A'
                if "SUMMARY B IS BETTER" in cleaned_line_upper or "B IS BETTER" in cleaned_line_upper: return 'B'
                if "SIMILAR QUALITY" in cleaned_line_upper or "TIE" in cleaned_line_upper or "EQUALLY GOOD" in cleaned_line_upper: return 'C'
                break

        print(f"Warning: Could not reliably extract decision from response:\n###{llm_response_text}###")
        return None
    except Exception as e:
        print(f"Error extracting decision: {e}\nFrom response:\n{llm_response_text[:500]}...")
        return None


def _process_item_task(item_index, item_m1, item_m2, prompt_template,
                       model1_name_param, model2_name_param, # These come from args.model1_name, args.model2_name
                       shared_results_dict, shared_detailed_results_list, results_lock,
                       gemini_client_instance, openai_client_instance,
                       args_config):
    """Processes a single item evaluation. Designed to be run in a thread."""
    # Note: total_items_count removed as it was only for a print statement easily derived outside
    print(f"--- Evaluating item index {item_index} (Processing order may vary) ---")

    original_doc = item_m1['document']
    summary_m1_text = item_m1['summary']
    summary_m2_text = item_m2['summary']

    assigned_summary_a_text = ""
    assigned_summary_b_text = ""
    summary_a_origin_model = ""
    summary_b_origin_model = ""

    is_m1_a = random.choice([True, False])
    if is_m1_a:
        assigned_summary_a_text = summary_m1_text
        assigned_summary_b_text = summary_m2_text
        summary_a_origin_model = model1_name_param
        summary_b_origin_model = model2_name_param
    else:
        assigned_summary_a_text = summary_m2_text
        assigned_summary_b_text = summary_m1_text
        summary_a_origin_model = model2_name_param
        summary_b_origin_model = model1_name_param

    current_prompt = prompt_template.format(
        document_text=original_doc,
        summary_a_text=assigned_summary_a_text,
        summary_b_text=assigned_summary_b_text
    )

    llm_response = None
    if args_config.evaluator_model_type == "GEMINI":
        evaluator_specific_model_name = args_config.gemini_model_name
        llm_response = call_gemini_api(
            current_prompt,
            gemini_client_instance,
            args_config.gemini_model_name,
            args_config.api_retries,
            args_config.api_delay_seconds,
            args_config.api_temperature,
            args_config.api_max_output_tokens
        )
    elif args_config.evaluator_model_type == "OPENAI":
        evaluator_specific_model_name = args_config.openai_model_name
        llm_response = call_openai_api(
            current_prompt,
            openai_client_instance,
            args_config.openai_model_name,
            args_config.api_retries,
            args_config.api_delay_seconds,
            args_config.api_temperature,
            args_config.api_max_output_tokens
        )
    else:
        print(f"Error: Unknown EVALUATOR_MODEL_TYPE: {args_config.evaluator_model_type}")
        with results_lock:
            shared_results_dict["Errors/NoDecision"] += 1
        return

    current_result_detail = {
        "item_index": item_index,
        "document": original_doc,
        "summary_A_text": assigned_summary_a_text,
        "summary_A_origin": summary_a_origin_model,
        "summary_B_text": assigned_summary_b_text,
        "summary_B_origin": summary_b_origin_model,
        "llm_evaluator": args_config.evaluator_model_type,
        "llm_evaluator_model": evaluator_specific_model_name,
        "llm_response_raw": llm_response,
        "decision_extracted": None,
        "winner": None
    }

    if not llm_response:
        with results_lock:
            shared_results_dict["Errors/NoDecision"] += 1
        print(f"Skipping item index {item_index} due to API error or no response.")
        current_result_detail["winner"] = "Error/NoDecision_API"
    else:
        decision = extract_decision(llm_response)
        current_result_detail["decision_extracted"] = decision

        with results_lock:
            if decision == 'A':
                winner = summary_a_origin_model
                shared_results_dict[winner] += 1
                current_result_detail["winner"] = winner
            elif decision == 'B':
                winner = summary_b_origin_model
                shared_results_dict[winner] += 1
                current_result_detail["winner"] = winner
            elif decision == 'C':
                shared_results_dict["Tie/Inconclusive"] += 1
                current_result_detail["winner"] = "Tie/Inconclusive"
            else:
                shared_results_dict["Errors/NoDecision"] += 1
                current_result_detail["winner"] = "Error/NoDecision_Extraction"

    with results_lock:
        shared_detailed_results_list.append(current_result_detail)

    time.sleep(args_config.api_delay_seconds) # Small delay after processing each item, per thread


def run_evaluation(args, gemini_model_client, openai_client_instance):
    print("Loading datasets...")
    data_model1_full = load_json_dataset(args.model1_json_path)
    data_model2_full = load_json_dataset(args.model2_json_path)

    if not data_model1_full or not data_model2_full:
        print("Failed to load one or both datasets. Exiting.")
        return

    min_len = min(len(data_model1_full), len(data_model2_full))
    num_samples_to_run = min(min_len, args.num_eval_samples)

    if len(data_model1_full) != len(data_model2_full):
        print(f"Warning: Datasets have different lengths ({len(data_model1_full)} vs {len(data_model2_full)}). Using {num_samples_to_run} common samples.")

    data_model1 = data_model1_full[:num_samples_to_run]
    data_model2 = data_model2_full[:num_samples_to_run]

    if not data_model1:
        print("No samples to evaluate based on NUM_EVAL_SAMPLES or dataset lengths.")
        return

    print(f"Successfully loaded {len(data_model1)} items for comparison.")

    if not os.path.exists(args.prompt_template_path):
        print(f"Error: Prompt template file not found at {args.prompt_template_path}")
        return

    with open(args.prompt_template_path, "r", encoding='utf-8') as rf:
        prompt_template_content = rf.read()

    results_summary = {
        args.model1_name: 0,
        args.model2_name: 0,
        "Tie/Inconclusive": 0,
        "Errors/NoDecision": 0
    }
    detailed_results_list = []
    processing_lock = threading.Lock()

    total_items_to_process = len(data_model1)
    print(f"\nStarting evaluation of {total_items_to_process} items using up to {args.max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i in range(total_items_to_process):
            item_m1 = data_model1[i]
            item_m2 = data_model2[i]

            if item_m1.get('document') != item_m2.get('document'):
                # This was a pass before, making it a warning. Consider if this should be an error.
                print(f"Warning: Documents for item index {i} are different. Proceeding with m1's document as primary.")

            futures.append(executor.submit(
                _process_item_task,
                i,
                item_m1,
                item_m2,
                prompt_template_content,
                args.model1_name, # Pass model name from args
                args.model2_name, # Pass model name from args
                results_summary,
                detailed_results_list,
                processing_lock,
                gemini_model_client, # Pass initialized Gemini client
                openai_client_instance, # Pass initialized OpenAI client
                args  # Pass the full args object
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'An item processing generated an exception: {exc}')

    detailed_results_list.sort(key=lambda x: x["item_index"])

    print("\n--- Evaluation Complete ---")
    print("Final Results:")
    for model_name_key, count in results_summary.items():
        print(f"{model_name_key}: {count}")

    total_compared_for_wins = results_summary[args.model1_name] + results_summary[args.model2_name]
    if total_compared_for_wins > 0:
        print(f"\nWin percentage for {args.model1_name} (vs {args.model2_name}, excluding ties & errors): {results_summary[args.model1_name] / total_compared_for_wins * 100:.2f}%")
        print(f"Win percentage for {args.model2_name} (vs {args.model1_name}, excluding ties & errors): {results_summary[args.model2_name] / total_compared_for_wins * 100:.2f}%")
    else:
        print("\nNo conclusive wins to calculate percentages.")

    results_output_filename = f"evaluation_results_{args.evaluator_model_type.lower()}_{args.model1_name}_vs_{args.model2_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_output_filename, 'w', encoding='utf-8') as f_out:
            json.dump(detailed_results_list, f_out, indent=2)
        print(f"\nDetailed results saved to: {results_output_filename}")
    except IOError as e:
        print(f"Error saving detailed results to {results_output_filename}: {e}")

    summary_stats_filename = f"evaluation_summary_{args.evaluator_model_type.lower()}_{args.model1_name}_vs_{args.model2_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(summary_stats_filename, 'w', encoding='utf-8') as f_out:
            json.dump(results_summary, f_out, indent=2)
        print(f"Summary statistics saved to: {summary_stats_filename}")
    except IOError as e:
        print(f"Error saving summary statistics to {summary_stats_filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run side-by-side LLM evaluation of summaries.")
    
    # File and Model Name Arguments
    parser.add_argument("--model1_json_path", type=str, required=True, help="Path to the JSON file for Model 1's summaries.")
    parser.add_argument("--model2_json_path", type=str, required=True, help="Path to the JSON file for Model 2's summaries.")
    parser.add_argument("--model1_name", type=str, default="Model1", help="Name for Model 1.")
    parser.add_argument("--model2_name", type=str, default="Model2", help="Name for Model 2.")
    parser.add_argument("--prompt_template_path", type=str, default="prompt_template.txt", help="Path to the prompt template file.")

    # Evaluation Control Arguments
    parser.add_argument("--num_eval_samples", type=int, default=100, help="Number of samples to evaluate.")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of threads for parallel processing.")

    # API Key Arguments (default to environment variables)
    parser.add_argument("--google_api_key", type=str, default=os.getenv("GEMINI_API_KEY"), help="Google API Key. Defaults to GEMINI_API_KEY environment variable.")
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key. Defaults to OPENAI_API_KEY environment variable.")

    # Evaluator Model Choice Arguments
    parser.add_argument("--evaluator_model_type", type=str, choices=["GEMINI", "OPENAI"], default="GEMINI", help="Evaluator LLM type.")
    parser.add_argument("--gemini_model_name", type=str, default="gemini-1.5-flash", help="Specific Gemini model to use for evaluation.") # Updated default
    parser.add_argument("--openai_model_name", type=str, default="gpt-4o-mini", help="Specific OpenAI model to use for evaluation.") # Updated default

    # API Call Setting Arguments
    parser.add_argument("--api_retries", type=int, default=5, help="Number of retries for API calls.")
    parser.add_argument("--api_delay_seconds", type=float, default=0.1, help="Delay in seconds between retries and after each item processing.")
    parser.add_argument("--api_temperature", type=float, default=0.1, help="Temperature for LLM generation.")
    parser.add_argument("--api_max_output_tokens", type=int, default=2048, help="Max output tokens for LLM generation.")

    args = parser.parse_args()

    # Initialize API Clients based on parsed arguments
    gemini_client_instance_main = None
    openai_client_instance_main = None

    if args.evaluator_model_type == "GEMINI":
        if not args.google_api_key:
            print("Error: Gemini evaluator selected, but Google API Key is not provided via argument or GEMINI_API_KEY env var. Exiting.")
            exit(1)
        try:
            gemini_client_instance_main = genai.Client(api_key=args.google_api_key)
            print("Gemini client initialized.")
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            gemini_client_instance_main = None

    if args.evaluator_model_type == "OPENAI":
        if not args.openai_api_key:
            print("Error: OpenAI evaluator selected, but OpenAI API Key is not provided via argument or OPENAI_API_KEY env var. Exiting.")
            exit(1)
        try:
            openai_client_instance_main = OpenAI(api_key=args.openai_api_key)
            print("OpenAI client initialized.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            openai_client_instance_main = None

    # Validation for client initialization
    valid_config = True
    if args.evaluator_model_type == "GEMINI" and not gemini_client_instance_main:
        print("Error: Gemini client failed to initialize. Exiting.")
        valid_config = False
    elif args.evaluator_model_type == "OPENAI" and not openai_client_instance_main:
        print("Error: OpenAI client failed to initialize. Exiting.")
        valid_config = False
    
    if valid_config:
        run_evaluation(args, gemini_client_instance_main, openai_client_instance_main)
    else:
        print("Configuration or client initialization failed. Exiting.")