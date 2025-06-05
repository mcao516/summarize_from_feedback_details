import json
import random
import os
import time
from google import genai
from google.genai import types
from openai import OpenAI
import concurrent.futures
import threading

# --- Configuration ---
MODEL1_JSON_PATH = "/home/mila/c/caomeng/summarize_from_feedback_details/analysis/outputs/tldr_test_reference.json"
MODEL2_JSON_PATH = "/home/mila/c/caomeng/summarize_from_feedback_details/analysis/outputs/tldr_test_shap_span_20250603_023511.json"
MODEL1_NAME = "Reference"
MODEL2_NAME = "RLHF"
NUM_EVAL_SAMPLES = 1000
MAX_WORKERS = 30 # Number of threads for parallel processing

# API Keys - Prefer environment variables
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# For OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluator Choice: "GEMINI" or "OPENAI"
EVALUATOR_MODEL_TYPE = "GEMINI"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14" # o4-mini-2025-04-16

# API Call Settings
API_RETRIES = 5
API_DELAY_SECONDS = 0.1
API_TEMPERATURE = 0.1
API_MAX_OUTPUT_TOKENS = 2048


# --- Initialize API Clients ---
# Google Gemini
if EVALUATOR_MODEL_TYPE == "GEMINI" and GOOGLE_API_KEY:
    try:
        gemini_model_client = genai.Client(api_key=GOOGLE_API_KEY) # User's original client initialization
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        gemini_model_client = None
else:
    gemini_model_client = None

# OpenAI
if EVALUATOR_MODEL_TYPE == "OPENAI" and OPENAI_API_KEY:
    try:
        openai_client_instance = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        openai_client_instance = None
else:
    openai_client_instance = None


# --- Helper Functions ---
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


def call_gemini_api(prompt_text, retries=API_RETRIES, delay=API_DELAY_SECONDS):
    """Calls the Gemini API with retry logic."""
    if not gemini_model_client:
        print("Error: Gemini client not initialized. Check API Key or initialization.")
        return None
    for attempt in range(retries):
        try:
            response = gemini_model_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    max_output_tokens=API_MAX_OUTPUT_TOKENS,
                    temperature=API_TEMPERATURE,
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


def call_openai_api(prompt_text, retries=API_RETRIES, delay=API_DELAY_SECONDS):
    """Calls the OpenAI API with retry logic."""
    if not openai_client_instance:
        print("Error: OpenAI client not initialized. Check API Key or initialization.")
        return None
    for attempt in range(retries):
        try:
            response = openai_client_instance.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of text summaries."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=API_MAX_OUTPUT_TOKENS,
                temperature=API_TEMPERATURE
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
                # Check for A, B, C directly after this, or in common phrases
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
                       model1_name_const, model2_name_const, evaluator_type_const,
                       shared_results_dict, shared_detailed_results_list, results_lock, total_items_count):
    """Processes a single item evaluation. Designed to be run in a thread."""
    print(f"--- Evaluating item index {item_index} (Processing order may vary) ---")

    original_doc = item_m1['document']
    summary_m1_text = item_m1['summary']
    summary_m2_text = item_m2['summary']

    # Ensure document for model2 is the same if a common 'id' or 'document' is expected
    # For now, assuming item_m1['document'] is the reference. If item_m2 can have a different
    # 'document' for the same index, this needs careful handling based on dataset structure.
    # The current code implies they are paired by index and should share the document.

    assigned_summary_a_text = ""
    assigned_summary_b_text = ""
    summary_a_origin_model = ""
    summary_b_origin_model = ""

    is_m1_a = random.choice([True, False])
    if is_m1_a:
        assigned_summary_a_text = summary_m1_text
        assigned_summary_b_text = summary_m2_text
        summary_a_origin_model = model1_name_const
        summary_b_origin_model = model2_name_const
    else:
        assigned_summary_a_text = summary_m2_text
        assigned_summary_b_text = summary_m1_text
        summary_a_origin_model = model2_name_const
        summary_b_origin_model = model1_name_const

    current_prompt = prompt_template.format(
        document_text=original_doc,
        summary_a_text=assigned_summary_a_text,
        summary_b_text=assigned_summary_b_text
    )

    llm_response = None
    # print(f"Thread {threading.get_ident()}: Calling {evaluator_type_const} API for item {item_index}...") # Optional: for thread debugging
    if evaluator_type_const == "GEMINI":
        llm_response = call_gemini_api(current_prompt) # API_DELAY_SECONDS is for retries inside this
    elif evaluator_type_const == "OPENAI":
        llm_response = call_openai_api(current_prompt) # API_DELAY_SECONDS is for retries inside this
    else:
        print(f"Error: Unknown EVALUATOR_MODEL_TYPE: {evaluator_type_const}")
        with results_lock:
            shared_results_dict["Errors/NoDecision"] += 1
        # No detailed result to append here other than error, or could append a minimal one
        return # Exit task for this item

    current_result_detail = {
        "item_index": item_index,
        "document": original_doc,
        "summary_A_text": assigned_summary_a_text,
        "summary_A_origin": summary_a_origin_model,
        "summary_B_text": assigned_summary_b_text,
        "summary_B_origin": summary_b_origin_model,
        "llm_evaluator": evaluator_type_const,
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
        # print(f"Thread {threading.get_ident()}: LLM Decision for item {item_index} Raw: '{decision}' (A={summary_a_origin_model}, B={summary_b_origin_model})")

        with results_lock:
            if decision == 'A':
                winner = summary_a_origin_model
                shared_results_dict[winner] += 1
                current_result_detail["winner"] = winner
                # print(f"LLM preferred {winner} (assigned to A) for item {item_index}")
            elif decision == 'B':
                winner = summary_b_origin_model
                shared_results_dict[winner] += 1
                current_result_detail["winner"] = winner
                # print(f"LLM preferred {winner} (assigned to B) for item {item_index}")
            elif decision == 'C':
                shared_results_dict["Tie/Inconclusive"] += 1
                current_result_detail["winner"] = "Tie/Inconclusive"
                # print(f"LLM judged item {item_index} as a Tie/Inconclusive.")
            else:
                shared_results_dict["Errors/NoDecision"] += 1
                current_result_detail["winner"] = "Error/NoDecision_Extraction"
                # print(f"Could not determine a clear decision for item {item_index} from LLM response.")

    with results_lock:
        shared_detailed_results_list.append(current_result_detail)

    time.sleep(API_DELAY_SECONDS) # Small delay after processing each item, per thread


def run_evaluation():
    print("Loading datasets...")
    data_model1_full = load_json_dataset(MODEL1_JSON_PATH)
    data_model2_full = load_json_dataset(MODEL2_JSON_PATH)

    if not data_model1_full or not data_model2_full:
        print("Failed to load one or both datasets. Exiting.")
        return

    min_len = min(len(data_model1_full), len(data_model2_full))
    num_samples_to_run = min(min_len, NUM_EVAL_SAMPLES)

    if len(data_model1_full) != len(data_model2_full):
        print(f"Warning: Datasets have different lengths ({len(data_model1_full)} vs {len(data_model2_full)}). Using {num_samples_to_run} common samples.")

    data_model1 = data_model1_full[:num_samples_to_run]
    data_model2 = data_model2_full[:num_samples_to_run]

    if not data_model1: # handles num_samples_to_run being 0
        print("No samples to evaluate based on NUM_EVAL_SAMPLES or dataset lengths.")
        return

    print(f"Successfully loaded {len(data_model1)} items for comparison.")

    prompt_template_path = "prompt_template.txt"
    if not os.path.exists(prompt_template_path):
        print(f"Error: Prompt template file not found at {prompt_template_path}")
        # (Example prompt content printout omitted for brevity, it's in the original)
        return

    with open(prompt_template_path, "r", encoding='utf-8') as rf:
        prompt_template_content = rf.read()

    # Shared resources for threads
    results_summary = {
        MODEL1_NAME: 0,
        MODEL2_NAME: 0,
        "Tie/Inconclusive": 0,
        "Errors/NoDecision": 0
    }
    detailed_results_list = []
    processing_lock = threading.Lock()

    total_items_to_process = len(data_model1)
    print(f"\nStarting evaluation of {total_items_to_process} items using up to {MAX_WORKERS} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(total_items_to_process):
            item_m1 = data_model1[i]
            item_m2 = data_model2[i] # Assuming documents are paired correctly by index
            
            # Ensure documents match if that's an assumption.
            # If item_m1['document'] could differ from item_m2['document'] for the same conceptual item,
            # then a matching key (e.g., 'id') would be needed to pair them from the full datasets.
            # The current code structure implies data_model1[i] and data_model2[i] form a pair.
            if item_m1.get('document') != item_m2.get('document'):
                print("Documents are NOT the same!")
                pass

            futures.append(executor.submit(
                _process_item_task,
                    i,  # item_index
                    item_m1,
                    item_m2,
                    prompt_template_content,
                    MODEL1_NAME,
                    MODEL2_NAME,
                    EVALUATOR_MODEL_TYPE,
                    results_summary,
                    detailed_results_list,
                    processing_lock,
                    total_items_to_process
            ))

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # To raise exceptions if any occurred in the thread
            except Exception as exc:
                print(f'An item processing generated an exception: {exc}')

    # Sort detailed_results by item_index for consistent output if needed
    detailed_results_list.sort(key=lambda x: x["item_index"])

    # Print final results
    print("\n--- Evaluation Complete ---")
    print("Final Results:")
    for model_name_key, count in results_summary.items():
        print(f"{model_name_key}: {count}")

    total_compared_for_wins = results_summary[MODEL1_NAME] + results_summary[MODEL2_NAME]
    if total_compared_for_wins > 0:
        print(f"\nWin percentage for {MODEL1_NAME} (vs {MODEL2_NAME}, excluding ties & errors): {results_summary[MODEL1_NAME] / total_compared_for_wins * 100:.2f}%")
        print(f"Win percentage for {MODEL2_NAME} (vs {MODEL1_NAME}, excluding ties & errors): {results_summary[MODEL2_NAME] / total_compared_for_wins * 100:.2f}%")
    else:
        print("\nNo conclusive wins to calculate percentages.")
    
    # Save detailed results (optional, uncomment if needed)
    results_output_filename = f"evaluation_results_{EVALUATOR_MODEL_TYPE.lower()}_{MODEL1_NAME}_vs_{MODEL2_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_output_filename, 'w', encoding='utf-8') as f_out:
            json.dump(detailed_results_list, f_out, indent=2)
        print(f"\nDetailed results saved to: {results_output_filename}")
    except IOError as e:
        print(f"Error saving detailed results to {results_output_filename}: {e}")

    summary_stats_filename = f"evaluation_summary_{EVALUATOR_MODEL_TYPE.lower()}_{MODEL1_NAME}_vs_{MODEL2_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(summary_stats_filename, 'w', encoding='utf-8') as f_out:
            json.dump(results_summary, f_out, indent=2)
        print(f"Summary statistics saved to: {summary_stats_filename}")
    except IOError as e:
        print(f"Error saving summary statistics to {summary_stats_filename}: {e}")


if __name__ == "__main__":
    valid_config = True
    if EVALUATOR_MODEL_TYPE == "GEMINI":
        if not GOOGLE_API_KEY:
            print("Error: Gemini evaluator selected, but GOOGLE_API_KEY is not configured. Exiting.")
            valid_config = False
        if not gemini_model_client:
            print("Error: Gemini client failed to initialize. Exiting.")
            valid_config = False
    elif EVALUATOR_MODEL_TYPE == "OPENAI":
        if not OPENAI_API_KEY:
            print("Error: OpenAI evaluator selected, but OPENAI_API_KEY is not configured. Exiting.")
            valid_config = False
        if not openai_client_instance:
            print("Error: OpenAI client failed to initialize. Exiting.")
            valid_config = False
    else:
        print(f"Error: Unknown EVALUATOR_MODEL_TYPE: {EVALUATOR_MODEL_TYPE}. Choose 'GEMINI' or 'OPENAI'.")
        valid_config = False
    
    if valid_config:
        run_evaluation()