import json
import random
import os
import time
from google import genai
from google.genai import types
from openai import OpenAI

# --- Configuration ---
MODEL1_JSON_PATH = "/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/results/scar_evaluation_output.json"
MODEL2_JSON_PATH = "/home/mila/s/shuyuan.zhang/shapley/summarize_from_feedback_details/results/rlhf_step1500_evaluation_output.json"
MODEL1_NAME = "SCAR"
MODEL2_NAME = "RLHF"
NUM_EVAL_SAMPLES = 100 # Set to a small number for testing, e.g., 5

# API Keys - Prefer environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# For OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Evaluator Choice: "GEMINI" or "OPENAI"
EVALUATOR_MODEL_TYPE = "GEMINI"  # Change to "OPENAI" to use GPT-4o
GEMINI_MODEL_NAME = "gemini-2.0-flash"
OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14" # o4-mini-2025-04-16

# API Call Settings
API_RETRIES = 5
API_DELAY_SECONDS = 0.01
API_TEMPERATURE = 0.1
API_MAX_OUTPUT_TOKENS = 2048 # Reduced for A/B/C decision + brief reasoning

# --- Initialize API Clients ---
# Google Gemini
if EVALUATOR_MODEL_TYPE == "GEMINI" and GOOGLE_API_KEY:
    gemini_model = genai.Client(api_key=GOOGLE_API_KEY)
else:
    gemini_model = None

# OpenAI
if EVALUATOR_MODEL_TYPE == "OPENAI" and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


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
    if not gemini_model:
        print("Error: Gemini client not initialized. Check API Key.")
        return None
    for attempt in range(retries):
        try:
            response = gemini_model.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    max_output_tokens=API_MAX_OUTPUT_TOKENS,
                    temperature=API_TEMPERATURE,
                )
            )
            return response.text
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
    if not openai_client:
        print("Error: OpenAI client not initialized. Check API Key.")
        return None
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
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
        # Get all non-empty lines, stripped of whitespace
        lines = [line.strip() for line in llm_response_text.strip().split('\n') if line.strip()]
        if not lines:
            print(f"Warning: LLM response was empty or only whitespace.")
            return None

        # Primary strategy: Check the very last line for a single character A, B, or C
        last_line_decision = lines[-1].upper()
        if last_line_decision in ['A', 'B', 'C']:
            return last_line_decision

        # Secondary strategy: Check if the last character of the last line is A, B, or C
        if last_line_decision.endswith(('A', 'B', 'C')):
            char = last_line_decision[-1]
            if char in ['A', 'B', 'C']: # Double check, should be redundant but safe
                 return char

        # Fallback strategy: Look for specific phrases in reverse order of lines
        # This is more robust if the LLM includes reasoning after the "Overall Decision:" line
        # or if the prompt asks for a specific format.
        print(f"Warning: Could not parse decision ('A', 'B', 'C') from last line: '{last_line_decision}'. Trying fallback.")
        for line in reversed(lines):
            # Check for a line that seems to explicitly state the decision choice
            # Useful if the prompt asks for "Decision: A" or similar
            cleaned_line_upper = line.upper()
            if cleaned_line_upper.startswith("DECISION: A") or cleaned_line_upper == "A": return 'A'
            if cleaned_line_upper.startswith("DECISION: B") or cleaned_line_upper == "B": return 'B'
            if cleaned_line_upper.startswith("DECISION: C") or cleaned_line_upper == "C": return 'C'

            # Your original fallback
            if line.startswith("**Overall Decision:**"):
                if "SUMMARY A IS BETTER" in cleaned_line_upper or "A IS BETTER" in cleaned_line_upper: return 'A'
                if "SUMMARY B IS BETTER" in cleaned_line_upper or "B IS BETTER" in cleaned_line_upper: return 'B'
                if "SIMILAR QUALITY" in cleaned_line_upper or "TIE" in cleaned_line_upper or "EQUALLY GOOD" in cleaned_line_upper: return 'C'
                # If "**Overall Decision:**" is found but no keyword matches, stop this fallback for this line
                break

        print(f"Warning: Could not reliably extract decision from response:\n###{llm_response_text}###")
        return None
    except Exception as e:
        print(f"Error extracting decision: {e}\nFrom response:\n{llm_response_text[500:]}...")
        return None


# --- Main Evaluation Logic ---
def run_evaluation():
    print("Loading datasets...")
    data_model1_full = load_json_dataset(MODEL1_JSON_PATH)
    data_model2_full = load_json_dataset(MODEL2_JSON_PATH)

    if not data_model1_full or not data_model2_full:
        print("Failed to load one or both datasets. Exiting.")
        return

    # Take the minimum length if datasets are different, up to NUM_EVAL_SAMPLES
    min_len = min(len(data_model1_full), len(data_model2_full))
    num_samples_to_run = min(min_len, NUM_EVAL_SAMPLES)

    if len(data_model1_full) != len(data_model2_full):
        print(f"Warning: Datasets have different lengths ({len(data_model1_full)} vs {len(data_model2_full)}). Using {num_samples_to_run} common samples.")

    data_model1 = data_model1_full[:num_samples_to_run]
    data_model2 = data_model2_full[:num_samples_to_run]


    print(f"Successfully loaded {len(data_model1)} items for comparison.")

    prompt_template_path = "prompt_template.txt"
    if not os.path.exists(prompt_template_path):
        print(f"Error: Prompt template file not found at {prompt_template_path}")
        print("Please create 'prompt_template.txt'. Example content:")
        print("""
        You are an expert evaluator comparing two summaries (Summary A and Summary B) of a given Document.
        Your goal is to determine which summary is better based on criteria such as accuracy, completeness, conciseness, and coherence.

        Document:
        {document_text}

        Summary A:
        {summary_a_text}

        Summary B:
        {summary_b_text}

        Please provide a brief step-by-step reasoning for your decision.
        Then, on the VERY LAST LINE, state your overall decision as a single capital letter:
        - 'A' if Summary A is better.
        - 'B' if Summary B is better.
        - 'C' if both summaries are of similar quality or it's a tie.

        Reasoning:
        [Your reasoning here]

        Overall Decision:
        [A, B, or C]
        """) # The "Overall Decision:" text in example should guide the LLM for a simple last line output
        return

    with open(prompt_template_path, "r", encoding='utf-8') as rf:
        prompt_template = rf.read()

    results = {
        MODEL1_NAME: 0,
        MODEL2_NAME: 0,
        "Tie/Inconclusive": 0,
        "Errors/NoDecision": 0
    }
    
    # Store detailed results for later analysis if needed
    detailed_results = []

    total_items = len(data_model1)
    for i in range(total_items):
        print(f"\n--- Evaluating item {i+1}/{total_items} ---")
        item_m1 = data_model1[i]
        item_m2 = data_model2[i]

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
            summary_a_origin_model = MODEL1_NAME
            summary_b_origin_model = MODEL2_NAME
            # print(f"Assigning: Summary A = {MODEL1_NAME}, Summary B = {MODEL2_NAME}")
        else:
            assigned_summary_a_text = summary_m2_text
            assigned_summary_b_text = summary_m1_text
            summary_a_origin_model = MODEL2_NAME
            summary_b_origin_model = MODEL1_NAME
            # print(f"Assigning: Summary A = {MODEL2_NAME}, Summary B = {MODEL1_NAME}")

        current_prompt = prompt_template.format(
            document_text=original_doc,
            summary_a_text=assigned_summary_a_text,
            summary_b_text=assigned_summary_b_text
        )

        llm_response = None
        print(f"Calling {EVALUATOR_MODEL_TYPE} API...")
        if EVALUATOR_MODEL_TYPE == "GEMINI":
            llm_response = call_gemini_api(current_prompt)
        elif EVALUATOR_MODEL_TYPE == "OPENAI":
            llm_response = call_openai_api(current_prompt)
        else:
            print(f"Error: Unknown EVALUATOR_MODEL_TYPE: {EVALUATOR_MODEL_TYPE}")
            results["Errors/NoDecision"] += 1
            continue

        current_result_detail = {
            "item_index": i,
            "document": original_doc,
            "summary_A_text": assigned_summary_a_text,
            "summary_A_origin": summary_a_origin_model,
            "summary_B_text": assigned_summary_b_text,
            "summary_B_origin": summary_b_origin_model,
            "llm_evaluator": EVALUATOR_MODEL_TYPE,
            "llm_response_raw": llm_response,
            "decision_extracted": None,
            "winner": None
        }

        if not llm_response:
            results["Errors/NoDecision"] += 1
            print("Skipping item due to API error or no response.")
            current_result_detail["winner"] = "Error/NoDecision"
            detailed_results.append(current_result_detail)
            time.sleep(API_DELAY_SECONDS) # Still sleep to avoid hammering on next attempt
            continue

        decision = extract_decision(llm_response)
        current_result_detail["decision_extracted"] = decision
        print(f"LLM Decision Raw: '{decision}' (Assignment: A={summary_a_origin_model}, B={summary_b_origin_model})")

        if decision == 'A':
            winner = summary_a_origin_model
            results[winner] += 1
            current_result_detail["winner"] = winner
            print(f"LLM preferred {winner} (assigned to A)")
        elif decision == 'B':
            winner = summary_b_origin_model
            results[winner] += 1
            current_result_detail["winner"] = winner
            print(f"LLM preferred {winner} (assigned to B)")
        elif decision == 'C':
            results["Tie/Inconclusive"] += 1
            current_result_detail["winner"] = "Tie/Inconclusive"
            print("LLM judged it as a Tie/Inconclusive.")
        else:
            results["Errors/NoDecision"] += 1
            current_result_detail["winner"] = "Error/NoDecision (Extraction)"
            print("Could not determine a clear decision from LLM response.")
        
        detailed_results.append(current_result_detail)
        time.sleep(API_DELAY_SECONDS)

    # Print final results
    print("\n--- Evaluation Complete ---")
    print("Final Results:")
    for model_name_key, count in results.items(): # Use a different variable name here
        print(f"{model_name_key}: {count}")

    total_compared_for_wins = results[MODEL1_NAME] + results[MODEL2_NAME]
    if total_compared_for_wins > 0:
        print(f"\nWin percentage for {MODEL1_NAME} (vs {MODEL2_NAME}, excluding ties & errors): {results[MODEL1_NAME] / total_compared_for_wins * 100:.2f}%")
        print(f"Win percentage for {MODEL2_NAME} (vs {MODEL1_NAME}, excluding ties & errors): {results[MODEL2_NAME] / total_compared_for_wins * 100:.2f}%")
    else:
        print("\nNo conclusive wins to calculate percentages.")

    # Save detailed results to a JSON file
    results_output_path = f"evaluation_results_{EVALUATOR_MODEL_TYPE.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_output_path, 'w', encoding='utf-8') as f_out:
            json.dump(detailed_results, f_out, indent=4)
        print(f"\nDetailed results saved to: {results_output_path}")
    except IOError as e:
        print(f"Error saving detailed results to {results_output_path}: {e}")

    summary_stats_path = f"evaluation_summary_{EVALUATOR_MODEL_TYPE.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(summary_stats_path, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=4)
        print(f"Summary statistics saved to: {summary_stats_path}")
    except IOError as e:
        print(f"Error saving summary statistics to {summary_stats_path}: {e}")


if __name__ == "__main__":
    # Basic check for API keys before running
    if EVALUATOR_MODEL_TYPE == "GEMINI" and (not GOOGLE_API_KEY):
        print("Error: Gemini evaluator selected, but GOOGLE_API_KEY is not configured. Exiting.")
    elif EVALUATOR_MODEL_TYPE == "OPENAI" and (not OPENAI_API_KEY):
        print("Error: OpenAI evaluator selected, but OPENAI_API_KEY is not configured. Exiting.")
    else:
        run_evaluation()