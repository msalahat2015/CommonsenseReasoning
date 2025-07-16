# eval.py (Revised for Task C: Commonsense Explanation - Generation)
import argparse
import csv
import json
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

# We will remove sklearn metrics as they are for classification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm
from datasets import load_from_disk
from blm.utils.prompter import Prompter
from blm.utils.log import log_message
import os

# Import the compute_metrics function from your metrics.py
# Make sure metrics.py is accessible in the PYTHONPATH or current directory
from metrics import compute_metrics, bert_score, bleu_score, rouge # Import specific functions as well if needed for debugging


# --- NEW: Function to extract generated reasons from the model's output ---
def parse_generated_reasons(generated_text):
    """
    Parses the model's generated text to extract three reasons.
    Expected format: "0: [Reason 0]\n1: [Reason 1]\n2: [Reason 2]"
    Returns a list of strings [reason0, reason1, reason2] or empty strings if not found.
    """
    reasons = ["", "", ""]
    # Use re.findall to find all lines starting with "0:", "1:", or "2:"
    # and capture the text after the label.
    # The regex looks for a digit, followed by ":", optional space, then captures everything until newline or end of string.
    matches = re.findall(r"(\d):\s*(.+?)(?:\n|$)", generated_text)

    for label, reason_text in matches:
        try:
            idx = int(label)
            if 0 <= idx <= 2:
                reasons[idx] = reason_text.strip()
        except ValueError:
            # Handle cases where the label might not be a valid integer if necessary
            pass
    return reasons


# Removed old extract_predicted_number_after_last_assistant and extract_predicted_label
# as they are not relevant for generation task evaluation.


def evaluate(model_path, test_data_path, hf_token, tokenizer_name, output_predictions_path, test_sample_limit):
    """Evaluates the model on the provided test dataset for a generation task."""
    log_message("INFO", "Evaluation start...")
    if hf_token:
        log_message("INFO", f"Logging into the Hugging Face Hub with token {hf_token[:10]}...")
        login(token=hf_token)

    log_message("INFO", f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log_message("INFO", "Initializing the prompter...")
    prompter = Prompter(tokenizer)
    log_message("INFO", "Prompter initialized successfully.")
    log_message("INFO", f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # Ensure special tokens are added and model embedding resized if necessary, consistent with training
    initial_token_count = len(tokenizer)
    # The Prompter's response_template should match the format the model is trained to generate (e.g., "0: ")
    # If the response_template is just a marker, ensure it's handled.
    # For this generation task, you likely don't need a specific response_template token in prompter,
    # as the model generates the structured output directly.
    # Check your Prompter implementation: if it adds special tokens, keep this block.
    # If it's a simple text formatting function, you might not need this.
    # Example: If Prompter.response_template is "### Assistant:\n"
    # response_template = prompter.response_template
    # added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    # model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)


    model.eval()
    log_message("INFO", "Model loaded successfully in evaluation mode.")

    # Lists to store generated outputs and references for metric calculation
    all_generated_texts = []
    all_reference_texts = [] # Will store concatenated reasons from dataset

    # To store individual results for CSV if needed
    evaluation_results_for_csv = []

    log_message("INFO", f"Loading test dataset from disk: {test_data_path}")
    try:
        test_dataset = load_from_disk(test_data_path)
        log_message("INFO", f"Test dataset loaded successfully with {test_dataset.num_rows} rows.")
    except FileNotFoundError:
        log_message("ERROR", f"Test dataset not found at: {test_data_path}")
        return
    except Exception as e:
        log_message("ERROR", f"Error loading test dataset: {e}")
        return

    log_message("INFO", "Mapping the prompter to the dataset...")
    # The prompter should now only take the 'statement' to form the user prompt
    # It should NOT include the assistant's response part (reasons) from the dataset.
    # Ensure your Prompter.__call__ method correctly generates only the user prompt.
    #test_dataset = test_dataset.map(lambda examples: {'prompt': prompter(examples['prompt'])}, batched=True)
    test_dataset = test_dataset.map(prompter, batched=True)
    log_message("INFO", "Prompter mapping to the dataset complete.")


    log_message("INFO", "Preparing prompts and generating explanations...")
    prompt_count = 1
    for example in tqdm(test_dataset, desc="Generating Explanations"):
        prompt = example['prompt']
        # The true references for this generation task are the reasons from the dataset
        # We need to reconstruct the expected concatenated format for comparison
        # Ensure your dataset has 'reason1', 'reason2', 'reason3' columns.
        reference_text = example['label']
        all_reference_texts.append(reference_text) # Store the full concatenated reference

        log_message("INFO", f"\n-------------------------------------------------------------------------")
        log_message("INFO", f"Prompt:({prompt_count}) ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Adjust max_new_tokens to accommodate the expected length of 3 reasons
            # A rough estimate for 3 reasons might be 50-100 tokens per reason, plus labels/newlines.
            # You might need to experiment with this value.
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000, # Increased max_new_tokens for generation
                do_sample=True, # Often prefer do_sample=True for generation tasks for diversity
                temperature=0.7, # Example temperature
                top_p=0.9 # Example top_p
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = extract_generated_response(generated_text, "")
        # Extract only the generated part from the full output (remove the input prompt repetition)
        # This assumes the model's output starts with the response after the prompt.
        # You may need to refine this based on your Prompter's exact output format.
        # A common pattern is to find the last occurrence of the prompt and take the rest.
        if prompt in generated_text:
            generated_response = generated_text[generated_text.rfind(prompt) + len(prompt):].strip()
        else:
            generated_response = generated_text.strip() # Fallback if prompt isn't directly repeated

        all_generated_texts.append(generated_response) # Store the generated response

        log_message("INFO", f"Generated text: ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{generated_response}")
        log_message("INFO", f"Reference Text: {reference_text}")
        log_message("INFO", f"--------------------------------------------------------")

        # Store for CSV output (optional, customize columns as needed)
        evaluation_results_for_csv.append({
            'statement': example['prompt'],
            'generated_explanation': generated_response,
            'reference_reason': example['label']
        })

        prompt_count += 1
        if prompt_count > test_sample_limit: # Use '>' or '>=' based on if test_sample_limit is inclusive
            break

    # # Save individual predictions if an output path is provided
    # output_directory = os.path.dirname(output_predictions_path)
    # os.makedirs(output_directory, exist_ok=True)
    # if output_predictions_path:
    #     log_message("INFO", f"Saving individual predictions to: {output_predictions_path}")
    #     # Customize fieldnames based on what you stored in evaluation_results_for_csv
    #     fieldnames = ['statement', 'generated_explanation', 'reference_reason1', 'reference_reason2', 'reference_reason3']
    #     with open(output_predictions_path, 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(evaluation_results_for_csv)
    #     log_message("INFO", "Individual predictions saved successfully.")

    log_message("INFO", "Calculating generation metrics...")

    # Create a temporary DataFrame for compute_metrics
    # The compute_metrics function expects 'ref_col' and 'output_col'
    # We will use the concatenated string as the reference for BERTScore, ROUGE, BLEU
    df_metrics = pd.DataFrame({
        'reference_text': all_reference_texts,
        'generated_text': all_generated_texts
    })

    metrics = bert_score(all_reference_texts, all_generated_texts)
    
    log_message("INFO", f"Metrics calculated by bert_score: {metrics}")

    # Use the compute_metrics function from your metrics.py
    # We are comparing the full generated text against the full concatenated reference text
    metrics = compute_metrics(
        df_metrics,
        ref_col='reference_text',
        output_col='generated_text',
        get_all_scores=False # Set to True if you want individual scores for analysis
    )

    log_message("INFO", "--- Evaluation Metrics ---")
    for metric_name, value in metrics.items():
        if isinstance(value, dict) and 'bleu' in value: # For BLEU_JSON detail
            log_message("INFO", f"BLEU Score: {value['bleu'] * 100:.2f}%")
        else:
            log_message("INFO", f"{metric_name}: {value:.4f}")

    log_message("INFO", "Evaluation end.")

def extract_generated_response(full_generated_text, statement_text):
    """
    Extracts the newly generated text (reasons) from the full model output.
    It expects the prompt to end with "Respond **in Arabic** :".
    It then cleans up any numerical prefixes (e.g., "1. ") and placeholder text.
    """
    # Define the fixed end part of the prompt that the model should generate *after*.
    # Make sure this matches the exact string given to the tokenizer.
    prompt_ending_marker = "Respond **in Arabic** :"

    # Find the last occurrence of the prompt ending marker.
    # This accounts for cases where the model might repeat parts of the prompt.
    marker_index = full_generated_text.rfind(prompt_ending_marker)

    if marker_index != -1:
        # Extract the part of the string that comes *after* the marker.
        # Add the length of the marker to get past it.
        raw_generated_part = full_generated_text[marker_index + len(prompt_ending_marker):].strip()
    else:
        # Fallback if the marker isn't found (less ideal, implies prompt structure changed or model deviated wildly)
        # In this scenario, we might need to rely on the "0:" or other starting patterns if applicable,
        # but given your current prompt, the absence of the marker indicates a problem.
        # For now, let's assume the generation is the last significant block of text.
        log_message("WARNING", "Prompt ending marker not found. Attempting a less precise extraction.")
        # Try to find the first newline that marks the start of the actual output
        first_newline_after_statement = full_generated_text.find(statement_text)
        if first_newline_after_statement != -1:
            raw_generated_part = full_generated_text[first_newline_after_statement + len(statement_text):].strip()
        else:
            raw_generated_part = full_generated_text.strip() # Last resort


    # Now, clean up the extracted `raw_generated_part`.
    # Split by newline, remove numbering (1., 2., 3.), and placeholder text.
    cleaned_lines = []
    # Regex to match optional leading whitespace, then a digit, then a dot, then optional whitespace
    # (e.g., "   1. ")
    numbering_pattern = re.compile(r'^\s*\d+\.\s*')
    # Regex to match common placeholder patterns.
    placeholder_pattern = re.compile(r'\[Your (first|second|third) reason here\]', re.IGNORECASE)


    for line in raw_generated_part.split('\n'):
        # 1. Remove numbering (e.g., "1. ")
        cleaned_line = numbering_pattern.sub('', line).strip()

        # 2. Remove placeholder text remnants (e.g., "[Your first reason here]")
        cleaned_line = placeholder_pattern.sub('', cleaned_line).strip()

        # Only add non-empty lines
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    # Join the cleaned lines back with a newline
    final_generated_text = "\n".join(cleaned_lines)
    
    return final_generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a language model for common sense reasoning generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved merged model directory")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the HuggingFace dataset directory for test data")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token (optional)")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer name or path")
    parser.add_argument("--output_predictions_path", type=str, default="evaluation_results/predictions_generation.csv", help="Path to save individual predictions (optional)")
    parser.add_argument("--test_sample_limit", type=int, default=10, help="Number of test samples used in evaluation (optional)")

    args = parser.parse_args()
    evaluate(args.model_path, args.test_data_path, args.hf_token, args.tokenizer_name, args.output_predictions_path, args.test_sample_limit)