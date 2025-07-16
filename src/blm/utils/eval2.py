# eval.py (Final Consolidated Fixes + Prompter `messages` column handling)
import argparse
import csv
import json
import re
import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets import load_from_disk # Assuming datasets library is used for your test data

# --- Custom Utilities (Adjust these imports if your structure is different) ---
# Assuming these are in blm/utils. You need to ensure they match your actual files.
# If your prompter is simple, you can integrate it directly as shown below
# by uncommenting the placeholder class and adapting it.
# from blm.utils.prompter import Prompter
# from blm.utils.log import log_message

# Placeholder for your log_message function if you don't have blm/utils/log.py
def log_message(level, message):
    """Simple logger function for demonstration. Replace with your actual log."""
    print(f"{level}: {message}")

# IMPORTANT: THIS IS A PLACEHOLDER PROMPTER.
# YOU MUST REPLACE THIS WITH THE ACTUAL LOGIC FROM YOUR
# 'blm/utils/prompter.py' FILE.
# This version is adapted to handle the 'messages' column structure.
class Prompter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Example hardcoded templates (adjust based on your actual prompter.py)
        # These are based on your previous logs indicating 'gpt' family templates
        self.instruction_template = '<user>'
        self.response_template = '<assistant>'
        log_message("INFO", "Initializing Prompter.")
        log_message("INFO", f"Tokenizer name or path: {tokenizer.name_or_path}")
        
        # This part might be in your original prompter.py too
        # Your previous logs showed 'Detected model family: Kamyar-zeinalipour' and then 'Selected instruction template: '<user>''
        # This implies a mapping logic. For this placeholder, we'll assume the 'gpt' template is selected.
        self.model_family = "gpt" 
        
        template_map = {
            'mistralai': {'instruction_template': '[INST]', 'response_template': '[/INST]'},
            'meta-llama': {'instruction_template': '<|system|>', 'response_template': '<|assistant|>'},
            'microsoft': {'instruction_template': '<|system|>', 'response_template': '<|assistant|>'},
            'default': {'instruction_template': '###Instructions:\n\n', 'response_template': '###Assistant:\n\n'},
            'gpt': {'instruction_template': '<user>', 'response_template': '<assistant>'}
        }
        
        selected_templates = template_map.get(self.model_family, template_map['default'])
        self.instruction_template = selected_templates['instruction_template']
        self.response_template = selected_templates['response_template']
        
        log_message("INFO", f"Selected instruction template: '{self.instruction_template}'")
        log_message("INFO", f"Selected response template: '{self.response_template}'")


    def __call__(self, examples):
        """
        Transforms a batch of raw examples into prompts and labels suitable for model input.
        This version is adapted to handle the 'messages' column, which is expected to be
        a list of dictionaries, e.g., [{'role': 'user', 'content': '...'}]
        
        Args:
            examples (dict): A dictionary of lists, where keys are column names from your dataset.
                             E.g., {'messages': [[{'role': 'user', 'content': 'sentence1'}]], 'label': [0]}
        Returns:
            dict: A dictionary with 'prompt' (list of formatted prompts) and 'label' (list of ground truth labels).
        """
        prompts = []
        labels = []

        # --- IMPORTANT: ADJUST THESE COLUMN NAMES TO MATCH YOUR ACTUAL DATASET ---
        input_text_column_name = 'messages' # Corrected from 'sentence1' to 'messages'
        ground_truth_label_column_name = 'label' # This seems to be correct based on your error
        # --- END IMPORTANT ADJUSTMENT ---

        # Basic check if the expected columns exist in the incoming examples batch
        if input_text_column_name not in examples:
            raise ValueError(f"Prompter Error: Input text column '{input_text_column_name}' not found in examples. Available: {list(examples.keys())}")
        if ground_truth_label_column_name not in examples:
            raise ValueError(f"Prompter Error: Ground truth label column '{ground_truth_label_column_name}' not found in examples. Available: {list(examples.keys())}")


        for i in range(len(examples[input_text_column_name])):
            messages_list = examples[input_text_column_name][i]
            ground_truth = examples[ground_truth_label_column_name][i]

            # --- Logic to extract instruction from 'messages' list ---
            instruction = ""
            # Assuming 'messages_list' is like [{'role': 'user', 'content': '...'}]
            # You might need more complex logic if there are multiple turns or system prompts
            for message in messages_list:
                if message['role'] == 'user':
                    instruction += message['content'] # Concatenate user messages if multiple
                # Add other roles if they are part of the prompt structure for the model
                # elif message['role'] == 'assistant':
                #     instruction += self.response_template + message['content'] 
                # elif message['role'] == 'system':
                #     instruction = "<|system|>" + message['content'] + instruction # Example for system role
            # --- End logic to extract instruction ---

            # Construct the final prompt string for the model
            prompt = (
                self.instruction_template + instruction + self.response_template
            )
            prompts.append(prompt)
            labels.append(ground_truth) # Append the extracted ground truth label

        # THIS IS THE CRUCIAL PART: MAKE SURE YOU RETURN A DICTIONARY WITH 'prompt' AND 'label' KEYS
        return {'prompt': prompts, 'label': labels}


# --- Utility functions (remain the same) ---
def extract_predicted_number_after_last_assistant(text):
    """
    Extracts a single digit (0, 1, or 2) that appears immediately after
    the last occurrence of an assistant-like response, or at the very end.
    Prioritizes a single digit at the end of the stripped text.
    """
    stripped_text = text.strip()
    if stripped_text:
        last_char = stripped_text[-1]
        try:
            last_number_int = int(last_char)
            if 0 <= last_number_int <= 2: # Assuming labels are 0, 1, or 2
                return last_number_int
        except ValueError:
            pass # Not a digit, continue to regex
    
    # Fallback to regex if last character is not a valid digit
    # This regex looks for "Respond with only the label number. X" or similar
    match = re.search(r"Respond with only the label number\.\s*(\d)", text)
    if match:
        predicted_number = match.group(1)
        if predicted_number in ["0", "1", "2"]:
            return int(predicted_number)
    return None

def extract_predicted_label(generated_text):
    # This function seems redundant given extract_predicted_number_after_last_assistant
    # and might need to be adjusted based on desired output.
    # Keeping it for now as it was in your original code.
    generated_lower = generated_text.lower()
    if "sentence 0" in generated_lower or "first sentence" in generated_lower or "0" in generated_lower:
        return 0
    elif "sentence 1" in generated_lower or "second sentence" in generated_lower or "1" in generated_lower:
        return 1
    return None

# --- Main Evaluation Function ---
def evaluate(model_path, test_data_path, hf_token, tokenizer_name, output_predictions_path, test_sample_limit):
    """Evaluates the model on the provided test dataset using the Prompter."""
    log_message("INFO", "Evaluation start...")
    if hf_token:
        log_message("INFO", f"Logging into the Hugging Face Hub with token {hf_token[:10]}...")
        login(token=hf_token)

    log_message("INFO", f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    log_message("INFO", "Initializing the prompter...")
    prompter = Prompter(tokenizer) # Pass tokenizer to Prompter
    log_message("INFO", "Prompter initialized successfully.")
    
    # Add special tokens from prompter's response_template to tokenizer
    response_template = prompter.response_template
    target_tokenizer_vocab_size = len(tokenizer) # Store the tokenizer's current size (before adding)
    added_token_count = 0
    if response_template and response_template not in tokenizer.additional_special_tokens:
        added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
        target_tokenizer_vocab_size = len(tokenizer) # Update to reflect new token
        log_message("INFO", f"Added {added_token_count} new special tokens from prompter's response_template.")
        log_message("INFO", f"Tokenizer vocabulary size after adding token(s): {target_tokenizer_vocab_size}")
    else:
        log_message("INFO", "Prompter's response_template token already exists in tokenizer or is empty.")
    
    # Ensure pad_token and eos_token are correctly set for generation if they're not.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log_message("INFO", f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id}) for generation.")


    log_message("INFO", f"Loading model from: {model_path}")
    model = None

    # --- Quantization Configuration (Choose 4-bit for most memory saving) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )
    
    # --- Determine the Base Model ---
    # Based on the error, the PEFT adapter expects a base model with vocab size 32768.
    # Mistral-7B-v0.1 is a common base.
    base_model_name = "mistralai/Mistral-7B-v0.1" 
    log_message("INFO", f"Inferred base model for PEFT adapters: {base_model_name}")

    try:
        # Step 1: Load the base model with quantization
        log_message("INFO", f"Loading base model ({base_model_name}) with quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=hf_token,
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16 # Ensures compute dtype is bfloat16 for 4-bit operations
        )
        log_message("INFO", "Base model loaded.")

        # Step 2: Resize the base model's embeddings to match the vocab size the PEFT adapter EXPECTS (32768)
        PEFT_EXPECTED_VOCAB_SIZE = 32768 # Crucial: This comes directly from the "size mismatch" error
        
        current_base_model_vocab_size = base_model.config.vocab_size

        if current_base_model_vocab_size != PEFT_EXPECTED_VOCAB_SIZE:
            log_message("INFO", f"Resizing base model embeddings from {current_base_model_vocab_size} to {PEFT_EXPECTED_VOCAB_SIZE} tokens to match PEFT adapter's expectation.")
            base_model.resize_token_embeddings(PEFT_EXPECTED_VOCAB_SIZE)
            log_message("INFO", "Base model embeddings resized for PEFT adapter.")
        else:
            log_message("INFO", "Base model vocabulary size already matches PEFT adapter's expectation. No resize needed.")

        # Step 3: Load the PEFT adapters on top of the (now correctly sized) base model
        log_message("INFO", f"Loading PEFT adapters from {model_path} onto the base model...")
        model = PeftModel.from_pretrained(base_model, model_path, token=hf_token)
        log_message("INFO", "PEFT adapters loaded successfully onto the base model.")

        # Step 4: Merge adapter weights into the base model for cleaner inference
        log_message("INFO", "Merging PEFT adapters into the base model for inference...")
        model = model.merge_and_unload()
        log_message("INFO", "PEFT adapters merged successfully.")

        # Step 5: Now, resize the *merged* model's embeddings to match your tokenizer's FINAL size (32770)
        # This accounts for the token YOU added via Prompter.
        if target_tokenizer_vocab_size != model.config.vocab_size:
            log_message("INFO", f"Resizing merged model embeddings from {model.config.vocab_size} to {target_tokenizer_vocab_size} tokens to match tokenizer's final vocabulary size.")
            model.resize_token_embeddings(target_tokenizer_vocab_size)
            log_message("INFO", "Merged model embeddings resized to match tokenizer.")
        else:
            log_message("INFO", "Merged model and tokenizer vocabulary sizes match. No final embedding resize needed.")

        # The model should already be on the correct device and have the correct compute dtype
        # due to `device_map="auto"` and `bnb_4bit_compute_dtype=torch.bfloat16` in bnb_config.
        # DO NOT re-cast a bitsandbytes model after loading.
        # if torch.cuda.is_available(): # Only if you absolutely need to manually move it
        #     model = model.cuda()


    except Exception as e:
        log_message("ERROR", f"Failed to load model and/or PEFT adapters: {e}")
        model = None 

    if model is None:
        log_message("ERROR", "Failed to load any model type. Exiting evaluation.")
        return

    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    log_message("INFO", "Model set to evaluation mode.")

    predictions = []
    ground_truth_labels = []
    evaluation_results = [] 

    log_message("INFO", f"Loading test dataset from disk: {test_data_path}")
    try:
        test_dataset = load_from_disk(test_data_path)
        log_message("INFO", f"Test dataset loaded successfully with {test_dataset.num_rows} rows.")
        
        # --- DEBUGGING LINES FOR KEYERROR: 'label' ---
        log_message("INFO", f"Original test dataset columns: {test_dataset.column_names}")
        if test_dataset.num_rows > 0:
            first_example = test_dataset[0]
            log_message("INFO", f"First example keys: {list(first_example.keys())}")
            log_message("INFO", f"First example content (first 500 chars): {str(first_example)[:500]}...")
        # --- END DEBUGGING LINES ---

    except FileNotFoundError:
        log_message("ERROR", f"Test dataset not found at: {test_data_path}. Please check the path.")
        return
    except Exception as e:
        log_message("ERROR", f"Error loading test dataset: {e}")
        return
    
    log_message("INFO", "Mapping the prompter to the dataset...")
    # This line uses the __call__ method of the Prompter class
    test_dataset = test_dataset.map(prompter, batched=True, remove_columns=test_dataset.column_names)
    log_message("INFO", "Prompter mapping to the dataset complete.")

    log_message("INFO", "Preparing prompts and generating predictions...")
    promptCount = 1
    dataset_to_evaluate = test_dataset
    if test_sample_limit is not None and test_sample_limit > 0:
        dataset_to_evaluate = test_dataset.select(range(min(test_sample_limit, len(test_dataset))))
        log_message("INFO", f"Evaluating a limited sample of {len(dataset_to_evaluate)} from the dataset.")


    for example in tqdm(dataset_to_evaluate, desc="Evaluating"):
        # This is the line that caused 'KeyError: 'label' previously.
        # It relies on your prompter.__call__ returning a 'label' key.
        ground_truth_label = int(example['label']) # Ensure 'label' key exists here!
        ground_truth_labels.append(ground_truth_label)
        
        log_message("INFO", f"\n-------------------------------------------------------------------------")
        log_message("INFO", f"Prompt ({promptCount}):")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{example['prompt']}") # Access the 'prompt' key from the mapped dataset
        
        inputs = tokenizer(example['prompt'], return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10, # Keep this small for classification tasks
                do_sample=False,   # For deterministic output in evaluation
                pad_token_id=tokenizer.eos_token_id, # Ensure pad_token_id is set
                attention_mask=inputs.get("attention_mask", None)
            )
            # Decode only the newly generated tokens
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        predicted_label = extract_predicted_number_after_last_assistant(generated_text)
        
        log_message("INFO", f"Generated text: ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{generated_text.strip()}") 
        
        predictions.append(predicted_label)
        log_message("INFO", f"Ground Truth Label: {ground_truth_label}, Predicted Label: {predicted_label}")
        log_message("INFO", f"--------------------------------------------------------")
        
        evaluation_results.append({'actual_value': ground_truth_label, 'predicted_value': predicted_label})
        
        promptCount += 1
    
    if output_predictions_path:
        output_directory = os.path.dirname(output_predictions_path)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        log_message("INFO", f"Saving individual predictions to: {output_predictions_path}")
        with open(output_predictions_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['actual_value', 'predicted_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_results)
        log_message("INFO", "Individual predictions saved successfully.")

    filtered_pairs = [(gt, pred) for gt, pred in zip(ground_truth_labels, predictions) if pred is not None]

    if not filtered_pairs:
        log_message("WARNING", "No valid predictions found for metric calculation. All predictions were None.")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        valid_ground_truth = [item[0] for item in filtered_pairs]
        valid_predictions = [item[1] for item in filtered_pairs]

        unique_labels = sorted(list(set(valid_ground_truth + valid_predictions)))
        if len(unique_labels) > 2:
            average_strategy = 'weighted'
            log_message("INFO", f"Multi-class classification detected (Labels: {unique_labels}). Using '{average_strategy}' averaging for Precision, Recall, F1.")
        else:
            average_strategy = 'binary'
            log_message("INFO", f"Binary classification detected (Labels: {unique_labels}). Using '{average_strategy}' averaging for Precision, Recall, F1.")

        accuracy = accuracy_score(valid_ground_truth, valid_predictions)
        precision = precision_score(valid_ground_truth, valid_predictions, average=average_strategy, zero_division=0)
        recall = recall_score(valid_ground_truth, valid_predictions, average=average_strategy, zero_division=0)
        f1 = f1_score(valid_ground_truth, valid_predictions, average=average_strategy, zero_division=0)
        
        log_message("INFO", f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        log_message("INFO", f"Evaluation Precision ({average_strategy}): {precision * 100:.2f}%")
        log_message("INFO", f"Evaluation Recall ({average_strategy}): {recall * 100:.2f}%")
        log_message("INFO", f"Evaluation F1 Score ({average_strategy}): {f1 * 100:.2f}%")
            
    log_message("INFO", "Evaluation end.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a language model for common sense reasoning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory (PEFT adapter or merged model).")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the HuggingFace dataset directory for test data.")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token (optional). Required for private models or if rate-limited.")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer name or path (e.g., 'meta-llama/Llama-2-7b-hf'). Should match the model's tokenizer.")
    parser.add_argument("--output_predictions_path", type=str, default=None, help="Path to save individual predictions (optional, e.g., 'results/predictions.csv').")
    parser.add_argument("--test_sample_limit", type=int, default=None, help="Number of test samples to use for evaluation (optional). If None, evaluate all samples.")

    args = parser.parse_args()
    evaluate(args.model_path, args.test_data_path, args.hf_token, args.tokenizer_name, args.output_predictions_path, args.test_sample_limit)