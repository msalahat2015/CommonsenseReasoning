# eval.py (Revised for Prompter __call__)
import argparse
import csv
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets import load_from_disk
from blm.utils.prompter import Prompter
from blm.utils.log import log_message
import os



def extract_predicted_number_after_last_assistant(text):
    last_char = text[-1]
    try:
     last_number_int = int(last_char)
     return last_number_int
    except ValueError:  
        match = re.search(r"Respond with only the label number\.\s*(\d)", text)
        if match:
            predicted_number = match.group(1)
            if predicted_number in ["0", "1", "2"]:
                return int(predicted_number)
        return None
    
    
def extract_predicted_label(generated_text):
    """Extracts the predicted label (0 or 1) from the LLM's response."""
    generated_lower = generated_text.lower()
    if "sentence 0" in generated_lower or "first sentence" in generated_lower or "0" in generated_lower:
        return 0
    elif "sentence 1" in generated_lower or "second sentence" in generated_lower or "1" in generated_lower:
        return 1
    return None

def evaluate(model_path, test_data_path, hf_token, tokenizer_name,output_predictions_path,test_sample_limit): # Removed prompt paths
    """Evaluates the model on the provided test dataset using the Prompter."""
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
   
    initial_token_count = len(tokenizer)
    response_template = prompter.response_template 
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)

    model.eval()
    log_message("INFO", "Model loaded successfully in evaluation mode.")
   
    predictions = []
    ground_truth_labels = []
    evaluation_results = [] # To store individual results

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
    test_dataset = test_dataset.map(prompter, batched=True)
    log_message("INFO", "Prompter mapping to the dataset complete.")

    log_message("INFO", "Preparing prompts using the prompter...")
    promptCount=1
    for example in tqdm(test_dataset, desc="Evaluating"):
        prompt = example['prompt']
        ground_truth_label = int(example['label'])
        ground_truth_labels.append(ground_truth_label)
        log_message("INFO", f"\n-------------------------------------------------------------------------")
        log_message("INFO", f"Prompt:({promptCount}) ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #predicted_label = extract_predicted_label(generated_text)
       
        predicted_label=extract_predicted_number_after_last_assistant(generated_text)
        log_message("INFO", f"Generated text: ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"{generated_text}")
        predictions.append(predicted_label)
        log_message("INFO", f"Ground Truth Label: {ground_truth_label}, Predicted Label: {predicted_label}")
        log_message("INFO", f"--------------------------------------------------------")
        evaluation_results.append({'actual_value': ground_truth_label, 'predicted_value': predicted_label})
        promptCount=promptCount+1
        if promptCount >= test_sample_limit:
            break
    # Save individual predictions if an output path is provided
    output_directory = os.path.dirname(output_predictions_path)
    os.makedirs(output_directory, exist_ok=True)
    if output_predictions_path:
        log_message("INFO", f"Saving individual predictions to: {output_predictions_path}")
        with open(output_predictions_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['actual_value', 'predicted_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_results)
        log_message("INFO", "Individual predictions saved successfully.")

    # Calculate metrics
    valid_predictions = [p for p in predictions if p is not None]
    valid_ground_truth = [g for i, g in enumerate(ground_truth_labels) if predictions[i] is not None]


    if not valid_ground_truth:
        log_message("WARNING", "No valid predictions found for metric calculation.")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        accuracy = accuracy_score(valid_ground_truth, valid_predictions)
        precision = precision_score(valid_ground_truth, valid_predictions)
        recall = recall_score(valid_ground_truth, valid_predictions)
        f1 = f1_score(valid_ground_truth, valid_predictions)
        log_message("INFO", f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        log_message("INFO", f"Evaluation Precision: {precision * 100:.2f}%")
        log_message("INFO", f"Evaluation Recall: {recall * 100:.2f}%")
        log_message("INFO", f"Evaluation F1 Score: {f1 * 100:.2f}%")
        
log_message("INFO", f"Evaluation end")
        
             


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a language model for common sense reasoning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved merged model directory")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the HuggingFace dataset directory for test data")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token (optional)")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer name or path")
    parser.add_argument("--output_predictions_path", type=str, default=None, help="Path to save individual predictions (optional)")
    parser.add_argument("--test_sample_limit", type=int, default=10, help="no of test sample used in evaluation (optional)")

    args = parser.parse_args()
    evaluate(args.model_path, args.test_data_path, args.hf_token, args.tokenizer_name, args.output_predictions_path,args.test_sample_limit)