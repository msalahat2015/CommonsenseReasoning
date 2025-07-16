from blm.utils.prompter import Prompter
from blm.utils.log import log_message
import torch
import os
import json
import argparse
import logging
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from blm.utils.helpers import logging_config

logger = logging.getLogger(__name__)


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

def main(args):
    log_message("INFO", f"Starting LoRA merging process.")
    log_message("INFO", f"Merged path will be: {args.merged_path}")
    os.makedirs(args.merged_path, exist_ok=True)
    log_message("INFO", f"Created merged output directory: {args.merged_path}")

    log_message("INFO", f"LoRA checkpoint path: {args.lora_path}")
    device = None if torch.cuda.is_available() else "cpu"
    log_message("INFO", f"Using device: {'cuda' if device is None else device}")

    adapter_config_file = os.path.join(args.lora_path, "adapter_config.json")
    log_message("INFO", f"Loading adapter configuration from: {adapter_config_file}")
    try:
        with open(adapter_config_file, "r") as fh:
            adapter_config = json.load(fh)
        log_message("INFO", f"Adapter configuration loaded successfully: {adapter_config}")
    except FileNotFoundError:
        log_message("ERROR", f"Adapter configuration file not found at: {adapter_config_file}")
        return
    except json.JSONDecodeError:
        log_message("ERROR", f"Error decoding JSON from adapter configuration file at: {adapter_config_file}")
        return

    model_id = adapter_config.get("base_model_name_or_path")
    if not model_id:
        log_message("ERROR", "Base model name or path not found in adapter configuration.")
        return

    # Load base model, model ID is in the adapter configuration
    log_message("INFO", f"Loading base model: {model_id}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        log_message("INFO", f"Base model loaded successfully.")
    except Exception as e:
        log_message("ERROR", f"Error loading base model {model_id}: {e}")
        return

    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompter = Prompter(tokenizer)
    initial_token_count = len(tokenizer)
    response_template = prompter.response_template 
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    base_model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)
    # Load the PEFT model
    log_message("INFO", f"Loading PEFT model from: {args.lora_path}")
    try:
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model = model.to(device)
        log_message("INFO", f"PEFT model loaded successfully and moved to device: {'cuda' if device is None else device}")
    except Exception as e:
        log_message("ERROR", f"Error loading PEFT model from {args.lora_path}: {e}")
        return

    # Merge PEFT with base model
    log_message("INFO", f"Merging PEFT weights into the base model.")
    try:
        model = model.merge_and_unload()
        log_message("INFO", f"PEFT weights merged successfully.")
    except Exception as e:
        log_message("ERROR", f"Error during PEFT weights merging: {e}")
        return

    # Save model and tokenizer
    log_message("INFO", "Saving merged model.")
    try:
        model.save_pretrained(args.merged_path, safe_serialization=True, max_shard_size="2GB")
        log_message("INFO", f"Merged model saved successfully to: {args.merged_path}")
    except Exception as e:
        log_message("ERROR", f"Error saving merged model to {args.merged_path}: {e}")
        return

    log_message("INFO", "Loading and saving tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(args.merged_path)
        log_message("INFO", f"Tokenizer loaded and saved successfully to: {args.merged_path}")
    except Exception as e:
        log_message("ERROR", f"Error loading and saving tokenizer: {e}")
        return

    log_message("INFO", "LoRA merging process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, help="Path to LoRA adapters")
    parser.add_argument("--merged_path", type=str, help="Local path where the merged LoRA adapters are saved")
    parser.add_argument("--hf_token", type=str, help="Huggingface token")
    args = parser.parse_args()

    if args.hf_token:
        log_message("INFO", f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        try:
            login(token=args.hf_token)
            log_message("INFO", "Successfully logged into the Hugging Face Hub.")
        except Exception as e:
            log_message("ERROR", f"Error logging into the Hugging Face Hub: {e}")

    logging_config("/rep/msalahat/LLMTraining/log/merge_lora.log")
    log_message("INFO", "Logging configured to merge_lora.log")

    main(args)