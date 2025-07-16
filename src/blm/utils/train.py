import io
import sys
from datasets import load_from_disk
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from blm.utils.peft import create_and_prepare_model
from blm.utils.prompter import Prompter
import os
import datetime
from  blm.utils.log import log_message 

logger = logging.getLogger(__name__)

def train(args):
   
    try:
        # Load and create peft model
        log_message("INFO", "Loading and preparing PEFT model...")
        model, peft_config, tokenizer = create_and_prepare_model(args)
        log_message("INFO", "PEFT model loaded and prepared successfully.")
        
        log_message("INFO", "Checking if pad_token_id and eos_token_id are the same...")
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            log_message("warning", "pad_token_id and eos_token_id are the same.")
            print("pad_token_id and eos_token_id are the same.")
            # Try setting pad_token_id to a different existing special token ID
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                log_message("INFO", f"pad_token_id set to unk_token_id: {tokenizer.pad_token_id}")
                print(f"pad_token_id set to unk_token_id: {tokenizer.pad_token_id}")
            elif tokenizer.bos_token_id is not None and tokenizer.bos_token_id != tokenizer.eos_token_id:
                tokenizer.pad_token_id = tokenizer.bos_token_id
                log_message("INFO", f"pad_token_id set to bos_token_id: {tokenizer.pad_token_id}")
                print(f"pad_token_id set to bos_token_id: {tokenizer.pad_token_id}")
            else:
                # If no other suitable special token exists, you might need to add one
                # and then set its ID as the pad_token_id. This is more advanced.
                log_message("warning", "Warning: No suitable existing special token found for padding. Consider adding a new special token.")
                print("Warning: No suitable existing special token found for padding. Consider adding a new special token.")
        else:
            log_message("INFO", "pad_token_id and eos_token_id are already different.")
            print("pad_token_id and eos_token_id are already different.")

        # Handle padding token
        log_message("INFO", "Handling padding token...")
        if tokenizer.pad_token is None:
            special_tokens_dict = {'pad_token': '<pad>'}
            tokenizer.add_special_tokens(special_tokens_dict)
            log_message("INFO", f"Added padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
            print(f"Added padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        elif tokenizer.pad_token_id == tokenizer.eos_token_id:
            special_tokens_dict = {'pad_token': '<pad>'}
            tokenizer.add_special_tokens(special_tokens_dict)
            log_message("INFO", f"Added padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
            print(f"Added padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        else:
            log_message("info", f"Padding token is already set: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
            print(f"Padding token is already set: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        log_message("INFO", f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

        model.config.use_cache = False

        log_message("INFO", "Initializing the prompter...")
        prompter = Prompter(tokenizer)
        log_message("INFO", "Prompter initialized successfully.")
        
        log_message("INFO", f"Loading dataset from disk: {args.data_path}")
        dataset = load_from_disk(args.data_path)
        log_message("INFO", f"Dataset loaded successfully with {dataset.num_rows} rows.")
        
        
        log_message("INFO", "Mapping the prompter to the dataset...")
        dataset = dataset.map(prompter, batched=True)
        log_message("INFO", "Prompter mapping to the dataset complete.")

        log_message("INFO", f"Saving tokenizer to: {args.output_dir}")
        tokenizer.save_pretrained(args.output_dir)
        log_message("INFO", f"Tokenizer saved successfully to: {args.output_dir}")

        log_message("INFO", "--- Special Tokens Information ---")
        log_message("INFO", "The following special tokens and their assigned names are configured in the tokenizer:")
        log_message("INFO", f"{tokenizer.special_tokens_map}")
        log_message("INFO", "--- End of Special Tokens Information ---")

        if peft_config:
            log_message("INFO", f"Saving PEFT configuration to: {args.output_dir}")
            peft_config.save_pretrained(args.output_dir)
            log_message("INFO", f"PEFT configuration saved successfully to: {args.output_dir}")
        else:
            log_message("INFO", "No PEFT configuration found. Skipping saving PEFT config.")

        if hasattr(model, "config"):
            log_message("INFO", f"Saving model configuration to: {args.output_dir}")
            model.config.save_pretrained(args.output_dir)
            log_message("INFO", f"Model configuration saved successfully to: {args.output_dir}")
        else:
            log_message("INFO", "Model object does not have a 'config' attribute. Skipping saving model config.")

        log_message("INFO", "Initializing the data collator for completion-only LM...")
        
        if tokenizer.chat_template is  None:
            log_message("WARNING", "Tokenizer chat template is None. Using default template.")
            tokenizer.chat_template = """
            {% for message in messages %}
            {% if message['role'] == 'system' %}
            {{ '<|system|>\n' + message['content'] + eos_token + '\n' }}
            {% elif message['role'] == 'user' %}
            {{ '<|user|>\n' + message['content'] + eos_token + '\n' }}
            {% elif message['role'] == 'assistant' %}
            {{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}
            {% endif %}
            {% endfor %}
            """
        log_message("DEBUG", f"Chat template set: {tokenizer.chat_template}")
        
        # initial_token_count = len(tokenizer)
        # response_template = prompter.response_template 
        # added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
        # model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)
        
        # initial_token_count = len(tokenizer)
        # instruction_template = prompter.instruction_template 
        # added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [instruction_template]})
        # model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)
        
        # response_template_ids = tokenizer.encode("<|assistant|>", add_special_tokens=False)[1:]

        # collator = DataCollatorForCompletionOnlyLM(
        #     response_template=response_template_ids,
        #     tokenizer=tokenizer,
        # )
        
        # collator = DataCollatorForCompletionOnlyLM(response_template=prompter.response_template,
        #                                   tokenizer=tokenizer,
        #                                  mlm=False)
        collator = DataCollatorForCompletionOnlyLM(response_template=prompter.response_template,
                                          instruction_template=prompter.instruction_template,
                                          tokenizer=tokenizer,
                                         mlm=False)
        log_message("INFO", "Data collator for completion-only LM initialized successfully.")

      

        log_message("INFO", "Initializing the SFT Trainer...")
        trainer = SFTTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            dataset_text_field="prompt",
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            args=args,
            data_collator=collator
        )
        log_message("INFO", "SFT Trainer initialized successfully.")

        
        log_message("INFO", "--- Trainable Parameters Information ---")

       # Capture the output of print_trainable_parameters
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        trainer.model.print_trainable_parameters()
        sys.stdout = old_stdout

        # Log the captured output
        trainable_params_log = captured_output.getvalue()
        for line in trainable_params_log.splitlines():
         log_message("INFO", f"  {line}")

        log_message("INFO", "--- End of Trainable Parameters Information ---")

        log_message("INFO", "--- Training Started ---")
        # Capture the output of trainer.train()
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        trainer.train()
        sys.stdout = old_stdout

        # Log the captured training output
        training_log = captured_output.getvalue()
        for line in training_log.splitlines():
            log_message("INFO", f"  [Training Output] {line}")
        log_message("INFO", "--- Training Finished ---")

    except FileNotFoundError as e:      
        error_message = f"FileNotFoundError: {e}"
        log_message("error", error_message)
        print(error_message)
        print("Please ensure that the specified file paths are correct and the files exist.")
    except OSError as e:
        error_message = f"OSError: {e}"
        log_message("error", error_message)
        print(error_message)
        print("An operating system error occurred. Check file permissions or disk space.")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        log_message("error", error_message)
        print(error_message)
        print("Please review the error message and your training setup.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument("--model_name_or_path", type=str, default="akhooli/gpt2-small-arabic", help="Path to pretrained model or model identifier")
    parser.add_argument("--quantize", type=str, default="False", help="Whether to quantize the model")
    parser.add_argument("--data_path", type=str, default="D:/Code/LLMTraining-main/input_folder", help="Path to the training data (HuggingFace dataset format)")
    parser.add_argument("--max_seq_length", type=int, default=215, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training")
    parser.add_argument("--gradient_checkpointing", type=str, default="True", help="Whether to use gradient checkpointing")
    parser.add_argument("--output_dir", type=str, default="D:/Code/LLMTraining-main/output_folder", help="Directory to save the output")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--bf16", type=str, default="False", help="Whether to use bfloat16")
    parser.add_argument("--bf16_full_eval", type=str, default="False", help="Whether to use bfloat16 for full evaluation")
    parser.add_argument("--tf32", type=str, default="False", help="Whether to use tf32")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--deepspeed", type=str, default="D:/Code/LLMTraining-main/src/blm/config/deepspeed_zero3.json", help="Path to the DeepSpeed configuration file")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of save steps")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device for evaluation")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Number of evaluation accumulation steps")
    parser.add_argument("--loss_type", type=str, default=None, help="Type of loss to use") # Keep this for consistency
    parser.add_argument("--weight_decay", type=str, default=0.0001, help="Type of loss to use") # Keep this for consistency

    args = parser.parse_args()
    
    train(args)