import argparse
from transformers import AutoTokenizer
from blm.utils.prompter import Prompter
from datasets import load_from_disk
from tqdm import tqdm
import os

def calculate_total_tokens(tokenizer_name: str, dataset_path: str, prompt_column: str = 'prompt') -> int:
    """
    Calculates the total number of tokens for all samples in a specified dataset.

    Args:
        tokenizer_name (str): The name or path of the pre-trained tokenizer (e.g., "meta-llama/Llama-3.2-1B-Instruct").
        dataset_path (str): The file path to the Hugging Face dataset stored on disk.
        prompt_column (str, optional): The name of the column in the dataset that contains
                                       the text to be tokenized. Defaults to 'prompt'.

    Returns:
        int: The total number of tokens across all samples in the dataset.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"ERROR: Could not load tokenizer '{tokenizer_name}': {e}")
        return 0

    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully with {len(dataset)} samples.")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at: {dataset_path}")
        return 0
    except Exception as e:
        print(f"ERROR: Error loading dataset: {e}")
        return 0

    total_tokens = 0
    print(f"Calculating tokens for '{prompt_column}' column in the dataset...")
    prompter = Prompter(tokenizer)
    dataset = dataset.map(prompter, batched=True)
    # Iterate through each sample and tokenize the text in the specified column
    for i, example in enumerate(tqdm(dataset, desc="Counting tokens")):
        if prompt_column not in example:
            print(f"WARNING: Column '{prompt_column}' not found in sample {i}. Skipping.")
            continue
        
        # Get the text from the specified column
        text_to_tokenize = example[prompt_column]
        
        # Tokenize the text. `return_tensors="pt"` is not strictly needed for counting,
        # but it's good practice if you were to use it with a model.
        # We access the first element of 'input_ids' because `tokenizer()` can return
        # a batch, even for a single input.
        inputs = tokenizer(text_to_tokenize, return_tensors="pt")
        
        # Add the number of tokens (length of input_ids) to the total
        total_tokens += inputs['input_ids'].shape[1] # .shape[1] gets the sequence length

    print(f"\nTotal tokens calculated: {total_tokens}")
    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total tokens in a dataset.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name or path of the tokenizer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Hugging Face dataset directory.")
    parser.add_argument("--prompt_column", type=str, default="prompt", 
                        help="Name of the column in the dataset containing the text to tokenize (default: 'prompt').")

    args = parser.parse_args()
    
    # Example usage:
    # Make sure you have a dataset saved locally, for instance:
    # from datasets import load_dataset
    # dataset = load_dataset("some_huggingface_dataset", split="train")
    # dataset.save_to_disk("./my_local_dataset")
    # Then run:
    # python your_script_name.py --tokenizer_name "meta-llama/Llama-3.2-1B-Instruct" --dataset_path "./my_local_dataset"

    calculated_tokens = calculate_total_tokens(
        tokenizer_name=args.tokenizer_name,
        dataset_path=args.dataset_path,
        prompt_column=args.prompt_column
    )
    print(f"Final Result: The dataset contains a total of {calculated_tokens} tokens.")

