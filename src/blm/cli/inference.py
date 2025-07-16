import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API Token")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    
    args = parser.parse_args()
    
    hf_token = args.token  # Use the token provided in the argument
    
    if not hf_token:
        raise ValueError("Hugging Face token is required")

    # Define the chat template
    SYSTEMPROMPT = (
        "You are an AI assistant, trained to compare sentences and detect common sense. "
        "Based on the provided instructions, you will be given two sentences, each starting with an index (0 and 1). "
        "Your task is to determine which sentence reflects common sense or if both are equally plausible. "
        "You should focus on the meaning and logical coherence of the sentences, and ensure that the comparison is fair, objective, and neutral, "
        "without any bias towards specific perspectives or individuals."
    )

    formatted_instructions = (
        "Given the following two sentences, each sentence starts with an index:\n"
        "0 ميغان رابينو تمتلك قوى خارقة تجعلها تلعب كرة القدم على القمر وتحقق أهدافًا من خلال جاذبيته المنخفضة.\n"
        "1 ميغان رابينو هي لاعبة كرة قدم أمريكية بارزة، قادت منتخب الولايات المتحدة للفوز بكأس العالم للسيدات 2019 وحصلت على جائزة أفضل لاعبة.\n\n"
        "Which sentence has common sense, 0 or 1?"
    )

    # Input data for the chat
    input_data = [
        {"content": SYSTEMPROMPT, "role": "system"},
        {"content": formatted_instructions, "role": "user"},
        {"content": "", "role": "assistant"}  # Assistant's response will be generated
    ]

    # Initialize the tokenizer and model
    model_path = args.model_path  # Get the model path from arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Using the model name for tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # Manually format the chat input for generation
    chat_text = ""
    for message in input_data:
        chat_text += f"{message['role']}: {message['content']}\n"

    # Create the text generation pipeline
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)

    # Generate the response
    response = generator(chat_text, max_length=500, num_return_sequences=1, return_full_text=True)

    # Output the result
    print(response)


if __name__ == "__main__":
    main()
