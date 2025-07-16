from blm.utils.log import log_message


class Prompter:
    def __init__(self, tokenizer):
        log_message("INFO", "Initializing Prompter.")
        log_message("DEBUG", f"Tokenizer name or path: {tokenizer.name_or_path}")

        templates = {
            "mistralai": {"instruction_template": "[INST]",
                          "response_template": "[/INST]"},
            #"meta-llama": {"instruction_template": "<|start_header_id|>system<|end_header_id|>", 
            #               "response_template": "<|start_header_id|>assistant<|end_header_id|>"},
            "meta-llama": {"instruction_template": "<|system|>", 
                           "response_template": "<|assistant|>"},
            "microsoft": {"instruction_template": "<|system|>", 
                           "response_template": "<|assistant|>"},
            "default": {"instruction_template": "###Instructions:\n\n",
                          "response_template": "###Assistant:\n\n"},
            "gpt": {"instruction_template": "<user>",
                    "response_template": "<assistant>"}
        }

        log_message("DEBUG", f"Available prompt templates: {templates}")
        
        self.tokenizer = tokenizer
        self.model_family = self.tokenizer.name_or_path.split("/")[0]
        log_message("INFO", f"Detected model family: {self.model_family}")

        self.instruction_template = templates.get(self.model_family, templates["gpt"])["instruction_template"]
        self.response_template = templates.get(self.model_family, templates["gpt"])["response_template"]
        log_message("INFO", f"Selected instruction template: '{self.instruction_template}'")
        log_message("INFO", f"Selected response template: '{self.response_template}'")
        log_message("INFO", "Prompter initialized successfully.")

    def __call__(self, data):
        """Prepare input for model training or inference.
        Pass data to generate prompts for training
        Pass system and user to generate one time prompt for a specific model
        based on the model ID in the tokenizer.

        Args:
            data (DatasetDict): dataset that should contains prompt 
                                components (system, instructions, data and output)
                                Use this option when generating training data
            system (str): system prompt
            user (str): user prompt
        """
        log_message("INFO", "Processing data to generate prompts for SFT...")
        log_message("DEBUG", f"Input data keys: {list(data.keys())}")
        if "messages" not in data:
            log_message("ERROR", "Key 'messages' not found in the input data. Unable to generate prompts.")
            return data  # Or raise an exception, depending on desired behavior
        log_message("INFO", f"Generating prompts for {len(data['messages'])} message sequences.")
        prompts = [self._for_sft(messages) for messages in data["messages"]]
        log_message("INFO", f"Generated {len(prompts)} prompts.")
        data["prompt"] = prompts
        #log_message("DEBUG", f"First few generated prompts: {prompts[:2] if prompts else []}")
        log_message("INFO", "Prompt generation for SFT complete. Adding 'prompt' key to data.")
        return data

    def _for_sft(self, messages):
        """
        Convert the list of messages into a prompt by injecting the LLM special
        instruction and assistant tokens.
        :param messages: List[Dict] - list of user and assistant messages
        :return: str - prompt used as input to model training
        """
        log_message("INFO", "Starting to create SFT prompt from messages.")
        #log_message("DEBUG", f"Input messages: {messages}")
        
        if self.tokenizer.chat_template is None:
            log_message("WARNING", "Tokenizer chat template is None. Using default template.")
            self.tokenizer.chat_template = """
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
            log_message("DEBUG", f"Default chat template set: {self.tokenizer.chat_template}")
        else:
            log_message("DEBUG", f"Using existing chat template: {self.tokenizer.chat_template}")

        log_message("INFO", "Applying chat template to messages.")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # log_message("DEBUG", f"Generated prompt: {prompt}")
        log_message("INFO", "SFT prompt creation complete.")
        
        return prompt