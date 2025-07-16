import io
import logging
import sys
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import transformers
from  blm.utils.log import log_message 
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def create_and_prepare_model(args):
    log_message("INFO", f"Loading configuration from: {args.model_name_or_path}")
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=args.token)
    log_message("INFO", "Configuration loaded successfully.")
    quantization_config = None

    if args.quantize:
        log_message("INFO", "Loading quantization configuration for 4-bit training.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        log_message("INFO", "Quantization configuration loaded.")
    else:
        log_message("INFO", "Quantization is disabled.")

    log_message("INFO", f"Loading pre-trained model from: {args.model_name_or_path}")
    
    # Capture standard output
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        token=args.token,
        device_map=None if args.deepspeed else 'auto'
    )
    
     # Restore standard output
    sys.stdout = old_stdout

    # Log the captured output
    console_output = captured_output.getvalue()
    if console_output:
        log_message("DEBUG", "--- Console Output during model loading ---")
        for line in console_output.splitlines():
            log_message("DEBUG", f"[Console] {line}")
        log_message("DEBUG", "--- End of Console Output ---")
    
    log_message("INFO", "Pre-trained model loaded successfully.")

    log_message("INFO", "Finding all linear layer names for Lora.")
    target_modules = find_all_linear_names(model)
    if len(target_modules) == 0:
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    log_message("INFO", f"Found target modules for Lora: {target_modules}")

    log_message("INFO", "Creating Lora configuration.")
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    log_message("INFO", "Lora configuration created.")

    log_message("INFO", "Pre-processing model for PEFT (upcasting layer norms and embeddings).")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)
    log_message("INFO", "Model pre-processing for PEFT complete.")

    log_message("INFO", "Preparing model for k-bit training.")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    log_message("INFO", "Model prepared for k-bit training.")

    if args.gradient_checkpointing:
        log_message("INFO", "Enabling gradient checkpointing.")
        model.gradient_checkpointing_enable()
        log_message("INFO", "Gradient checkpointing enabled.")
    else:
        log_message("INFO", "Gradient checkpointing is disabled.")

    log_message("INFO", "Getting PEFT model.")
    model = get_peft_model(model, peft_config)
    log_message("INFO", "PEFT model created.")

    log_message("INFO", "Printing trainable parameters of the PEFT model:")
    model.print_trainable_parameters()
    log_message("INFO", "Trainable parameters information printed.")

    log_message("INFO", f"Loading tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    log_message("INFO", "Tokenizer loaded.")
    tokenizer.pad_token = tokenizer.eos_token
    log_message("INFO", f"Setting pad_token to eos_token: {tokenizer.pad_token}")

    log_message("INFO", "create_and_prepare_model function completed.")
    return model, peft_config, tokenizer


def find_all_linear_names(model):
    log_message("INFO", "Starting to find all linear module names.")
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    log_message("INFO", f"Initial list of potential Lora modules: {list(lora_module_names)}")

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
        log_message("INFO", "Removed 'lm_head' from Lora module names.")

    result = list(lora_module_names)
    log_message("INFO", f"Final list of Lora module names: {result}")
    return result