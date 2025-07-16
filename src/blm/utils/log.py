# log.py

import os
import datetime
from venv import logger

# Define the log file name and folder structure (moved inside the function)
log_folder = "/rep/msalahat/LLMTraining/log"
backup_folder = "back"
log_file_name = "log.log"  # Define a base log file name

argument_info = {
    "output_dir": {"description": "The output directory where the model predictions and checkpoints will be written.", "classification": "Basic Setup", "caption": "Output Directory"},
    "overwrite_output_dir": {"description": "Overwrite the content of the output directory.", "classification": "Basic Setup", "caption": "Overwrite Output Directory"},
    "seed": {"description": "Random seed for initialization.", "classification": "Basic Setup", "caption": "Random Seed"},
    "full_determinism": {"description": "Whether to make the training fully deterministic.", "classification": "Basic Setup", "caption": "Full Determinism"},
    "do_train": {"description": "Whether to run training.", "classification": "Basic Setup", "caption": "Run Training"},
    "do_eval": {"description": "Whether to run evaluation on the dev set.", "classification": "Basic Setup", "caption": "Run Evaluation"},
    "do_predict": {"description": "Whether to run predictions on the test set.", "classification": "Basic Setup", "caption": "Run Prediction"},
    "eval_strategy": {"description": "The evaluation strategy to adopt during training.", "classification": "Basic Setup", "caption": "Evaluation Strategy"},
    "eval_steps": {"description": "Run evaluation every X steps.", "classification": "Basic Setup", "caption": "Evaluation Steps"},
    "load_best_model_at_end": {"description": "Whether to load the best model found during training at the end.", "classification": "Basic Setup", "caption": "Load Best Model At End"},
    "metric_for_best_model": {"description": "The metric to use to compare models.", "classification": "Basic Setup", "caption": "Metric For Best Model"},
    "greater_is_better": {"description": "Whether the `metric_for_best_model` should be maximized or minimized.", "classification": "Basic Setup", "caption": "Greater Is Better"},

    "num_train_epochs": {"description": "Total number of training epochs to perform.", "classification": "Training Parameters", "caption": "Number of Training Epochs"},
    "learning_rate": {"description": "The initial learning rate for AdamW optimizer.", "classification": "Training Parameters", "caption": "Learning Rate"},
    "optim": {"description": "The optimizer to use.", "classification": "Training Parameters", "caption": "Optimizer"},
    "weight_decay": {"description": "Weight decay for AdamW if applied.", "classification": "Training Parameters", "caption": "Weight Decay"},
    "max_grad_norm": {"description": "Maximum gradient norm (for gradient clipping).", "classification": "Training Parameters", "caption": "Max Gradient Norm"},
    "warmup_ratio": {"description": "Linear warmup over warmup_ratio fraction of total steps.", "classification": "Training Parameters", "caption": "Warmup Ratio"},
    "warmup_steps": {"description": "Linear warmup over warmup_steps.", "classification": "Training Parameters", "caption": "Warmup Steps"},
    "gradient_accumulation_steps": {"description": "Number of updates steps to accumulate before performing a backward/update pass.", "classification": "Training Parameters", "caption": "Gradient Accumulation Steps"},
    "gradient_checkpointing": {"description": "If True, use gradient checkpointing to save memory.", "classification": "Training Parameters", "caption": "Gradient Checkpointing"},
    "max_steps": {"description": "If set to a positive value, overrides num_train_epochs.", "classification": "Training Parameters", "caption": "Max Steps"},
    "label_smoothing_factor": {"description": "The label smoothing epsilon to apply (zero means no label smoothing).", "classification": "Training Parameters", "caption": "Label Smoothing Factor"},
    "group_by_length": {"description": "Whether or not to group samples of roughly the same length together when forming batches.", "classification": "Training Parameters", "caption": "Group By Length"},

    "per_device_train_batch_size": {"description": "The batch size per GPU/TPU core/CPU for training.", "classification": "Batch Sizes", "caption": "Train Batch Size (Per Device)"},
    "per_device_eval_batch_size": {"description": "The batch size per GPU/TPU core/CPU for evaluation.", "classification": "Batch Sizes", "caption": "Eval Batch Size (Per Device)"},
    "eval_accumulation_steps": {"description": "Number of evaluation steps to accumulate predictions for before moving to the next example.", "classification": "Batch Sizes", "caption": "Eval Accumulation Steps"},
    "auto_find_batch_size": {"description": "Whether to try to automatically find a batch size that will fit into memory.", "classification": "Batch Sizes", "caption": "Auto Find Batch Size"},

    "_n_gpu": {"description": "Number of GPUs available.", "classification": "Hardware & Distributed Training", "caption": "Number of GPUs"},
    "no_cuda": {"description": "Whether to avoid using CUDA even if it is available.", "classification": "Hardware & Distributed Training", "caption": "No CUDA"},
    "fp16": {"description": "Whether to use 16-bit (mixed) precision training (through AMP).", "classification": "Hardware & Distributed Training", "caption": "FP16 Training"},
    "bf16": {"description": "Whether to use bf16 (mixed) precision training.", "classification": "Hardware & Distributed Training", "caption": "BF16 Training"},
    "deepspeed": {"description": "Enable deepspeed and pass the path to the deepspeed json config file.", "classification": "Hardware & Distributed Training", "caption": "DeepSpeed Config"},
    "fsdp": {"description": "Enable fully sharded data parallelism (FSDP).", "classification": "Hardware & Distributed Training", "caption": "FSDP"},
    "ddp_backend": {"description": "The distributed backend to use.", "classification": "Hardware & Distributed Training", "caption": "DDP Backend"},
    "local_rank": {"description": "Rank of the process during distributed training.", "classification": "Hardware & Distributed Training", "caption": "Local Rank"},
    "use_cpu": {"description": "Whether to force CPU even if GPUs/TPUs are available.", "classification": "Hardware & Distributed Training", "caption": "Use CPU"},
    "use_ipex": {"description": "Use Intel PyTorch Extensions (IPEX) for training.", "classification": "Hardware & Distributed Training", "caption": "Use IPEX"},
    "use_mps_device": {"description": "Enable MPS (Metal Performance Shaders) training on Apple Silicon devices.", "classification": "Hardware & Distributed Training", "caption": "Use MPS Device"},
    "torch_compile": {"description": "Whether to use torch.compile() for compilation.", "classification": "Hardware & Distributed Training", "caption": "Torch Compile"},
    "tpu_num_cores": {"description": "When training on TPU, the number of TPU cores (automatically inferred by default).", "classification": "Hardware & Distributed Training", "caption": "TPU Num Cores"},

    "logging_dir": {"description": "The directory for storing logs.", "classification": "Logging & Saving", "caption": "Logging Directory"},
    "logging_steps": {"description": "Log every X updates steps.", "classification": "Logging & Saving", "caption": "Logging Steps"},
    "save_steps": {"description": "Save checkpoint every X updates steps.", "classification": "Logging & Saving", "caption": "Save Steps"},
    "save_strategy": {"description": "The saving strategy to adopt during training.", "classification": "Logging & Saving", "caption": "Save Strategy"},
    "save_total_limit": {"description": "Limit the total amount of checkpoints.", "classification": "Logging & Saving", "caption": "Save Total Limit"},
    "save_only_model": {"description": "If True, only the model will be saved.", "classification": "Logging & Saving", "caption": "Save Only Model"},
    "save_safetensors": {"description": "If True, the model will be saved using the safetensors format.", "classification": "Logging & Saving", "caption": "Save Safetensors"},
    "push_to_hub": {"description": "Whether to push the model to Hub.", "classification": "Logging & Saving", "caption": "Push To Hub"},
    "hub_model_id": {"description": "The name of the repository to keep in sync with the local one.", "classification": "Logging & Saving", "caption": "Hub Model ID"},
    "report_to": {"description": "The list of integrations to report the results and logs to.", "classification": "Logging & Saving", "caption": "Report To"},
    "disable_tqdm": {"description": "Whether or not to disable the tqdm progress bars.", "classification": "Logging & Saving", "caption": "Disable TQDM"},
    "logging_first_step": {"description": "Log the first global_step.", "classification": "Logging & Saving", "caption": "Log First Step"},
    "logging_nan_inf_filter": {"description": "Filter nan and inf losses for logging.", "classification": "Logging & Saving", "caption": "Filter NaN/Inf Losses"},

    "data_seed": {"description": "Random seed to be used with data samplers.", "classification": "Data Loading", "caption": "Data Seed"},
    "dataloader_num_workers": {"description": "Number of subprocesses to use for data loading.", "classification": "Data Loading", "caption": "Dataloader Workers"},
    "dataloader_pin_memory": {"description": "Whether to pin memory on GPU accelerators.", "classification": "Data Loading", "caption": "Pin Dataloader Memory"},
    "dataloader_drop_last": {"description": "Drop the last incomplete batch if it is smaller than the others.", "classification": "Data Loading", "caption": "Drop Last Batch"},
    "accelerator_config": {"description": "Accelerator config launched in distributed training.", "classification": "Hardware & Distributed Training", "caption": "Accelerator Config"},
    "adafactor": {"description": "Whether to use the adafactor optimizer.", "classification": "Training Parameters", "caption": "Use Adafactor"},
    "adam_beta1": {"description": "Beta1 for AdamW optimizer.", "classification": "Training Parameters", "caption": "Adam Beta1"},
    "adam_beta2": {"description": "Beta2 for AdamW optimizer.", "classification": "Training Parameters", "caption": "Adam Beta2"},
    "adam_epsilon": {"description": "Epsilon for AdamW optimizer.", "classification": "Training Parameters", "caption": "Adam Epsilon"},
    "average_tokens_across_devices": {"description": "If set, will average the number of tokens across all devices when logging training speed.", "classification": "Logging & Saving", "caption": "Average Tokens Across Devices"},
    "batch_eval_metrics": {"description": "Whether to also log the information of the single batches.", "classification": "Logging & Saving", "caption": "Log Batch Eval Metrics"},
    "bf16_full_eval": {"description": "Whether to run evaluation in bf16 even if training isn't in bf16.", "classification": "Hardware & Distributed Training", "caption": "BF16 Full Eval"},
    "dataloader_persistent_workers": {"description": "Whether to use persistent workers in dataloaders.", "classification": "Data Loading", "caption": "Persistent Dataloader Workers"},
    "dataloader_prefetch_factor": {"description": "Number of samples loaded in advance by each worker.", "classification": "Data Loading", "caption": "Dataloader Prefetch Factor"},
    "ddp_broadcast_buffers": {"description": "When using DDP, whether to broadcast buffers at the beginning of training.", "classification": "Hardware & Distributed Training", "caption": "DDP Broadcast Buffers"},
    "ddp_bucket_cap_mb": {"description": "When using DDP, the size of the bucketing used for reduction.", "classification": "Hardware & Distributed Training", "caption": "DDP Bucket Cap MB"},
    "ddp_find_unused_parameters": {"description": "When using DDP, whether to find unused parameters.", "classification": "Hardware & Distributed Training", "caption": "DDP Find Unused Parameters"},
    "ddp_timeout": {"description": "Timeout for DDP training in seconds.", "classification": "Hardware & Distributed Training", "caption": "DDP Timeout"},
    "debug": {"description": "List of debugging flags.", "classification": "Debugging", "caption": "Debug Flags"},
    "dispatch_batches": {"description": "Whether to dispatch batches to devices as they become free.", "classification": "Hardware & Distributed Training", "caption": "Dispatch Batches"},
    "eval_delay": {"description": "Number of steps to wait before the first evaluation.", "classification": "Basic Setup", "caption": "Evaluation Delay"},
    "eval_do_concat_batches": {"description": "Whether to concatenate batches for evaluation.", "classification": "Basic Setup", "caption": "Concat Eval Batches"},
    "eval_on_start": {"description": "Run evaluation on the first step.", "classification": "Basic Setup", "caption": "Evaluate On Start"},
    "eval_use_gather_object": {"description": "Whether to use gather_object in distributed evaluation.", "classification": "Hardware & Distributed Training", "caption": "Eval Use Gather Object"},
    "evaluation_strategy": {"description": "The evaluation strategy to use.", "classification": "Basic Setup", "caption": "Evaluation Strategy"},
    "fp16_backend": {"description": "The backend to use for fp16 training.", "classification": "Hardware & Distributed Training", "caption": "FP16 Backend"},
    "fp16_full_eval": {"description": "Whether to run evaluation in fp16 even if training isn't in fp16.", "classification": "Hardware & Distributed Training", "caption": "FP16 Full Eval"},
    "fp16_opt_level": {"description": "The optimization level to use for fp16.", "classification": "Hardware & Distributed Training", "caption": "FP16 Opt Level"},
    "fsdp_config": {"description": "Configuration for FSDP.", "classification": "Hardware & Distributed Training", "caption": "FSDP Config"},
    "fsdp_min_num_params": {"description": "Minimal number of parameters for a layer to be wrapped with FSDP.", "classification": "Hardware & Distributed Training", "caption": "FSDP Min Num Params"},
    "fsdp_transformer_layer_cls_to_wrap": {"description": "The Transformer layer class to wrap in FSDP.", "classification": "Hardware & Distributed Training", "caption": "FSDP Transformer Layer To Wrap"},
    "gradient_checkpointing_kwargs": {"description": "Keyword arguments for gradient checkpointing.", "classification": "Training Parameters", "caption": "Gradient Checkpointing Kwargs"},
    "half_precision_backend": {"description": "The backend to use for half-precision training.", "classification": "Hardware & Distributed Training", "caption": "Half Precision Backend"},
    "hub_always_push": {"description": "Whether to push the model to Hub at the end of every training.", "classification": "Logging & Saving", "caption": "Hub Always Push"},
    "hub_private_repo": {"description": "Whether the model repo on Hub should be private.", "classification": "Logging & Saving", "caption": "Hub Private Repo"},
    "hub_strategy": {"description": "The Hub strategy to use.", "classification": "Logging & Saving", "caption": "Hub Strategy"},
    "hub_token": {"description": "The token to use when pushing to Hub.", "classification": "Logging & Saving", "caption": "Hub Token", "sensitive": True},
    "ignore_data_skip": {"description": "When resuming training, whether to skip the epochs and steps to the last saving.", "classification": "Training Parameters", "caption": "Ignore Data Skip"},
    "include_for_metrics": {"description": "List of arguments to include in the metric logs.", "classification": "Logging & Saving", "caption": "Include For Metrics"},
    "include_inputs_for_metrics": {"description": "Whether to include inputs in the metric logs.", "classification": "Logging & Saving", "caption": "Include Inputs For Metrics"},
    "include_num_input_tokens_seen": {"description": "Whether to include the number of input tokens seen in the logs.", "classification": "Logging & Saving", "caption": "Include Num Input Tokens Seen"},
    "include_tokens_per_second": {"description": "Whether to include tokens per second in the logs.", "classification": "Logging & Saving", "caption": "Include Tokens Per Second"},
    "jit_mode_eval": {"description": "Whether to use JIT mode for evaluation.", "classification": "Hardware & Distributed Training", "caption": "JIT Mode Eval"},
    "label_names": {"description": "The list of keys in your dictionary of inputs that correspond to the labels.", "classification": "Data Loading", "caption": "Label Names"},
    "length_column_name": {"description": "Name of the column containing the lengths of the texts.", "classification": "Data Loading", "caption": "Length Column Name"},
    "log_level": {"description": "Verbosity level for the main process.", "classification": "Logging & Saving", "caption": "Log Level"},
    "log_level_replica": {"description": "Verbosity level for the replica processes.", "classification": "Logging & Saving", "caption": "Log Level Replica"},
    "log_on_each_node": {"description": "When doing a multi-node distributed training, whether to log once per node or only on the main node.", "classification": "Hardware & Distributed Training", "caption": "Log On Each Node"},
    "logging_strategy": {"description": "The logging strategy to adopt during training.", "classification": "Logging & Saving", "caption": "Logging Strategy"},
    "lr_scheduler_kwargs": {
    "description": "Keyword arguments for the learning rate scheduler.",
    "classification": "Training Parameters",
    "caption": "LR Scheduler Arguments"
  },
  "lr_scheduler_type": {
    "description": "The scheduler type to use.",
    "classification": "Training Parameters",
    "caption": "LR Scheduler Type"
  },
  "mp_parameters": {
    "description": "Parameters passed along to the multi-processing launching script.",
    "classification": "Hardware & Distributed Training",
    "caption": "Multi-processing Parameters"
  },
  "neftune_noise_alpha": {
    "description": "The noise alpha parameter for NEFTune.",
    "classification": "Training Parameters",
    "caption": "NEFTune Noise Alpha"
  },
  "optim_args": {
    "description": "Keyword arguments for the optimizer.",
    "classification": "Training Parameters",
    "caption": "Optimizer Arguments"
  },
  "optim_target_modules": {
    "description": "The modules to apply the optimizer on.",
    "classification": "Training Parameters",
    "caption": "Optimizer Target Modules"
  },
  "past_index": {
    "description": "The index of the key to use to represent the past.",
    "classification": "Model Configuration",
    "caption": "Past Index"
  },
  "prediction_loss_only": {
    "description": "When performing evaluation and predictions, only returns the loss.",
    "classification": "Basic Setup",
    "caption": "Prediction Loss Only"
  },
  "push_to_hub_model_id": {
    "description": "The name of the repository to create/update when pushing to hub.",
    "classification": "Logging & Saving",
    "caption": "Push to Hub Model ID"
  },
  "push_to_hub_organization": {
    "description": "The name of the organization in which to create/update the model repo.",
    "classification": "Logging & Saving",
    "caption": "Push to Hub Organization"
  },
  "push_to_hub_token": {
    "description": "The token to use when pushing to Hub.",
    "classification": "Logging & Saving",
    "caption": "Push to Hub Token"
  },
  "ray_scope": {
    "description": "The scope to use when running on Ray.",
    "classification": "Hardware & Distributed Training",
    "caption": "Ray Scope"
  },
  "remove_unused_columns": {
    "description": "Remove columns not required by the model when using a HuggingFace dataset.",
    "classification": "Data Loading",
    "caption": "Remove Unused Columns"
  },
  "restore_callback_states_from_checkpoint": {
    "description": "Whether to restore the states of the callbacks when resuming training from a checkpoint.",
    "classification": "Logging & Saving",
    "caption": "Restore Callback States"
  },
  "resume_from_checkpoint": {
    "description": "The path to a folder containing a previously saved checkpoint.",
    "classification": "Training Parameters",
    "caption": "Resume from Checkpoint"
  },
  "run_name": {
    "description": "A descriptor for the run.",
    "classification": "Logging & Saving",
    "caption": "Run Name"
  },
  "save_on_each_node": {
    "description": "When doing a multi-node distributed training, whether to save checkpoints on each node or only on the main node.",
    "classification": "Hardware & Distributed Training",
    "caption": "Save on Each Node"
  },
  "skip_memory_metrics": {
    "description": "Whether to skip adding of memory usage metrics.",
    "classification": "Logging & Saving",
    "caption": "Skip Memory Metrics"
  },
  "split_batches": {
    "description": "Whether to split batches among devices.",
    "classification": "Hardware & Distributed Training",
    "caption": "Split Batches"
  },
  "tf32": {
    "description": "Whether to enable the use of TensorFloat32 on supported GPUs.",
    "classification": "Hardware & Distributed Training",
    "caption": "Enable TF32"
  },
  "torch_compile_backend": {
    "description": "The backend to use for torch.compile().",
    "classification": "Hardware & Distributed Training",
    "caption": "Torch Compile Backend"
  },
  "torch_compile_mode": {
    "description": "The mode to use for torch.compile().",
    "classification": "Hardware & Distributed Training",
    "caption": "Torch Compile Mode"
  },
  "torch_empty_cache_steps": {
    "description": "Number of steps to perform a torch.cuda.empty_cache() during training.",
    "classification": "Hardware & Distributed Training",
    "caption": "Torch Empty Cache Steps"
  },
  "torchdynamo": {
    "description": "The torchdynamo mode to use.",
    "classification": "Hardware & Distributed Training",
    "caption": "TorchDynamo Mode"
  },
  "tpu_metrics_debug": {
    "description": "Whether to output debug metrics on TPU.",
    "classification": "Hardware & Distributed Training",
    "caption": "TPU Metrics Debug"
  },
  "use_legacy_prediction_loop": {
    "description": "Whether to use the legacy prediction loop.",
    "classification": "Basic Setup",
    "caption": "Use Legacy Prediction Loop"
  }
}

def log_args(args):
    """Logs training arguments, grouped by classification."""
    log_message("INFO", "--- Training Arguments ---")

    classified_arguments = {}
    other_arguments = {}

    # Separate arguments by classification
    for key, value in vars(args).items():
        if key in argument_info:
            classification = argument_info[key]["classification"]
            if classification not in classified_arguments:
                classified_arguments[classification] = {}
            classified_arguments[classification][key] = value
        else:
            other_arguments[key] = value

    # Print arguments by classification
    for classification, args_in_class in classified_arguments.items():
        log_message("INFO", f"  --- {classification} ---")
        for key, value in args_in_class.items():
            description = argument_info[key]["description"]
            caption=argument_info[key]["caption"]
            log_message("INFO", f"   {caption}-({key}): {value} - {description}")

    # Print other arguments
    if other_arguments:
        log_message("INFO", "  --- Other Arguments ---")
        for key, value in other_arguments.items():
            log_message("INFO", f"    {key}: {value}")


def backup_log_and_create_new():
    """
    Backs up the log file by renaming it with a datetime timestamp
    and creates a new empty file with the original name.
    """
    # Create the log and backup folders if they don't exist
    os.makedirs(os.path.join(log_folder), exist_ok=True)
    os.makedirs(os.path.join(log_folder,backup_folder), exist_ok=True)
    

    log_file_path = os.path.join(log_folder, log_file_name)
    print(f"Starting training. Logs will be written to: {log_file_path}")

    if os.path.exists(log_file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(os.path.basename(log_file_name))
        backup_file_name = f"{base_name}_{timestamp}{ext}"
        backup_file_path = os.path.join(log_folder,backup_folder, backup_file_name)
        try:
            os.rename(log_file_path, backup_file_path)
            print(f"Backed up previous log file to: {backup_file_path}")
        except OSError as e:
            print(f"Error backing up log file: {e}")
            return  # Exit if backup fails

    # Create a new empty log file
    try:
        with open(log_file_path, 'w') as f:
            pass  # Creates an empty file
        print(f"Created a new log file at: {log_file_path}")
    except IOError as e:
        print(f"Error creating a new log file: {e}")

def log_message(level, message):
    log_file_path = os.path.join(log_folder, log_file_name)
    if log_file_path:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"{timestamp} , {message}\n"
        try:
            with open(log_file_path, "a",encoding="utf-8") as f:
              
                f.write(formatted_message)
        except OSError as e:
            print(f"Error writing to log file: {e}")
    else:
        print("Warning: Log file path not initialized.")

    

if __name__ == "__main__":
   print("INFO: Log  initialized.")