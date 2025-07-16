import math
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from datasets import load_dataset

def main():

    # Load the dataset in streaming mode
    dataset = load_dataset("ClusterlabAi/101_billion_arabic_words_dataset", split="train", streaming=True)

    # Take 1000 samples for training
    raw_train_data = [item["text"] for item in dataset.take(10000)]

    # Re-initialize streaming dataset (streaming iterators are one-time use)
    dataset = load_dataset("ClusterlabAi/101_billion_arabic_words_dataset", split="train", streaming=True)

    # Take next 200 samples for evaluation
    raw_eval_data = [item["text"] for item in dataset.skip(1000).take(2000)]

    # ----------------------------
    # Step 1: Load Model & Tokenizer
    # ----------------------------
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Step 2: Prepare Dummy Dataset
    # ----------------------------
    #raw_train_data = ["This is a sample text for training.", "Another training sample here."]
    #raw_eval_data = ["This is evaluation text one.", "And evaluation text two."]

    train_dataset_raw = Dataset.from_dict({"text": raw_train_data})
    eval_dataset_raw = Dataset.from_dict({"text": raw_eval_data})

    max_seq_length = 64

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    dataset = train_dataset_raw.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset_raw.map(tokenize_function, batched=True, remove_columns=["text"])

    dataset = dataset.map(lambda e: {"labels": e["input_ids"]}, batched=True)
    eval_dataset = eval_dataset.map(lambda e: {"labels": e["input_ids"]}, batched=True)

    # ----------------------------
    # Step 3: Define Metric
    # ----------------------------
    def compute_metrics(eval_pred):
        from torch.nn import CrossEntropyLoss

        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels)

        labels_tensor[labels_tensor == -100] = tokenizer.pad_token_id

        shift_logits = logits_tensor[..., :-1, :].contiguous()
        shift_labels = labels_tensor[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss.item())

        return {"perplexity": perplexity}

    # ----------------------------
    # Step 4: Training Arguments
    # ----------------------------
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,
        warmup_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="./test_outputs",
        report_to="none",
        eval_steps=5,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="perplexity",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ----------------------------
    # Step 5: Evaluate Before Training
    # ----------------------------
    print("Evaluating model BEFORE training...")
    before_results = trainer.evaluate()
    print(f"Pre-training Evaluation Results: {before_results}")

    if "eval_loss" in before_results and "perplexity" not in before_results:
        print("Pre-training Perplexity:", math.exp(before_results["eval_loss"]))
    elif "perplexity" in before_results:
        print("Pre-training Perplexity:", before_results["perplexity"])

    # ----------------------------
    # Step 6: Train
    # ----------------------------
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # ----------------------------
    # Step 7: Evaluate After Training
    # ----------------------------
    print("Evaluating model AFTER training...")
    after_results = trainer.evaluate()
    print(f"Post-training Evaluation Results: {after_results}")

    if "eval_loss" in after_results and "perplexity" not in after_results:
        print("Post-training Perplexity:", math.exp(after_results["eval_loss"]))
    elif "perplexity" in after_results:
        print("Post-training Perplexity:", after_results["perplexity"])


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
