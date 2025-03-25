import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, set_seed

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Load and preprocess dataset
def build_dataset(model_name, dataset_base_path, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load JSON dataset
    ds = load_dataset("json", data_files={
        "train": os.path.join(dataset_base_path, "train.json"),
        "test": os.path.join(dataset_base_path, "test.json"),
    })

    # Format data as chat-style prompts
    def generate_chat_text(sample):
        base_messages = [
            {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for books."},
            {"role": "user", "content": "Given the title and abstract of a book, predict as many relevant LCSH labels as possible."},
            {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."},
            {"role": "user", "content": f"Title: {sample.get('title', 'N/A')} [SEP] Abstract: {sample.get('abstract', 'N/A')}"},
            {"role": "assistant", "content": sample.get('lcsh_subject_headings', 'N/A')}
        ]
        chat_text = "\n".join([f"<s>[{msg['role'].upper()}] {msg['content']}" for msg in base_messages])
        sample["chat_text"] = chat_text
        return sample

    ds = ds.map(generate_chat_text, batched=False)

    # Filter out samples that exceed the token limit
    def tokenize_filter(sample):
        tokens = tokenizer(sample["chat_text"], truncation=True, max_length=max_length, padding="max_length")
        return len(tokens["input_ids"]) <= max_length

    ds = ds.filter(tokenize_filter, batched=False)

    print("Cleaned dataset sizes:")
    print(f"Training set size: {ds['train'].num_rows}")
    print(f"Testing set size: {ds['test'].num_rows}")

    return ds

# Load model and apply LoRA
def load_model_with_lora(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("LoRA config:", peft_config)

    return model, tokenizer, peft_config

# Fine-tuning using SFTTrainer
def fine_tune_with_sfttrainer(model, tokenizer, train_dataset, eval_dataset, output_dir, max_length, peft_config):
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        warmup_ratio=0.2,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        save_total_limit=1,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb",  # Log to W&B if configured
    )

    # Tokenize and format dataset
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["chat_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": model_inputs["labels"].squeeze(0),
        }

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        peft_config=peft_config,
    )

    print("Starting fine-tuning with SFTTrainer...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Entry point
def main():
    max_length = 256
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_base_path = "./"
    output_dir = "./fine_tuned_model_with_lora_sft-e10"

    print("Loading dataset...")
    dataset = build_dataset(model_name, dataset_base_path, max_length)

    print("Loading model and tokenizer with LoRA...")
    model, tokenizer, peft_config = load_model_with_lora(model_name)

    print("Fine-tuning model with SFTTrainer...")
    fine_tune_with_sfttrainer(
        model,
        tokenizer,
        dataset["train"],
        dataset["test"],
        output_dir,
        max_length,
        peft_config=peft_config,
    )

if __name__ == "__main__":
    main()
