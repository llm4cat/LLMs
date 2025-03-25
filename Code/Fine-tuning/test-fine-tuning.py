import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm  # Progress bar

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the fine-tuned model
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Clean text (remove newlines, trim spaces)
def clean_text(text):
    text = text.replace("\n", " ").strip()
    return text

# Construct input format and check length
def construct_input(title, abstract, tokenizer, max_length):
    """
    Construct model input from a single sample and check its length.
    """
    title = clean_text(title)
    abstract = clean_text(abstract)

    # Build the prompt text
    input_text = (
        f"<s>[SYSTEM] You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers.\n"
        f"[USER] Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible.\n"
        f"[USER] Respond only with the predicted LCSH labels separated by commas.\n"
        f"[USER] Title: {title} [SEP] Abstract: {abstract}\n"
        f"[ASSISTANT]"
    )

    # Check tokenized input length
    inputs = tokenizer(input_text, truncation=False, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])

    if input_length > max_length:
        return None
    return input_text

# Generate prediction
def generate_prediction(model, tokenizer, title, abstract, max_length=512):
    """
    Generate predictions using the model.
    """
    input_text = construct_input(title, abstract, tokenizer, max_length)
    if input_text is None:
        return "Input too long"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            temperature=0.6,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract predicted labels
    if "[ASSISTANT]" in generated_text:
        predicted_labels = generated_text.split("[ASSISTANT]")[-1].strip()
    else:
        predicted_labels = generated_text.strip()

    return predicted_labels

# Main process: prediction for all samples
def main():
    model_path = "./fine_tuned_model_with_lora_sft-e10"
    model, tokenizer = load_model(model_path)

    test_data_file = "./test.json"
    with open(test_data_file, "r") as f:
        data = json.load(f)  # assuming a list of dicts

    max_length = 512
    filtered_samples = []

    for sample in tqdm(data, desc="Filtering Samples", unit="sample"):
        title = sample.get("title", "N/A")
        abstract = sample.get("abstract", "N/A")
        input_text = construct_input(title, abstract, tokenizer, max_length)
        if input_text:
            filtered_samples.append(sample)

    print(f"Remaining samples after filtering: {len(filtered_samples)}")

    results = []
    for sample in tqdm(filtered_samples, desc="Processing Samples", unit="sample"):
        title = sample.get("title", "N/A")
        abstract = sample.get("abstract", "N/A")
        true_labels = sample.get("lcsh_subject_headings", "N/A")

        try:
            predicted_labels = generate_prediction(model, tokenizer, title, abstract)
            results.append({
                "title": title,
                "abstract": abstract,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels
            })
        except Exception as e:
            results.append({
                "title": title,
                "abstract": abstract,
                "true_labels": true_labels,
                "predicted_labels": "Error"
            })

    output_file = "./test_predictions_sft-e10-50.csv"
    pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
