import os
import numpy as np
import pandas as pd
import re
from transformers import pipeline
from datasets import Dataset

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using device: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Load Llama 3.1 8B-Instruct model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": "bfloat16"},
    device=0,
    pad_token_id=0
)
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token  

# Load JSON data
test_data_path = "test-n.json"
if not os.path.exists(test_data_path):
    print(f"File {test_data_path} not found! Please check the path.")
    exit()

data = Dataset.from_json(test_data_path)

# Keep required columns
required_columns = {"title", "abstract", "Predicted_Label_Count", "lcsh_subject_headings"}
if not required_columns.issubset(set(data.column_names)):
    print(f"`test.json` must contain the columns: {required_columns}")
    exit()

data = data.select_columns(list(required_columns))

# Filter out empty or invalid rows
data = data.filter(lambda x: x["title"] and x["abstract"] and x["Predicted_Label_Count"] > 0)

if len(data) == 0:
    print("Dataset is empty. Please check if `test.json` has valid data.")
    exit()


def clean_lcsh_labels(label_str):
    """
    Clean LCSH predictions:
    - Remove content after `--` or `-`
    - Remove newlines, dots, etc.
    - Remove duplicates and sort
    """
    if isinstance(label_str, (list, np.ndarray)):
        label_str = ", ".join(map(str, label_str))

    if not isinstance(label_str, str) or label_str.strip() == "":
        return ""

    labels = label_str.split(",")  
    cleaned_labels = set()

    for label in labels:
        label = label.strip()
        label = re.sub(r"\s*--.*", "", label)
        label = re.sub(r"\s+-\s+.*", "", label)

        if label:
            cleaned_labels.add(label)

    return ", ".join(sorted(cleaned_labels))


def generate_lcsh_labels(title, abstract, n):
    """
    Step-by-step LCSH prediction:
    - Round 1: generate n labels
    - Round 2: generate 2n more labels
    - Round 3: generate as many additional labels as possible
    """
    assistant_responses = []

    def generate_step(messages):
        response = pipe(messages, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)

        if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
            generated_text = response[0]["generated_text"]

            if isinstance(generated_text, list):
                generated_text = " ".join(str(item) if isinstance(item, str) else str(item.get("content", "")) for item in generated_text)
            elif isinstance(generated_text, dict):
                generated_text = generated_text.get("content", "")

            generated_text = generated_text.strip()

            extracted_text = generated_text.split("Respond only with the predicted LCSH labels separated by commas.")[-1].strip()
            
            labels = [label.strip() for label in extracted_text.split(",") if label.strip()]
            labels = clean_lcsh_labels(labels)

            assistant_responses.append(labels)
        else:
            print("Warning: unexpected response format:", response)
            return ""

        return labels

    # Round 1
    assistant_first_round = generate_step([
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        {"role": "user", "content": f"Predict exactly {n} Library of Congress Subject Headings (LCSH) labels. Respond only with the predicted LCSH labels separated by commas."}
    ])

    # Round 2
    assistant_second_round = ""
    if assistant_first_round:
        assistant_second_round = generate_step([
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
            {"role": "user", "content": f"Current LCSH labels: {assistant_first_round}"},
            {"role": "user", "content": f"Predict {2*n} additional Library of Congress Subject Headings (LCSH) labels. Respond only with the predicted LCSH labels separated by commas."}
        ])

    # Round 3
    assistant_third_round = ""
    if assistant_second_round:
        assistant_third_round = generate_step([
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
            {"role": "user", "content": f"Current LCSH labels: {assistant_second_round}"},
            {"role": "user", "content": f"Predict as many additional Library of Congress Subject Headings (LCSH) labels as possible. Respond only with the predicted LCSH labels separated by commas."}
        ])

    # Merge all predictions
    all_labels = set(filter(None, [
        *assistant_first_round.split(", "),
        *assistant_second_round.split(", "),
        *assistant_third_round.split(", ")
    ]))

    return {
        "title": title,
        "abstract": abstract,
        "Predicted_Label_Count": n,
        "predicted_LCSH": clean_lcsh_labels(", ".join(all_labels))
    }

# Run batch prediction
print("Starting batch prediction (first 5 rows for testing)...")
data = data.map(lambda x: generate_lcsh_labels(x["title"], x["abstract"], x["Predicted_Label_Count"]), batched=False)

# Save results
df_results = pd.DataFrame(data)[["title", "abstract", "lcsh_subject_headings", "Predicted_Label_Count", "predicted_LCSH"]]
output_file = "lcsh_zero-shot-llama-cot-amsp.csv"
df_results.to_csv(output_file, encoding="utf-8-sig", index=False)

print(f"Results successfully saved to {output_file}" if os.path.exists(output_file) else "Failed to save the file. Please check the path or permissions.")
