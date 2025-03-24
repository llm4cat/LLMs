import os
import pandas as pd
import re
from transformers import pipeline
from datasets import Dataset

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using device: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Load LLaMA 3.1 model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": "bfloat16"},
    device=0,
    trust_remote_code=True
)
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

# Load JSON dataset
test_data_path = "test-n.json"
data = Dataset.from_json(test_data_path)

# Check if required columns exist
required_columns = ["title", "abstract", "Predicted_Label_Count"]
for col in required_columns:
    if col not in data.column_names:
        raise ValueError(f"Missing required column: {col}")

# **Batch prediction function**
def batch_predict(batch):
    inputs = [
        [
            {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
            {"role": "user", "content": (
                f"Given the title and abstract of a scientific paper, generate exactly {3*label_count} LCSH labels.\n\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n\n"
                "Respond only with the predicted LCSH labels, separated by commas."
            )}
        ]
        for title, abstract, label_count in zip(batch["title"], batch["abstract"], batch["Predicted_Label_Count"])
    ]

    try:
        # Perform batch inference
        outputs = pipe(
            inputs,
            max_new_tokens=100,  # Limit max output length
            do_sample=True,
            temperature=0.3,     # Lower temperature for more deterministic output
            top_p=0.8,
            batch_size=1
        )

        # Parse the output
        predictions = []

        # Double-unpack the nested outputs
        for outer_list in outputs:
            for output in outer_list:
                generated_text = output["generated_text"]

                assistant_response = [
                    item["content"] for item in generated_text if item["role"] == "assistant"
                ]
                if assistant_response:
                    predictions.append(assistant_response[-1])
                else:
                    predictions.append("Error")

        return {"predicted_LCSH": predictions}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"predicted_LCSH": ["Error"] * len(batch["title"])}

#  Start batch inference
print("Starting batch prediction (first 5 entries)...")
data = data.map(batch_predict, batched=True, batch_size=5)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# **Clean predicted labels**
def clean_lcsh_labels(label_str):
    if pd.isna(label_str):  # Handle missing values
        return ""
    
    labels = label_str.split(",")  # Split by comma

    cleaned_labels = []
    for label in labels:
        label = label.strip()
        # Remove content after `--` or ` - `, but keep internal hyphens (e.g., "rock-and-roll")
        label = re.sub(r"\s*--.*", "", label)       # Remove after "--"
        label = re.sub(r"\s+-\s+.*", "", label)     # Remove after " - " with spaces

        cleaned_labels.append(label)

    return ", ".join(cleaned_labels)

# Apply cleaning function
df["predicted_LCSH"] = df["predicted_LCSH"].apply(clean_lcsh_labels)

# Save to CSV file, keeping only selected columns
output_file = "lcsh_llama-3n.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

# Check if file was saved successfully
if os.path.exists(output_file):
    print(f" Prediction and cleaning complete! Results saved to {output_file}")
else:
    print(" Failed to save the file. Please check the path or permissions.")
