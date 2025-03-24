import os
from transformers import pipeline
from datasets import Dataset

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using device: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Load model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": "bfloat16"},
    device=0,
    pad_token_id=0
)
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token  

# Load dataset from JSON file
test_data_path = "test.json"
data = Dataset.from_json(test_data_path)
data = data.select_columns(["title", "abstract", "lcsh_subject_headings"])  
data = data.filter(lambda x: x["title"] and x["abstract"] and x["lcsh_subject_headings"])  

# Define base messages (system prompt)
base_messages = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a book, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

# Define batch prediction function
def batch_predict(batch):
    inputs = [
        base_messages + [{"role": "user", "content": f"Title: {t}\nAbstract: {a}"}]
        for t, a in zip(batch["title"], batch["abstract"])
    ]
    try:
        # Run inference using the model
        outputs = pipe(
            inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            batch_size=1
        )

        predictions = []

        # Unpack nested output structure
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

# Run batch inference
print("Starting batch prediction...")
data = data.map(batch_predict, batched=True, batch_size=5)

# Save results to CSV file
output_file = "lcsh_zero-shot.csv"
data.to_csv(output_file)
if os.path.exists(output_file):
    print(f"Results successfully saved to {output_file}")
else:
    print("Failed to save the file. Please check path or permissions.")
