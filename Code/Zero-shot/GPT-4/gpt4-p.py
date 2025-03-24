from openai import OpenAI

client = OpenAI(api_key="Your API key here")
import pandas as pd
from datasets import Dataset

# Set OpenAI API key

# Load JSON dataset
data = Dataset.from_json("test.json")
data = data.select_columns(["title", "abstract", "lcsh_subject_headings"])

# Define base message template
base_messages = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

# Define single prediction function
def get_prediction(title, abstract):
    messages = base_messages + [
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}
    ]
    # Call the latest API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        temperature=0.6,
        top_p=0.9
    )
    return response.choices[0].message.content

# Define batch prediction function
def batch_predict(batch):
    predictions = []
    for title, abstract in zip(batch["title"], batch["abstract"]):
        prediction = get_prediction(title, abstract)
        predictions.append(prediction)
    return {"predicted_LCSH": predictions}

# Run batch inference
print("Starting batch prediction...")
data = data.map(batch_predict, batched=True, batch_size=16)

# Save results
data.to_pandas().to_csv("lcsh_gpt4-p.csv", index=False)
print("LCSH label prediction completed and saved to CSV file.")
