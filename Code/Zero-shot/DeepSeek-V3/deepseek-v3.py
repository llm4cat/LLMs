import openai
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Step 1: Set basic configuration for DeepSeek API
openai.api_base = "https://api.deepseek.com"  # Base URL for DeepSeek API
openai.api_key = "Your API key here"          # Replace with your DeepSeek API key

# Step 2: Load JSON data
data_path = "test.json"  # Replace with your JSON file path
data = Dataset.from_json(data_path).to_pandas()

# Ensure the data contains 'title' and 'abstract' columns
if "title" not in data.columns or "abstract" not in data.columns:
    raise ValueError("The dataset must contain 'title' and 'abstract' columns.")

# Step 3: Define single prediction function
def get_prediction(title, abstract, lcsh="N/A", model="deepseek-chat"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
        {"role": "user", "content": "Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible."},
        {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."},
        {"role": "user", "content": f"Title: {title} [SEP] Abstract: {abstract}"},
        {"role": "assistant", "content": lcsh}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=256,
            temperature=0.6,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error predicting for title: {title}, error: {e}")
        return ""

# Step 4: Run batch prediction
print("Starting batch prediction...")
all_predictions = []
for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing"):
    prediction = get_prediction(row["title"], row["abstract"])
    all_predictions.append(prediction)

# Add predictions to original dataframe
data['predicted_LCSH'] = all_predictions

# Drop unnecessary columns if present
columns_to_drop = ['lcc', 'toc', 'lcc_category']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Step 5: Save results to CSV file with utf-8-sig encoding
output_path = "deepseek_zero.csv"
data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Prediction completed. Results saved to {output_path}")
