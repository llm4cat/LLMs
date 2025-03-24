import os
import openai
import pandas as pd
from datasets import Dataset

# Set OpenAI API key
client = openai.OpenAI(api_key="Your API key here")  # New API requires creating a client

# Load dataset from JSON file
test_data_path = "test.json"
data = Dataset.from_json(test_data_path)
data = data.select_columns(["title", "abstract", "lcsh_subject_headings"])
data = data.filter(lambda x: x["title"] and x["abstract"] and x["lcsh_subject_headings"])

# Task instructions
base_messages = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

# Few-shot examples
few_shot_examples = [
    {"role": "user", "content": "Title: The Pentagon Papers: National Security Versus the Public's Right to Know\n"
                                "Abstract: Discusses the Supreme Court trial which resulted from the decision of the New York Times newspaper to publish secret government documents about the Vietnam War."},
    {"role": "assistant", "content": "National security, Freedom of the press, Vietnam War"},

    {"role": "user", "content": "Title: On the Water: Rowing, Yachting, Canoeing, and Lots, Lots, More\n"
                                "Abstract: Describes the sailing, rowing, and canoeing events of the Olympic Games and previews the athletic competition at the 2000 Summer Olympics in Sydney, Australia."},
    {"role": "assistant", "content": "Aquatic sports, Olympics"},

    {"role": "user", "content": "Title: The Andersonville Prison Civil War Crimes Trial: A Headline Court Case\n"
                                "Abstract: Examines the war crimes trial, in which Henry Wirz, the Confederate officer in charge of Andersonville prison camp was accused of allowing the prisoners to be deliberately abused and neglected."},
    {"role": "assistant", "content": "Andersonville Prison, Civil War, War Crimes, Trials, Henry Wirz, Confederate States of America, Prisoners of War, Human Rights Abuse"},
]

# Define batch prediction function
def batch_predict(batch):
    predictions = []
    for title, abstract in zip(batch["title"], batch["abstract"]):
        messages = base_messages + few_shot_examples + [
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        ]

        try:
            response = client.chat.completions.create(  # Compatible with new API
                model="gpt-4",
                messages=messages,
                max_tokens=150,
                temperature=0.6,
                top_p=0.9
            )

            predicted_text = response.choices[0].message.content.strip()
            predictions.append(predicted_text)

        except Exception as e:
            print(f"Error during prediction: {e}")
            predictions.append("Error")

    return {"predicted_LCSH": predictions}

print("Starting batch prediction...")
data = data.map(batch_predict, batched=True, batch_size=1)

# Convert to pandas DataFrame
df = pd.DataFrame({
    "Title": data["title"],
    "Abstract": data["abstract"],
    "True_LCSH": data["lcsh_subject_headings"],   # Ground-truth LCSH labels
    "Predicted_LCSH": data["predicted_LCSH"]       # Predicted LCSH labels
})

# Save as CSV file
output_file = "lcsh_gpt4_few3.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

if os.path.exists(output_file):
    print(f"Results successfully saved to {output_file}")
else:
    print("Failed to save the file. Please check the path or permissions.")
