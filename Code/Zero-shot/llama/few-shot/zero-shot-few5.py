import os
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import Dataset

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using device: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Load model and tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": "bfloat16"},
    device=0,
    pad_token_id=tokenizer.pad_token_id
)

# **Load dataset from JSON file**
test_data_path = "test.json"
data = Dataset.from_json(test_data_path)
data = data.select_columns(["title", "abstract", "lcsh_subject_headings"])
data = data.filter(lambda x: x["title"] and x["abstract"] and x["lcsh_subject_headings"])

# **Task instructions**
base_messages = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

# **Few-shot examples**
few_shot_examples = [
    {"role": "user", "content": "Title: The Pentagon Papers: National Security Versus the Public's Right to Know\n"
                                "Abstract: Discusses the Supreme Court trial which resulted from the decision of the New York Times newspaper to publish secret government documents about the Vietnam War."},
    {"role": "assistant", "content": "National security, Freedom of the press, Vietnam War"},

    {"role": "user", "content": "Title: On the Water: Rowing, Yachting, Canoeing, and Lots, Lots, More\n"
                                "Abstract: Describes the sailing, rowing, and canoeing events of the Olympic Games and previews the athletic competition at the 2000 Summer Olympics in Sydney, Australia."},
    {"role": "assistant", "content": "Aquatic sports, Olympics"},

    {"role": "user", "content": "Title: the andersonville prison civil war crimes trial : a headline court case\n"
                                "Abstract: examines the war crimes trial, in which henry wirz, the confederate officer in charge of andersonville prison camp was accused of allowing the prisoners to be deliberately abused and neglected."},
    {"role": "assistant", "content": "andersonville prison, civil war, war crimes, trials, wirz, henry, confederate states of america, prisoners of war, human rights abuse"},

    {"role": "user", "content": "Title: teen privacy rights : a hot issue\n"
                                "Abstract:examines all apects of teen privacy rights, from the history of this topic to how it came to be an issue also discusses the importance of knowing and exercising your rights."},
    {"role": "assistant", "content": "teenagers; privacy, right of"},

    {"role": "user", "content": "Title: egyptian mythology\n"
                                "Abstract: discusses various egyptian myths, including creation stories and histories of principal gods and goddesses, along with background information and discussion questions and answers."},
    {"role": "assistant", "content": "mythology, egyptian"},
]

# **Define batch prediction function**
def batch_predict(batch):
    inputs = [
        base_messages + few_shot_examples + [
            {"role": "user", "content": f"Title: {t}\nAbstract: {a}"},
            {"role": "assistant", "content": ""}
        ]
        for t, a in zip(batch["title"], batch["abstract"])
    ]

    try:
        outputs = pipe(
            inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            batch_size=1
        )

        predictions = []

        # **Double unpacking of outputs**
        for outer_list in outputs:
            for output in outer_list:
                generated_text = output["generated_text"]

                assistant_response = [
                    item["content"] for item in generated_text if item["role"] == "assistant"
                ]

                if assistant_response:
                    lcsh_text = assistant_response[-1]
                    lcsh_text = lcsh_text.replace("LCSH:", "").strip()
                    lcsh_text = lcsh_text.replace("Assistant:", "").replace("assistant", "").strip()
                    lcsh_text = lcsh_text.split("\n")[0]
                    predictions.append(lcsh_text)
                else:
                    predictions.append("Error")

        return {"predicted_LCSH": predictions}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"predicted_LCSH": ["Error"] * len(batch["title"])}

print("Starting batch prediction...")
data = data.map(batch_predict, batched=True, batch_size=1)

# **Convert to pandas DataFrame**
df = pd.DataFrame({
    "Title": data["title"],
    "Abstract": data["abstract"],
    "True_LCSH": data["lcsh_subject_headings"],  # Ground-truth LCSH labels
    "Predicted_LCSH": data["predicted_LCSH"]      # Predicted LCSH labels
})

# **Save to CSV file**
output_file = "lcsh_llama-few5.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

if os.path.exists(output_file):
    print(f"Results successfully saved to {output_file}")
else:
    print("Failed to save the file. Please check the path or file permissions.")
