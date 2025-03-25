import json
import os
import pickle
import faiss
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


#1. Embedding model class
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_vector(self, label):
        return self.model.encode(label, convert_to_numpy=True)


#2. Load LCSH embeddings
def load_embeddings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    return data["labels"], embeddings


#3. Clean predicted labels in CSV
def clean_label(label):
    """
    Clean a single predicted label from CSV:
    - Remove content after "--" or em dash
    - Trim whitespace
    """
    cleaned_label = re.split(r'--|—', label)[0]
    return cleaned_label.strip()


def clean_csv_predictions(df, column_name="predicted_LCSH"):
    """
    Clean the predicted_LCSH column in the CSV:
    - Clean each label
    - Remove duplicates
    """
    print("Cleaning predicted labels in CSV...")
    df[column_name] = df[column_name].apply(
        lambda x: ";".join(set(clean_label(label) for label in str(x).split(";"))) if isinstance(x, str) else x
    )
    return df


#4. Replace predicted labels with LCSH vocabulary 
def replace_and_save_predictions(df, predicted_labels, lcsh_labels, lcsh_vectors, embeddings_model, pca, save_path):
    """
    Replace predicted labels using FAISS nearest neighbor search
    with the most similar LCSH labels from the controlled vocabulary.
    """
    index = faiss.IndexFlatL2(lcsh_vectors.shape[1])
    index.add(lcsh_vectors)

    replaced_predictions = []

    for pred_label_set in tqdm(predicted_labels, desc="Replacing predictions"):
        replaced_set = []
        for pred_label in pred_label_set:
            if pred_label in lcsh_labels:
                replaced_set.append(pred_label)
            else:
                try:
                    # Step 1: Get embedding vector for the predicted label
                    pred_vector = embeddings_model.get_vector(pred_label).astype('float32')

                    # Step 2: Check dimension consistency with PCA
                    if pred_vector.shape[0] != pca.components_.shape[1]:
                        print(f"Warning: {pred_label} dimension {pred_vector.shape[0]} != PCA dimension {pca.components_.shape[1]}. Skipping...")
                        continue

                    # Step 3: Apply PCA to reduce dimensions
                    pred_vector_reduced = pca.transform(pred_vector.reshape(1, -1)).flatten()

                    # Step 4: Search nearest LCSH label using FAISS
                    _, indices = index.search(np.array([pred_vector_reduced], dtype=np.float32), 1)
                    most_similar_label = lcsh_labels[indices[0][0]]
                    replaced_set.append(most_similar_label)
                except Exception as e:
                    print(f"Error processing {pred_label}: {e}")
                    replaced_set.append(pred_label)

        replaced_predictions.append(",".join(replaced_set))

    df["Replaced_Predictions"] = replaced_predictions
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Replaced predictions saved to {save_path}")


#5. Main function 
if __name__ == "__main__":
    reduced_embedding_path = "lcsh_embeddings_reduced.json"
    pca_model_path = "pca_model.pkl"
    csv_file = "test_predictions_sft-e10-50.csv"
    pred_col = "predicted_labels"
    replaced_csv_path = "replaced_predictions_with_finetune_sft-e10-50.csv"

    if not os.path.exists(reduced_embedding_path) or not os.path.exists(pca_model_path):
        raise FileNotFoundError("LCSH embeddings or PCA model not found.")

    print("Loading LCSH embeddings...")
    lcsh_labels, lcsh_vectors = load_embeddings(reduced_embedding_path)

    print("Loading PCA model...")
    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    print("Loading input CSV...")
    df = pd.read_csv(csv_file)

    # Clean predicted labels from CSV
    df = clean_csv_predictions(df, pred_col)

    print("Initializing Sentence Transformer model...")
    embeddings_model = EmbeddingModel("all-MiniLM-L6-v2")

    print("Replacing cleaned labels with nearest LCSH match...")
    predicted_labels = df[pred_col].apply(lambda x: str(x).split(";") if isinstance(x, str) else []).tolist()

    replace_and_save_predictions(df, predicted_labels, lcsh_labels, lcsh_vectors, embeddings_model, pca, replaced_csv_path)
