import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load data
file_path = 'your_data_here.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Load sentence embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Label parsing function (supports multiple separators)
def parse_labels(label_string):
    if pd.isna(label_string):
        return []
    labels = label_string.replace(';', ',').split(',')
    return [label.strip().lower() for label in labels if label.strip()]

# Discounted Cumulative Gain
def dcg(relevance_scores):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

# Metric calculation for each row
def calculate_metrics(row):
    true_labels = set(parse_labels(row['lcsh_subject_headings']))
    predicted_labels = set(parse_labels(row['predicted_LCSH']))

    if not true_labels or not predicted_labels:
        return pd.Series({
            'recall': 0.00,
            'precision': 0.00,
            'ndcg': 0.00,
            'semantic_similarity': 0.00,
            'num_predicted_labels': 0.00
        })

    matched_count = len(true_labels.intersection(predicted_labels))
    recall = round(matched_count / len(true_labels), 2)
    precision = round(matched_count / len(predicted_labels), 2) if predicted_labels else 0.00

    relevance_scores = [1 if label in true_labels else 0 for label in predicted_labels]
    dcg_score = dcg(relevance_scores)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg_score = dcg(ideal_relevance_scores)
    ndcg = round(dcg_score / idcg_score, 2) if idcg_score > 0 else 0.00

    true_embeddings = model.encode(list(true_labels), convert_to_tensor=True)
    pred_embeddings = model.encode(list(predicted_labels), convert_to_tensor=True)
    similarity_matrix = util.cos_sim(pred_embeddings, true_embeddings)
    max_similarities = similarity_matrix.max(dim=1)[0]
    avg_similarity = round(max_similarities.mean().item(), 2) if len(max_similarities) > 0 else 0.00

    return pd.Series({
        'recall': recall,
        'precision': precision,
        'ndcg': ndcg,
        'semantic_similarity': avg_similarity,
        'num_predicted_labels': round(len(predicted_labels), 2)
    })

# Apply metric calculation
data[['recall', 'precision', 'ndcg', 'semantic_similarity', 'num_predicted_labels']] = data.apply(calculate_metrics, axis=1)

# Compute mean and std
metrics = ['recall', 'precision', 'ndcg', 'semantic_similarity', 'num_predicted_labels']
results = {metric: {'mean': round(data[metric].mean(), 2), 'std': round(data[metric].std(), 2)} for metric in metrics}

# Save to CSV with formatted values
output_path = 'metrics_results_lcsh_book.csv'
data.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.2f')

# Print summary
print(f"File saved as: {output_path}")
for metric, values in results.items():
    print(f"{metric.capitalize()} Mean: {values['mean']}, Standard Deviation: {values['std']:.2f}")
