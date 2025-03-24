import pandas as pd
import re

# Load the data
file_path = './bibli2.csv'
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Print the data size before cleaning
print(f"Data size before cleaning: {data.shape}")

# Columns to be cleaned
columns_to_clean = ['lcc','title', 'abstract','toc', 'lcsh_subject_headings']

# Check if the columns to be cleaned exist
for col in columns_to_clean:
    if col not in data.columns:
        print(f"Warning: Column {col} not found in the dataset.")

# Define a cleaning function: remove leading/trailing whitespace, convert to lowercase, remove unnecessary punctuation
def clean_text(text):
    if pd.isna(text):
        return text
    text = text.strip()  # Remove leading and trailing spaces
    text = re.sub(r'\.', '', text)  # Remove periods
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower()  # Convert to lowercase

# Apply cleaning function to the selected columns
for col in columns_to_clean:
    if col in data.columns:
        data[col] = data[col].apply(clean_text)

# Drop rows with missing values in the specified columns
data.dropna(subset=columns_to_clean, inplace=True)

# Remove duplicates based on the specified columns
data.drop_duplicates(subset=columns_to_clean, inplace=True)

# Keep only the specified columns
data = data[columns_to_clean]

# Check the data size after cleaning
print(f"Data size after cleaning: {data.shape}")

# Display the first few rows of the cleaned data
print(data.head())

# Save the cleaned data to a new CSV file, ensuring 'utf-8-sig' encoding to prevent encoding issues
data.to_csv('./cleaned_data_lcsh.csv', index=False, encoding='utf-8-sig')
