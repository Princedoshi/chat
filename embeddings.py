# embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load the CSV data using pandas, only necessary columns
df = pd.read_csv("salaries.csv", usecols=[
    'work_year', 'experience_level', 'employment_type', 'job_title',
    'salary', 'salary_currency', 'salary_in_usd', 'employee_residence',
    'remote_ratio', 'company_location', 'company_size'
])
df['combined_text'] = df.apply(lambda row: ' '.join(map(str, row.values)), axis=1)
texts = df['combined_text'].tolist()

# Initialize sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Batch processing for embeddings
def encode_batch(batch):
    return model.encode(batch, convert_to_tensor=False)

# Generate embeddings in batches
def generate_embeddings(texts, batch_size=100):
    n = len(texts)
    embeddings = []
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = encode_batch(batch)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings
embeddings = generate_embeddings(texts)

# Save embeddings and texts to disk
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((embeddings, texts), f)

# Print confirmation message
print("Embeddings and texts have been successfully saved to 'embeddings.pkl'")
