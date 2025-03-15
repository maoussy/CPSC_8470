import faiss
import numpy as np
import pandas as pd

# Sample DataFrame with random data for demonstration
data = {
    'question': ['What is AI?', 'What is machine learning?', 'Explain deep learning.', 'What is a neural network?', 'What is data science?'],
    'context': ['AI is the simulation of human intelligence in machines.', 
                'Machine learning is a subset of AI that involves the use of algorithms to learn from data.',
                'Deep learning is a subset of machine learning that uses neural networks with many layers.',
                'A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data.',
                'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge from data.']
}
df = pd.DataFrame(data)

# For simplicity, we'll use random embeddings for the context
embeddings = np.random.randn(len(df), 768).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(768)  # L2 distance (Euclidean distance)

# Add embeddings to the index
index.add(embeddings)

# Function to retrieve top-k documents for a given query
def retrieve(query_embedding, k=5):
    query_embedding = query_embedding.reshape(1, -1).astype('float32')  # Reshape and ensure the correct dtype
    D, I = index.search(query_embedding, k)  # Search the index
    return I

# Example usage
query_embedding = np.random.randn(768).astype('float32')  # Random query embedding for demonstration
top_k_indices = retrieve(query_embedding, k=5)
print(top_k_indices)