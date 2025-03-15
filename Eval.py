import faiss
import numpy as np
import pandas as pd
import json
from transformers import pipeline

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

# Load a pre-trained language generation model
generator = pipeline('text-generation', model='gpt2')

# Generate an answer based on the retrieved context
def generate_answer(context, question):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    result = generator(input_text, max_length=50)
    return result[0]['generated_text']

# Function to evaluate the system and save generated answers
def evaluate_and_save(question, true_answer, k=5):
    query_embedding = np.random.randn(1, 768).astype('float32')  # Random query embedding for simplicity
    retrieved_indices = retrieve(query_embedding, k)
    retrieved_contexts = df.iloc[retrieved_indices[0]]['context'].tolist()
    generated_answers = [generate_answer(context, question) for context in retrieved_contexts]
    
    # Save the generated answers to a file
    with open('generated_answers.json', 'a') as f:
        for context, answer in zip(retrieved_contexts, generated_answers):
            f.write(json.dumps({
                'question': question,
                'context': context,
                'generated_answer': answer
            }) + '\n')

    # Check if the true answer is in any of the generated answers
    return any(true_answer in answer for answer in generated_answers)

# Example usage
question = "What is AI?"
true_answer = "AI is the simulation of human intelligence in machines."
evaluate_and_save(question, true_answer, k=5)