# CPSC_8470
# Simple Retrieval-Augmented Generation (RAG) System

This is a simple implementation of a Retrieval-Augmented Generation (RAG) system using a smaller dataset. The project demonstrates how to build a retrieval system using FAISS and a generation system using a pre-trained language model. This repository includes the code to set up the environment, process the data, build the retrieval and generation systems, and evaluate the results.

## Table of Contents
- [Installation](#installation)
- [Data Processing](#data-processing)
- [Building the Retrieval System](#building-the-retrieval-system)
- [Building the Generation System](#building-the-generation-system)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/simple-rag.git
    cd simple-rag
    ```

2. **Set up a Conda environment**:
    ```sh
    conda create -n simple_rag python=3.9 --yes
    conda activate simple_rag
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Data Processing

1. **Load and process the SQuAD dataset**:
    ```python
    from datasets import load_dataset
    dataset = load_dataset('squad')

    import pandas as pd

    data = []
    for example in dataset['train']:
        data.append({
            'question': example['question'],
            'context': example['context'],
            'answer': example['answers']['text'][0]
        })
    df = pd.DataFrame(data)
    df.to_csv('squad_data.csv', index=False)
    ```

## Building the Retrieval System

1. **Create a FAISS index**:
    ```python
    import faiss
    import numpy as np

    embeddings = np.random.randn(len(df), 768).astype('float32')
    index = faiss.IndexFlatL2(768)
    index.add(embeddings)
    ```

2. **Define the retrieval function**:
    ```python
    def retrieve(query_embedding, k=5):
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        D, I = index.search(query_embedding, k)
        return I
    ```

## Building the Generation System

1. **Load a pre-trained language generation model**:
    ```python
    from transformers import pipeline
    generator = pipeline('text-generation', model='gpt2')
    ```

2. **Generate answers based on the retrieved context**:
    ```python
    def generate_answer(context, question):
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        result = generator(input_text, max_length=50)
        return result[0]['generated_text']
    ```

## Evaluation

1. **Evaluate the system and save generated answers**:
    ```python
    import json

    def evaluate_and_save(question, true_answer, k=5):
        query_embedding = np.random.randn(1, 768).astype('float32')
        retrieved_indices = retrieve(query_embedding, k)
        retrieved_contexts = df.iloc[retrieved_indices[0]]['context'].tolist()
        generated_answers = [generate_answer(context, question) for context in retrieved_contexts]
        
        with open('generated_answers.json', 'a') as f:
            for context, answer in zip(retrieved_contexts, generated_answers):
                f.write(json.dumps({
                    'question': question,
                    'context': context,
                    'generated_answer': answer
                }) + '\n')

        return any(true_answer in answer for answer in generated_answers)

    question = "What is AI?"
    true_answer = "AI is the simulation of human intelligence in machines."
    evaluate_and_save(question, true_answer, k=5)
    ```

## Usage

1. **Print the contents of `generated_answers.json`**:
    ```python
    import json

    with open('generated_answers.json', 'r') as f:
        for line in f:
            print(json.loads(line))
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
