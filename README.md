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
- [Model saved] #model saved on google drive
- [Future work](#Future work)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/maoussy/simple-rag.git
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
```py
   python squad.py 
   ```

## Building the Retrieval System

   ```py 
   python Retrieval.py
   ```

## Building the Generation System
    ```py
    python Generation.py
     ```

## Evaluation
```py
 python Eval.py 
 ```
 
## Usage
```py
   python print_answers.py
```

## model saved on google drive

https://drive.google.com/drive/folders/1ePZUt1iQ5atJJuxX9nldCgeNFisqWvat?usp=sharing

## Future work
on the rerport we will mention our future work for Phase 3
