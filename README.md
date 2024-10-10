# ExamCheetah

ExamCheetah is a Python-based RAG (Retrieval-Augmented Generation) application designed to answer questions from presentations and scientific papers. The application leverages document embeddings and a language model to retrieve relevant document chunks and generate responses.

## Features
- Load PDF documents and split them into chunks for easier processing.
- Store document embeddings in a Chroma vector database.
- Query the vector database using a prompt and receive contextually accurate answers.
- Validate responses against correct answers in a dataset.

## Project Structure
- `src/core.py`: Defines the embedding function using the Ollama model.
- `src/docs_pipeline.py`: Handles PDF loading, document chunking, and metadata creation.
- `src/rag_pipeline.py`: Manages the RAG query process, retrieves chunks, and generates responses.
- `src/vector_db.py`: Handles adding document chunks to the Chroma vector store.
- `src/validation.py`: Validates generated answers against a dataset.
- `src/cheetah.ipynb`: Jupyter Notebook for setting up the vector database and querying the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ExamCheetah.git
   ```
