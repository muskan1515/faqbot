# RAG QA / Chat Project

This project implements a **Retrieval-Augmented Generation (RAG)** system with:  
- **Chroma DB** for vector storage  
- **Embeddings** for semantic search  
- **Flan-T5** for generative answers  
- **Interactive chat** interface

---

## Project Structure

├── data/ # Text files for corpus
├── src/
│ ├── corpus.py # Prepares chunks from text data
│ ├── embedding.py # Loads embedder & computes embeddings
│ ├── database.py # Initializes Chroma DB and upserts embeddings
│ ├── rag_model.py # Loads Flan-T5 generation pipeline
│ ├── semantic_search.py # Search & generate answer
│ └── chat.py # Interactive chat interface
├── main.py # Entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml


---

## Setup Instructions

### 1. Python Environment
```bash
pip install -r requirements.txt
```
### 2. Run Locally

```
python main.py
```

### 3. Using Docker (Single Container)

```
docker build -t rag-app .
docker run -it --rm rag-app
```

## Improving Response Quality

If your data is low or embeddings are low-dimension, response quality may drop 80–90%. Recommended improvements:

### Better Embeddings
Switch to OpenAI embeddings: text-embedding-3-small or text-embedding-3-large

### Improves semantic alignment

### Data Augmentation
Add more text samples
Use paraphrasing or summarization to expand low-volume corpus

### Chunking & Overlap
Smaller chunks + overlap (50–100 tokens) improve context retention

### Hybrid Search
Combine semantic + keyword search for rare queries

### Fine-tuning
Fine-tune Flan-T5 on your domain-specific QA data


--------------------------------

## Interactive Chat

Once embeddings and model are loaded, run:
```
start_chat(embedder, collection, gen_pipeline)
```

### Ask Queries like
```
How do I reset my password?
What is the refund policy?
```

## References

### Chroma DB
### Hugging Face Transformers
### Flan-T5 Model