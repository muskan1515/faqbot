# How to Improve RAG Response Quality

## 1. Embeddings
- Current: low-dimensional embeddings → less semantic alignment
- Upgrade: OpenAI `text-embedding-3-large` → captures deeper semantic meaning
- Hugging Face alternatives: `sentence-transformers/all-mpnet-base-v2`

```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_KEY")
emb = client.embeddings.create(input="text here", model="text-embedding-3-large")
```

### 2. Corpus Size

More data → better retrieval

#### Techniques:
Data scraping / internal docs
Paraphrasing / summarization
Synthetic augmentation

### 3. Chunking

Smaller chunks (100–300 tokens)
Overlap chunks by 50–100 tokens for context continuity

### 4. Search Strategy

Semantic search only → fails on rare terms
Use hybrid: semantic + exact keyword match

### 5. Generative Model

Flan-T5 fine-tuning on domain-specific QA improves answers
Temperature & top-k/top-p tuning:

```
gen_pipeline(text, max_length=200, temperature=0.7, top_k=50)
```

### 6. Evaluation & Iteration

Track low-confidence answers
Re-insert missed chunks
Recompute embeddings after data growth