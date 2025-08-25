
### **Architecture & Flow (Markdown Diagram)**

```markdown
# RAG System Architecture


User Query
│
▼
[Embedder] -- computes embeddings --> Query vector
│
▼
[Chroma DB] -- vector search --> Top-K relevant chunks
│
▼
[Flan-T5] -- generates answer based on retrieved chunks
│
▼
User gets semantically aligned answer



**Optional Refinements:**
- Better embeddings (OpenAI / high-dim)
- More corpus data / augmentation
- Overlapping chunks
- Hybrid search (keywords + semantic)
- Fine-tuning LLM on domain data
