from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedder(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
    embedder = SentenceTransformer(model_name)
    return embedder

def compute_embeddings(embedder, chunks, batch_size=64):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(embs)
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.zeros((0, embedder.get_sentence_embedding_dimension()))
    return embeddings
