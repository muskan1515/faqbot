import chromadb
from chromadb.config import Settings

def init_chroma(persist_dir="./chroma_persist", collection_name="faq_collection"):
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
    return client, collection

def upsert_collection(collection, chunks, metas, embeddings):
    if len(chunks) > 0:
        ids = [f"doc_{i}" for i in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings.tolist())
