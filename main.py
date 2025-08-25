# in main.py or notebook
from src.corpus import prepare_corpus_from_dir
from src.embedding import load_embedder, compute_embeddings
from src.database import init_chroma, upsert_collection
from src.rag_model import load_flant5
from src.semantic_search import answer_query
from src.chat import start_chat

def main():
    # Prepare corpus
    chunks, metas = prepare_corpus_from_dir("./data")

    # Load embedder & compute embeddings
    embedder = load_embedder()
    embeddings = compute_embeddings(embedder, chunks)

    # Setup Chroma
    client, collection = init_chroma()
    upsert_collection(collection, chunks, metas, embeddings)

    # Load generative model
    gen_pipeline = load_flant5()

    # Test a query
    out = answer_query("How can I reset my password?", embedder, collection, gen_pipeline)
    print(out)

    # Start interactive chat
    start_chat(embedder, collection, gen_pipeline)

if __name__ == "__main__":
    main()
