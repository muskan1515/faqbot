import re
from rag_model import generate_answer_with_context

def query_expansion(gen_pipeline, query):
    prompt = f"""
Paraphrase the following question into 5 semantically similar questions.
Return them as a numbered list (1-5), one per line. Correct grammar if needed.
Question: {query}
"""
    out = gen_pipeline(prompt, max_new_tokens=128, do_sample=True, temperature=0.8, top_k=50, top_p=0.95, num_beams=5, num_return_sequences=5)
    paraphrases = [o['generated_text'] for o in out]
    return [query] + paraphrases

def semantic_search(query, embedder, collection, gen_pipeline, k=3):
    paraphrases = query_expansion(gen_pipeline, query)
    q_emb = embedder.encode(paraphrases)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k, include=["documents", "metadatas", "distances"])
    hits = [{"text": doc, "meta": meta, "score": float(dist)}
            for doc, meta, dist in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0])]
    return hits

def answer_query(query, embedder, collection, gen_pipeline, k=3):
    hits = semantic_search(query, embedder, collection, gen_pipeline, k)
    retrieved_texts = [h["text"] for h in hits]
    if not retrieved_texts:
        return {"question": query, "answer": "I don't have any documents to answer from.", "retrieved": []}
    answer = generate_answer_with_context(gen_pipeline, query, retrieved_texts)
    return {"question": query, "answer": answer, "retrieved": hits}
