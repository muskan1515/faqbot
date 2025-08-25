from semantic_search import answer_query

def start_chat(embedder, collection, gen_pipeline):
    print("Interactive chat started. Type 'exit' or 'quit' to stop.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        out = answer_query(q, embedder, collection, gen_pipeline, k=3)
        print("\nBot:", out["answer"])
        print("\nSources:")
        for s in out["retrieved"]:
            print("-", s["meta"].get("source", "unknown"), "(score:", round(s["score"], 4), ")")
