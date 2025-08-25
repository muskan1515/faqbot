from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_flant5(model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    try:
        gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    except:
        gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return gen_pipeline

PROMPT_TEMPLATE = """ You are a helpful assistant. Use the provided documents as the main source of truth.  
If the documents do not directly answer, reply exactly: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer (clear and concise):
"""

def generate_answer_with_context(gen_pipeline, question, retrieved_texts, max_length=128):
    context = "\n\n".join(retrieved_texts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    out = gen_pipeline(prompt, max_length=max_length, do_sample=False)
    return out[0]["generated_text"].strip()
