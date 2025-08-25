from pathlib import Path
import re

def read_texts_from_dir(data_dir: str):
    texts, metas = [], []
    p = Path(data_dir)
    if not p.exists():
        return texts, metas
    for f in p.glob("**/*"):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            texts.append(txt)
            metas.append({"source": str(f.name)})
    return texts, metas

def simple_chunk(text: str, max_chars: int = 800, overlap: int = 120):
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, end)
    return [c for c in chunks if len(c) > 40]

def prepare_corpus_from_dir(data_dir: str):
    texts, metas = read_texts_from_dir(data_dir)
    all_chunks, all_meta = [], []
    for txt, m in zip(texts, metas):
        chs = simple_chunk(txt)
        for i, c in enumerate(chs):
            meta = dict(m)
            meta["chunk_id"] = i
            all_chunks.append(c)
            all_meta.append(meta)
    return all_chunks, all_meta
