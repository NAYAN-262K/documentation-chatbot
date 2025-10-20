import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
INDEX_PATH = "data/index.faiss"
DOCSTORE_PATH = "data/chunks.json"
MODEL_PATH = "models/sentence-transformer"

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    import json
    doc_texts = json.load(f)

# Load embedding model
embedder = SentenceTransformer(MODEL_PATH)

def get_top_chunks(query, k=3):
    """Return top k most relevant chunks for a query"""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype="float32"), k)
    return [doc_texts[i] for i in I[0]]

def answer_question(query):
    """Generate answer using only documentation context"""
    chunks = get_top_chunks(query)
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    # Placeholder LLM response
    answer = f"[LLM Placeholder] Answer based on context:\n{context[:500]}..."
    return answer

if __name__ == "__main__":
    while True:
        query = input("Ask a question about the documentation (or 'exit'): ")
        if query.lower() == "exit":
            break
        print("\n" + answer_question(query) + "\n" + "-"*80)
