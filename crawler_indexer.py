import requests, time, json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import numpy as np
from tqdm import tqdm
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import os

# ---------------- Config ----------------
ROOT = "https://example.com/docs" 
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DocCrawler/1.0)"}
MAX_PAGES = 200
MODEL_PATH = "models/sentence-transformer"
EMBED_MODEL_PATH = MODEL_PATH


# ---------------- NLTK setup ----------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ---------------- Load local model ----------------
EMBED_MODEL = SentenceTransformer(MODEL_PATH)
print("Loaded local embedding model from:", MODEL_PATH)

# ---------------- Crawler helpers ----------------
def is_internal(url):
    p = urlparse(url)
    return "example.com" in p.netloc

def normalize(base, link):
    return urljoin(base, link.split("#")[0])

def crawl(start):
    seen, q, pages = set(), deque([start]), []
    while q and len(pages) < MAX_PAGES:
        url = q.popleft()
        if url in seen: continue
        seen.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                print("skip", url, r.status_code)
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            pages.append((url, title, soup))
            for a in soup.find_all("a", href=True):
                full = normalize(url, a['href'])
                if is_internal(full) and full not in seen:
                    q.append(full)
            time.sleep(0.2)
        except Exception as e:
            print("error", url, e)
    return pages

def extract_chunks(url, title, soup, max_chars=1500, overlap=200):
    elems = soup.select("h1,h2,h3,h4,p,pre,code,li,td,th")
    parts, cur_heading, cur_text = [], title, []
    for el in elems:
        tag, text = el.name.lower(), el.get_text(separator=" ", strip=True)
        if not text: continue
        if tag.startswith("h"):
            if cur_text:
                parts.append({"heading": cur_heading, "text": " ".join(cur_text)})
                cur_text=[]
            cur_heading = text
        else:
            cur_text.append(text)
    if cur_text: parts.append({"heading": cur_heading, "text": " ".join(cur_text)})

    chunks=[]
    for part in parts:
        sents = sent_tokenize(part["text"])
        cur=""
        for s in sents:
            if len(cur)+len(s)+1 <= max_chars:
                cur = cur + " " + s if cur else s
            else:
                chunks.append({"url": url, "page_title": title, "heading": part["heading"], "text": cur.strip()})
                seed = cur[-overlap:] if overlap and len(cur) > overlap else ""
                cur = (seed + " " + s).strip()
        if cur: chunks.append({"url": url, "page_title": title, "heading": part["heading"], "text": cur.strip()})
    return chunks

# ---------------- Embedding helpers ----------------
def embed_texts(texts):
    print(f"Embedding {len(texts)} chunks locally...")
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

# ---------------- Build ----------------
def build(start=ROOT):
    print("Crawling documentation...")
    pages = crawl(start)
    print("Pages fetched:", len(pages))

    all_chunks=[]
    for url, title, soup in pages:
        ch = extract_chunks(url, title, soup)
        all_chunks.extend(ch)
    print("Total chunks:", len(all_chunks))

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # save chunks + embeddings
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    np.save("embeddings.npy", embeddings)

    # create FAISS index
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, "index.faiss")
    print("Saved chunks.json, embeddings.npy, index.faiss")

    return all_chunks, embeddings

if __name__ == "__main__":
    build()

