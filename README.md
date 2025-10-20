# Documentation Chatbot

A chatbot that can answer technical questions from **any website's documentation** using embeddings.
This project crawls documentation pages, chunks text into sections, generates embeddings with a Sentence Transformer, and serves answers via a **Streamlit** web app.

---

## Features

* Crawl documentation pages automatically.
* Chunk text into meaningful sections for better retrieval.
* Generate embeddings using [Sentence Transformers](https://www.sbert.net/).
* Store embeddings in **FAISS** for fast similarity search.
* Ask questions and get answers via a **Streamlit** chatbot interface.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NAYAN-262K/documentation-chatbot.git
cd documentation-chatbot
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download a Sentence Transformer model

For example, `all-MiniLM-L6-v2` from Hugging Face:

1. [Model page](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
2. Download files and place them in a folder:

```
models/all-MiniLM-L6-v2/
```

3. Update `MODEL_PATH` in `crawler_indexer.py` and `chat_server.py` to point to your model folder.

### 5. Build embeddings

```bash
python crawler_indexer.py
```

This will create:

* `chunks.json`
* `embeddings.npy`
* `index.faiss`

Store these in the data folder(create one)

### 6. Run the chatbot locally

```bash
streamlit run app.py
```

Open the URL shown in your terminal to interact with the chatbot.

---

## Notes

The chatbot currently answers questions using retrieved documentation chunks and embeddings.

You can integrate any local or cloud LLM in chat_server.py if you want to enhance responses.

The project is system-independent and works with any website documentation.

You can change the ROOT URL in crawler_indexer.py to point to any website documentation you want to crawl.
