#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Calmer les libs et éviter l'over-parallelisme
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# --- Config chemins & paramètres ---
BASE_DIR = Path(os.getenv("BASE_DIR", "/Volumes/Data/Rag"))
INDEX_DIR = BASE_DIR / "index"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_docs")

# Embeddings multilingues par défaut (FR/EN…)
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# LLM (API OpenAI-compatible : llama-server, ollama/openai-proxy, etc.)
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key")

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_PASSAGE_CHARS = int(os.getenv("MAX_PASSAGE_CHARS", "800"))

# --- Init services ---
client = chromadb.PersistentClient(
    path=str(INDEX_DIR),
    settings=Settings(allow_reset=False, anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

llm = OpenAI(base_url=LLM_URL, api_key=LLM_API_KEY)

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond STRICTEMENT à partir des extraits fournis.\n"
    "Si l'information n'apparait pas dans les extraits, dis que tu ne sais pas.\n"
    "Réponds brièvement et cite les sources (nom de fichier + n° de chunk/row)."
)

def build_context(query: str):
    # Embedding de la question -> liste de floats
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    ctx_parts = []
    citations = []
    for doc, meta in zip(docs, metas):
        snippet = (doc or "")[:MAX_PASSAGE_CHARS]
        ctx_parts.append(snippet)

        # Citation lisible: CSV -> row, sinon -> chunk
        source = (meta or {}).get("source", "unknown")
        row = (meta or {}).get("row_index")
        chunk = (meta or {}).get("chunk", 0)
        label = f"{Path(source).name}#row{row}" if row is not None else f"{Path(source).name}#chunk{chunk}"
        citations.append(label)

    context = "\n\n---\n\n".join(ctx_parts)
    return context, citations

def ask(query: str, debug: bool = False):
    context, cites = build_context(query)

    if debug:
        print("=== CONTEXTE ENVOYÉ AU LLM ===")
        print(context if context.strip() else "(vide)")
        print("=== SOURCES SÉLECTIONNÉES ===")
        print(cites)
        print("==============================", file=sys.stderr)

    user_prompt = (
        f"Question: {query}\n\n"
        f"Extraits pertinents:\n{context}\n\n"
        "Consigne: Utilise uniquement ces extraits."
    )

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    ans = resp.choices[0].message.content.strip()
    if cites:
        ans += "\n\nSources: " + ", ".join(cites)
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interroger l'index local (RAG).")
    parser.add_argument("question", nargs="+", help="Ta question")
    parser.add_argument("--debug", action="store_true", help="Afficher le contexte et les sources sélectionnées")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        print("Usage: python scripts/query.py \"ta question ici\"")
        sys.exit(1)

    print(ask(question, debug=args.debug))