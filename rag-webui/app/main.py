from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads BASE_DIR, LLM_URL, LLM_MODEL, etc. from /Volumes/Data/Rag/.env

# --- Config ---
BASE_DIR = Path(os.getenv("BASE_DIR", "/Volumes/Data/Rag"))
INDEX_DIR = BASE_DIR / "index"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key")

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_PASSAGE_CHARS = int(os.getenv("MAX_PASSAGE_CHARS", "800"))

# --- Init services ---
client = chromadb.PersistentClient(path=str(INDEX_DIR), settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
llm = OpenAI(base_url=LLM_URL, api_key=LLM_API_KEY)

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond STRICTEMENT à partir des extraits fournis.\n"
    "Si l'information n'apparait pas dans les extraits, dis que tu ne sais pas.\n"
    "Réponds brièvement et cite les sources (nom de fichier + n° de chunk)."
)

app = FastAPI(title="RAG Perso (Local)")

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class AskBody(BaseModel):
    query: str

def build_context(query: str):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    q_emb = q_emb.tolist()  # <<< IMPORTANT
    results = collection.query(
        query_embeddings=[q_emb],   # liste de listes
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ctx_parts, citations = [], []
    for doc, meta in zip(docs, metas):
        snippet = (doc or "")[:MAX_PASSAGE_CHARS]
        ctx_parts.append(snippet)
        src = (meta or {}).get("source", "unknown")
        ch = (meta or {}).get("chunk", "0")
        citations.append(f"{Path(src).name}#chunk{ch}")
    context = "\n\n---\n\n".join(ctx_parts)
    return context, citations

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
def ask(body: AskBody):
    q = (body.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question vide.")
    context, citations = build_context(q)
    user_prompt = (
        f"Question: {q}\n\n"
        f"Extraits pertinents:\n{context}\n\n"
        "Consigne: Utilise uniquement ces extraits."
    )
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2,
        max_tokens=512,
    )
    ans = resp.choices[0].message.content.strip()
    return JSONResponse({"answer": ans, "sources": citations})