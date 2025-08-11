#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingestion locale (RAG) :
- Lit TXT / MD / PDF / CSV depuis BASE_DIR/docs
- Crée des embeddings (multilingues) et les stocke dans Chroma (BASE_DIR/index)
- Déplace chaque fichier traité vers BASE_DIR/ingested
"""

import os, glob, re, csv, hashlib
from pathlib import Path
from typing import List, Iterable, Dict, Any, Optional
from dotenv import load_dotenv

# Limiter les threads pour éviter le OOM / sur-parallélisme
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # force fallback CPU si MPS instable

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(os.getenv("BASE_DIR", "/Volumes/Data/Rag"))
DOCS = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"
INGESTED_DIR = BASE_DIR / "ingested"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_docs")

# Modèle d'embeddings (multilingue conseillé)
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# CSV options (env modifiables)
CSV_MAX_ROWS = int(os.getenv("CSV_MAX_ROWS", "100000"))            # limite dure
CSV_TRUNCATE_CHARS = int(os.getenv("CSV_TRUNCATE_CHARS", "200"))   # tronque champs très longs
CSV_SAMPLE_EVERY = int(os.getenv("CSV_SAMPLE_EVERY", "1"))         # 1=toutes les lignes, 10=1/10e…

# --- Lectures fichiers ---
def read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")

def read_md(path: Path) -> str:
    return path.read_text(errors="ignore")

def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        texts = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(texts)
    except Exception as e:
        print(f"[erreur lecture PDF] {path.name}: {e}")
        return ""

# --- Utilitaires texte ---
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    # Sécurités pour éviter les boucles infinies
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)  # 20% par défaut
    n = len(text)
    if n <= chunk_size:
        return [text]  # un seul chunk
    step = max(1, chunk_size - overlap)
    chunks, start = [], 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks

def doc_id_for(path: Path) -> str:
    h = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return f"{path.name}:{h}"

# --- CSV helpers ---
def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|",":"])
        return dialect.delimiter
    except Exception:
        if ";" in sample: return ";"
        if "\t" in sample: return "\t"
        return ","

def truncate(s: Any, limit: int) -> str:
    t = str(s)
    return (t[:limit] + "…") if len(t) > limit else t

def iter_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
        f.seek(0)
        delim = sniff_delimiter(head)
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            yield row

def csv_row_to_text(row: Dict[str, Any], truncate_chars: int) -> str:
    parts = []
    for k, v in row.items():
        k_s = (k or "").strip()
        v_s = truncate("" if v is None else v, truncate_chars).strip()
        parts.append(f'{k_s}="{v_s}"')
    return "; ".join(parts)

# --- Déplacement post-traitement ---
def move_to_ingested(path: Path):
    """Déplace le fichier traité vers BASE_DIR/ingested"""
    try:
        INGESTED_DIR.mkdir(parents=True, exist_ok=True)
        target = INGESTED_DIR / path.name
        path.rename(target)
        print(f"[ok] {path.name} déplacé vers {target}")
    except Exception as e:
        print(f"[warn] Impossible de déplacer {path.name}: {e}")

# --- Init Chroma + Embeddings ---
client = chromadb.PersistentClient(
    path=str(INDEX_DIR),
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
print(f"[ok] Embedding model chargé: {EMBED_MODEL_NAME}")

# --- Ingesteurs ---
def add_csv_document(path: Path) -> bool:
    rows = []
    try:
        for i, row in enumerate(iter_csv_rows(path)):
            if i % CSV_SAMPLE_EVERY != 0:
                continue
            rows.append((i, row))
            if len(rows) >= CSV_MAX_ROWS:
                print(f"[info] {path.name}: atteint CSV_MAX_ROWS={CSV_MAX_ROWS}, arrêt lecture.")
                break
    except Exception as e:
        print(f"[skip] {path.name} (erreur CSV: {e})")
        return False

    if not rows:
        print(f"[skip] {path.name} (CSV vide ou header absent)")
        return False

    ids, docs, metas = [], [], []
    base_id = doc_id_for(path)
    for (row_idx, row_dict) in rows:
        text = csv_row_to_text(row_dict, CSV_TRUNCATE_CHARS)
        if not text.strip():
            continue
        uid = f"{base_id}:{row_idx}"
        ids.append(uid)
        docs.append(text)
        metas.append({"source": str(path), "row_index": row_idx, "chunk": row_idx})

    print(f"[ingest:csv] {path.name} → {len(docs)} lignes")

    embeddings = embedder.encode(
        docs,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()

    collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
    return True

def add_textlike_document(path: Path) -> bool:
    ext = path.suffix.lower()
    raw = read_txt(path) if ext == ".txt" else read_md(path) if ext == ".md" else read_pdf(path)
    if not raw:
        print(f"[skip] {path.name} (vide ou illisible)")
        return False

    chunks = chunk(raw)
    if not chunks:
        print(f"[skip] {path.name} (aucun texte extrait)")
        return False

    ids, docs, metas = [], [], []
    base = doc_id_for(path)
    for idx, ch in enumerate(chunks):
        ids.append(f"{base}:{idx}")
        docs.append(ch)
        metas.append({"source": str(path), "chunk": idx})

    print(f"[ingest] {path.name} → {len(docs)} chunks")

    embeddings = embedder.encode(
        docs,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()

    collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
    return True

def add_document(path: Path):
    ext = path.suffix.lower()
    success = False

    if ext == ".csv":
        success = add_csv_document(path)
    elif ext in {".txt", ".md", ".pdf"}:
        success = add_textlike_document(path)
    else:
        print(f"[skip] {path.name} (extension non supportée: {ext})")

    if success:
        move_to_ingested(path)

# --- Entrée principale ---
def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    patterns = ["*.pdf", "*.txt", "*.md", "*.csv"]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(DOCS / p)))
    files = [Path(f) for f in sorted(files)]

    if not files:
        print(f"Aucun fichier dans {DOCS}. Ajoute des documents puis relance.")
        return

    print(f"Ingestion de {len(files)} fichier(s)…")
    for f in files:
        add_document(f)
    print("✅ Ingestion terminée.")

if __name__ == "__main__":
    main()