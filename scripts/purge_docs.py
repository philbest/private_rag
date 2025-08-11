#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Calmer Chroma
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings

load_dotenv()

BASE_DIR = Path(os.getenv("BASE_DIR", "/Volumes/Data/Rag"))
INDEX_DIR = BASE_DIR / "index"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_docs")

client = chromadb.PersistentClient(
    path=str(INDEX_DIR),
    settings=Settings(allow_reset=False, anonymized_telemetry=False),
)
collection = client.get_or_create_collection(COLLECTION_NAME)

def delete_by_exact_source(src_path: str, dry_run: bool) -> int:
    """Supprime tous les chunks dont meta.source == src_path."""
    res = collection.get(where={"source": src_path}, include=["ids"])
    ids = res.get("ids", [])
    if not ids:
        return 0
    if dry_run:
        print(f"[dry-run] {src_path}: {len(ids)} chunk(s) correspondants")
        return len(ids)
    collection.delete(where={"source": src_path})
    print(f"[ok] Supprimé {len(ids)} chunk(s) pour source={src_path}")
    return len(ids)

def scan_and_delete_by_basename(basename: str, dry_run: bool) -> int:
    """
    Parcourt la collection pour trouver toutes les entrées dont
    meta.source se termine par basename (ex: 'chien.txt').
    """
    total = 0
    to_delete = []

    limit = 1000
    offset = 0
    while True:
        res = collection.get(include=["ids", "metadatas"], limit=limit, offset=offset)
        ids = res.get("ids", [])
        metas = res.get("metadatas", [])
        if not ids:
            break
        for id_, meta in zip(ids, metas):
            src = (meta or {}).get("source", "")
            if src and (Path(src).name == basename or src.endswith("/" + basename)):
                to_delete.append(id_)
        offset += len(ids)

    if not to_delete:
        return 0

    if dry_run:
        print(f"[dry-run] {basename}: {len(to_delete)} chunk(s) correspondants")
        return len(to_delete)

    collection.delete(ids=to_delete)
    print(f"[ok] Supprimé {len(to_delete)} chunk(s) pour basename={basename}")
    return len(to_delete)

def main():
    parser = argparse.ArgumentParser(
        description="Supprimer (désindexer) des documents de Chroma par source."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Chemins exacts (/…/docs/fichier.txt) ou noms de fichiers (fichier.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher ce qui serait supprimé sans rien effacer.",
    )
    parser.add_argument(
        "--purge-all",
        action="store_true",
        help="Vider entièrement la collection (ATTENTION, irréversible).",
    )
    args = parser.parse_args()

    if args.purge_all:
        if args.dry_run:
            print("[dry-run] purge complète de la collection")
            return 0
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(COLLECTION_NAME)
        print("[ok] Collection purgée et recréée.")
        return 0

    if not args.targets:
        print("Usage:\n  python scripts/purge_doc.py chien.txt\n  python scripts/purge_doc.py /Volumes/Data/Rag/docs/chien.txt\n  python scripts/purge_doc.py --dry-run chien.txt autre.pdf\n  python scripts/purge_doc.py --purge-all")
        return 1

    total_deleted = 0
    for t in args.targets:
        t = t.strip()
        # Si chemin absolu -> suppression exacte
        if t.startswith("/"):
            total_deleted += delete_by_exact_source(t, args.dry_run)
            continue

        # Sinon, on tente d'abord les chemins exacts docs/ et ingested/
        cand_docs = str(BASE_DIR / "docs" / t)
        cand_ing = str(BASE_DIR / "ingested" / t)

        deleted = 0
        deleted += delete_by_exact_source(cand_docs, args.dry_run)
        deleted += delete_by_exact_source(cand_ing, args.dry_run)

        # Si rien trouvé, on scanne par basename
        if deleted == 0:
            deleted += scan_and_delete_by_basename(t, args.dry_run)

        if deleted == 0:
            print(f"[info] Rien trouvé pour '{t}'")
        total_deleted += deleted

    if args.dry_run:
        print(f"[dry-run] total correspondances: {total_deleted}")
    else:
        print(f"[ok] total supprimé: {total_deleted}")
    return 0

if __name__ == "__main__":
    sys.exit(main())