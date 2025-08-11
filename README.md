# 📚 RAG Local avec ChromaDB & Sentence Transformers

Ce projet met en place un **RAG** (Retrieval-Augmented Generation) local sur macOS / Ubuntu,  
permettant d’indexer vos documents personnels (**TXT, PDF, MD, CSV**) et de les interroger avec un LLM.

Les données restent **100% locales**, et l’index est géré avec **ChromaDB**.

---

## 📂 Structure du projet

```
/Volumes/Data/Rag
 ├── docs/          # Déposez ici vos fichiers à ingérer
 ├── ingested/      # Les fichiers traités sont déplacés ici
 ├── index/         # Index ChromaDB (embeddings)
 ├── scripts/
 │    ├── ingest.py     # Ingestion des documents
 │    ├── query.py      # Requête à la base
 │    ├── purge_doc.py  # Suppression d'éléments de l'index
 ├── .env           # Variables d'environnement
 └── README.md
```

---

## ⚙️ Installation

### 1. Créer un environnement virtuel
```bash
cd /Volumes/Data/Rag
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

Exemple de `requirements.txt` minimal :
```txt
chromadb
sentence-transformers
pypdf
python-dotenv
uvicorn
fastapi
```

---

## 📥 Ingestion de documents

1. Placez vos fichiers dans `docs/` :
```
/Volumes/Data/Rag/docs/chien.txt
/Volumes/Data/Rag/docs/contrat.pdf
/Volumes/Data/Rag/docs/clients.csv
```

2. Lancez le script d’ingestion :
```bash
source .venv/bin/activate
python scripts/ingest.py
```

3. Résultat :
   - Les embeddings sont ajoutés dans ChromaDB (`index/`)
   - Les fichiers sont déplacés automatiquement dans `ingested/`

---

## 🔍 Interroger la base

Utilisez `query.py` pour poser une question :
```bash
python scripts/query.py "Comment s'appelle mon chien ?"
```

---

## 🧹 Supprimer un document de l'index

- **Par nom de fichier** (qu’il soit dans `docs` ou `ingested`) :
```bash
python scripts/purge_doc.py chien.txt
```

- **Par chemin exact** :
```bash
python scripts/purge_doc.py /Volumes/Data/Rag/docs/chien.txt
```

- **Voir ce qui serait supprimé (dry-run)** :
```bash
python scripts/purge_doc.py --dry-run chien.txt
```

- **Purger toute la collection** :
```bash
python scripts/purge_doc.py --purge-all
```

---

## 📊 Formats supportés

- **.txt** → texte brut
- **.md** → Markdown
- **.pdf** → PDF (extraction avec PyPDF)
- **.csv** → lecture ligne par ligne, conversion en phrases `colonne="valeur"`

---

## 🛠 Variables d'environnement (`.env`)

Vous pouvez personnaliser certains réglages dans `.env` :

```env
BASE_DIR=/Volumes/Data/Rag
COLLECTION_NAME=personal_docs
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CSV_MAX_ROWS=100000
CSV_TRUNCATE_CHARS=200
CSV_SAMPLE_EVERY=1
```

---

## 💡 Astuces

- **Nettoyer `docs/`** : grâce au déplacement automatique, `docs/` reste vide après ingestion
- **Gros CSV** : ajustez `CSV_SAMPLE_EVERY` pour réduire la charge (ex: 10 = 1 ligne sur 10)
- **Suppression rapide** : utilisez `purge_doc.py` pour retirer des infos sans toucher aux autres

---

## 📜 Licence

Projet privé - Usage personnel uniquement.
