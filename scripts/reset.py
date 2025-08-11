from chromadb import PersistentClient
from chromadb.config import Settings
import os
from pathlib import Path

BASE = Path(os.getenv("BASE_DIR", "/Volumes/Data/Rag"))
INDEX_DIR = BASE / "index"
COLL = os.getenv("COLLECTION_NAME", "personal_docs")

client = PersistentClient(path=str(INDEX_DIR), settings=Settings(anonymized_telemetry=False))
try:
    client.delete_collection(COLL)
    print("Index supprimé.")
except Exception as e:
    print("Info:", e)
client.create_collection(COLL)
print("Index recréé.")