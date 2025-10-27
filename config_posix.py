# config_posix.py
import os

EMBEDDING_MODEL = 'nomic-embed-text'  # Local embedding model served by Ollama
CHROMA_DB_DIR = 'chroma_store/'
DOCUMENTS_PATH = '~/work/rag_data/bullet3/docs'
SOURCES_PATH =   '~/work/rag_data/bullet3/src'
EXAMPLES_PATH =  "~/work/rag_data/bullet3/examples"
CHUNK_SIZE = 800  # tokens/words
CHUNK_OVERLAP = 50

# Telemetry configuration for ChromaDB
ANONYMIZED_TELEMETRY = False
CHROMA_TELEMETRY = False

os.environ.setdefault("ANONYMIZED_TELEMETRY", str(ANONYMIZED_TELEMETRY).lower())
os.environ.setdefault("CHROMA_TELEMETRY", str(CHROMA_TELEMETRY).lower())
