# config_posix.py
import os

EMBEDDING_MODEL = 'text-embedding-3-small'  # Or other model like "text-embedding-ada-002"
CHROMA_DB_DIR = 'chroma_store/'
DOCUMENTS_PATH = '~/work/rag_data/bullet3/docs'
SOURCES_PATH =   '~/work/rag_data/bullet3/src'
EXAMPLES_PATH =  "~/work/rag_data/bullet3/examples"

# Files to ignore during code processing (updatedb_code.py)
# Supports exact filenames and wildcard patterns
# Examples:
#   - Exact: "landscapeData.h", "generated.cpp"
#   - Wildcards: "test_*.cpp", "*_generated.h", "*.test.cpp"
IGNORE_FILES = {
    "landscapeData.h",      # Large landscape data file
    "ignore.c",             # Example ignore file
    # Add more files to ignore here:
    # "test_*.cpp",         # Ignore all test files starting with test_
    # "*_generated.h",      # Ignore all generated header files
    # "*.test.cpp",         # Ignore all test source files
}

CHUNK_SIZE = 800  # tokens/words
CHUNK_OVERLAP = 50

# Telemetry configuration for ChromaDB
ANONYMIZED_TELEMETRY = False
CHROMA_TELEMETRY = False

os.environ.setdefault("ANONYMIZED_TELEMETRY", str(ANONYMIZED_TELEMETRY).lower())
os.environ.setdefault("CHROMA_TELEMETRY", str(CHROMA_TELEMETRY).lower())
