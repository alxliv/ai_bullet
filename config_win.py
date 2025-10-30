# config_win.py
import os

EMBEDDING_MODEL = 'nomic-embed-text'  # Local embedding model served by Ollama
CHROMA_DB_DIR = 'chroma_store/'
DOCUMENTS_PATH = 'D:/Work22/bullet3/docs'
SOURCES_PATH =   'D:/Work22/bullet3/src'
EXAMPLES_PATH =  "D:/Work22/bullet3/examples"
# Files to ignore during code processing (updatedb_code.py)
# Supports exact filenames and wildcard patterns
# Examples:
#   - Exact: "landscapeData.h", "generated.cpp"
#   - Wildcards: "test_*.cpp", "*_generated.h", "*.test.cpp"
IGNORE_FILES = {
    "landscapeData.h",      # Large landscape data file
    # Add more files to ignore here:
    # "test_*.cpp",         # Ignore all test files starting with test_
    # "*_generated.h",      # Ignore all generated header files
    # "*.test.cpp",         # Ignore all test source files
}

# DEPRECATED: These values are not currently used in the codebase
# Actual chunking parameters are in updatedb_docs.py and updatedb_code.py
# See CHUNK_SIZE_ANALYSIS.md for details
#
# Current optimal values (in use):
#   - Documents: MAX_ITEM_TOKENS=2048, overlap=100 (updatedb_docs.py)
#   - Code: MAX_ITEM_TOKENS=3000, overlap=0 (updatedb_code.py)
#
CHUNK_SIZE = 800  # DEPRECATED - Not used
CHUNK_OVERLAP = 50  # DEPRECATED - Not used

# Telemetry configuration for ChromaDB
ANONYMIZED_TELEMETRY = False
CHROMA_TELEMETRY = False

os.environ.setdefault("ANONYMIZED_TELEMETRY", str(ANONYMIZED_TELEMETRY).lower())
os.environ.setdefault("CHROMA_TELEMETRY", str(CHROMA_TELEMETRY).lower())
