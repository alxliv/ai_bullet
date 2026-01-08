# config_win.py
import os
from enum import Enum

USE_OPENAI = True

if USE_OPENAI:
    EMBEDDING_MODEL = 'text-embedding-3-small'  # Or other model like "text-embedding-ada-002"
    LLM_DEFAULT_MODEL = "gpt-4o"
    CHROMA_DB_DIR = 'chroma_store_openai_small/'
else:
    EMBEDDING_MODEL = 'snowflake-arctic-embed2'  # Local embedding model served by Ollama
    LLM_DEFAULT_MODEL = "gpt-oss:20b"
    CHROMA_DB_DIR = 'chroma_store_snowflake2/'

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

class RAGType(str, Enum):
    DOC = "doc"
    SRC = "src"

# For Windows the RAGDATA map would be something like:
'''
GLOBAL_RAGDATA_MAP = {
    "BOOKS":    ("D:/AL_KB/rag_data/books",            RAGType.DOC),
    "BASECODE": ("D:/AL_KB/rag_data/sources/basecode", RAGType.SRC),
    "EXTRAS":   ("D:/AL_KB/rag_data/sources/Extras",   RAGType.SRC),
    "EXAMPLES": ("D:/AL_KB/rag_data/sources/examples", RAGType.SRC),
    "TESTS":    ("D:/AL_KB/rag_data/sources/tests",    RAGType.SRC),
    "NOTES":    ("D:/AL_KB/rag_data/notes",            RAGType.DOC),
    "DOCS":     ("D:/AL_KB/rag_data/docs",             RAGType.DOC),
    "ARTICLES": ("D:/AL_KB/rag_data/articles",         RAGType.DOC),
    "MINCODE": ("D:/AL_KB/rag_data/sources/code_min", RAGType.SRC),

}
'''

# For posix (Linux) the RAGDATA map would be something like:
GLOBAL_RAGDATA_MAP = {
    "BOOKS":    ("~/work/rag_data/books",       RAGType.DOC),
    "ARTICLES": ("~/work/rag_data/articles",    RAGType.DOC),
    "DOCS":     ("~/work/rag_data/docs",        RAGType.DOC),
    "BASECODE": ("~/work/rag_data/sources/basecode/",    RAGType.SRC),
    "EXAMPLES": ("~/work/rag_data/sources/examples/",    RAGType.SRC),
    "EXTRAS":   ("~/work/rag_data/sources/Extras/",      RAGType.SRC),
    "TEST":     ("~/work/rag_data/sources/test/",        RAGType.SRC)
}

# Files to ignore during code processing (updatedb_code.py)
# Supports exact filenames and wildcard patterns
# Examples:
#   - Exact: "landscapeData.h", "generated.cpp"
#   - Wildcards: "test_*.cpp", "*_generated.h", "*.test.cpp"
IGNORE_FILES = {
    "landscapeData.h",      # Large landscape data file
    "b3Serializer.cpp",
    "bullet2.h"
    # Add more files to ignore here:
    # "test_*.cpp",         # Ignore all test files starting with test_
    # "*_generated.h",      # Ignore all generated header files
    # "*.test.cpp",         # Ignore all test source files
}

IGNORE_FOLDERS = {
    "xx_*",      # ignore all folders that start with xx_
}

QUERY_EXAMPLES = [
        "Describe DeformableDemo",
        "What value of timeStep is recommended for the integration?",
        "What is the Jacobi solver implementation?",
        "Explain btMotionState class definition",
        "How does collision detection work in Bullet3?",
        "Describe the constraint solver architecture",
        "Explain struct LuaPhysicsSetup",
        "How to compute the object AABBs?",
        "What types of constraints are available in Bullet3 and how do I create a hinge joint?",
        "Explain integration using Monte Carlo method (Note: if fails - increase Ollama context, see comment->)," # in app.py
            # see client.chat_stream(  # type: ignore[call-arg]
            # model=model,
            # messages=messages,
            # options={"num_ctx": 8192}
]


def get_collection_type(collection_name):
    for key, (_, rag_type) in GLOBAL_RAGDATA_MAP.items():
        if collection_name.upper() == key.upper():
            return rag_type.value
    return "UNKNOWN"

# Telemetry configuration for ChromaDB
ANONYMIZED_TELEMETRY = False
CHROMA_TELEMETRY = False

os.environ.setdefault("ANONYMIZED_TELEMETRY", str(ANONYMIZED_TELEMETRY).lower())
os.environ.setdefault("CHROMA_TELEMETRY", str(CHROMA_TELEMETRY).lower())
