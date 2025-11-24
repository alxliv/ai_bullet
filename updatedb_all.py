#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
from chromadb_shim import chromadb
from updatedb_docs import update_docs_collection
from updatedb_code import update_code_collection

from config import (
    CHROMA_DB_DIR,
    GLOBAL_RAGDATA_MAP,
    RAGType,
)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

def main():

    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)

    for cname, (cpath, ctype) in GLOBAL_RAGDATA_MAP.items():
        if ctype is RAGType.DOC:
            update_docs_collection(client, cname, cpath)
        elif ctype is RAGType.SRC:
            update_code_collection(client, cname, cpath)
        else:
            print(f"ERROR: collection {cname} has unsupported type {ctype}. Ignored")

    print("All collections updated")

if __name__ == "__main__":
    main()
