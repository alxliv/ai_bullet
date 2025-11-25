#!/usr/bin/env python3
"""
db_tools_simple.py
==================

Simple utility for managing ChromaDB collections using ChromaDB client API.

Usage:
    python db_tools_simple.py info
    python db_tools_simple.py remove <collection_name>
    python db_tools_simple.py clean [--force]
"""

import os
import sys
from pathlib import Path

# Disable telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from chromadb_shim import chromadb

try:
    from config import CHROMA_DB_DIR, GLOBAL_RAGDATA_MAP, RAGType
except ImportError:
    CHROMA_DB_DIR = "chroma_store_qwen3/"
    GLOBAL_RAGDATA_MAP = {}
    from enum import Enum
    class RAGType(str, Enum):
        DOC = "doc"
        SRC = "src"


def get_collection_type(collection_name):
    """Determine collection type (CODE/DOC) from collection name."""
    # Check if collection name matches any key in GLOBAL_RAGDATA_MAP
    for key, (_, rag_type) in GLOBAL_RAGDATA_MAP.items():
        if collection_name.upper() == key.upper():
            return rag_type.value

    return "UNKNOWN"


def get_client():
    """Get ChromaDB client."""
    db_path = os.path.expanduser(CHROMA_DB_DIR)

    if not os.path.exists(db_path):
        print(f"Warning: Database directory does not exist: {db_path}")
        print("Creating directory...")
        os.makedirs(db_path, exist_ok=True)

    return chromadb.PersistentClient(path=db_path)


def remove_collection_simple(collection_name, force=False):
    """Remove a collection using ChromaDB client API."""
    try:
        client = get_client()
        collections = client.list_collections()
    except Exception as e:
        print(f"Error accessing database: {e}")
        return False

    # Check if collection exists
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        print(f"Error: Collection '{collection_name}' does not exist")
        print(f"\nAvailable collections: {', '.join(collection_names)}")
        return False

    # Get records count for confirmation
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
    except Exception as e:
        print(f"Warning: Could not get records count: {e}")
        count = "unknown"

    # Confirmation
    if not force:
        print(f"\n⚠️  WARNING: About to delete collection '{collection_name}'")
        if count != "unknown":
            print(f"   This collection contains {count:,} records.")
        print(f"   This action cannot be undone!\n")

        response = input("Proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled")
            return False

    # Delete using ChromaDB API
    try:
        client.delete_collection(collection_name)
        if count != "unknown":
            print(f"\n✓ Successfully deleted collection '{collection_name}' ({count:,} records)")
        else:
            print(f"\n✓ Successfully deleted collection '{collection_name}'")
        return True
    except Exception as e:
        print(f"\n✗ Error deleting collection: {e}")
        return False


def show_info():
    """Show database information."""
    db_path = Path(os.path.expanduser(CHROMA_DB_DIR))

    print(f"\n{'='*60}")
    print(f"ChromaDB Database Information")
    print(f"{'='*60}")
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")

    if db_path.exists():
        # Calculate size
        total_size = 0
        try:
            total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
        except Exception as e:
            print(f"Warning: Could not calculate size: {e}")

        if total_size > 0:
            if total_size < 1024:
                size_str = f"{total_size} bytes"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.2f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.2f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
            print(f"Database size: {size_str}")

    # Get collections using ChromaDB client
    try:
        client = get_client()
        collections = client.list_collections()

        print(f"\nCollections: {len(collections)}")

        if not collections:
            print("No collections found in the database.")
            print("\nTip: Run updatedb_code.py or updatedb_docs.py to create collections")
        else:
            print()
            total_docs = 0
            for idx, collection in enumerate(collections, 1):
                try:
                    count = collection.count()
                    total_docs += count
                    col_type = get_collection_type(collection.name)
                    print(f"{idx}. {collection.name} ({col_type})")
                    print(f"   └─ Records: {count:,}")
                except Exception as e:
                    col_type = get_collection_type(collection.name)
                    print(f"{idx}. {collection.name} ({col_type})")
                    print(f"   └─ Error getting count: {e}")

            if total_docs > 0:
                print(f"\nTotal records: {total_docs:,}")
    except Exception as e:
        print(f"Error accessing collections: {e}")
        print("\nThis may indicate a database created with an older ChromaDB version.")
        print("Recommendation: Delete the database and recreate it:")
        print(f"  rm -rf {db_path}")
        print(f"  python updatedb_code.py")
        print(f"  python updatedb_docs.py")

    print(f"{'='*60}\n")


def clean_database(force=False):
    """Delete the entire database directory."""
    import shutil

    db_path = Path(os.path.expanduser(CHROMA_DB_DIR))

    if not db_path.exists():
        print(f"Database directory does not exist: {db_path}")
        return False

    if not force:
        print(f"\n⚠️  WARNING: About to delete the ENTIRE database!")
        print(f"   Location: {db_path}")
        print(f"   This will remove all collections and data.")
        print(f"   This action cannot be undone!\n")

        response = input("Type 'DELETE ALL' to confirm: ").strip()
        if response != 'DELETE ALL':
            print("Cancelled")
            return False

    try:
        shutil.rmtree(db_path)
        print(f"\n✓ Successfully deleted database: {db_path}")
        print("\nTo rebuild:")
        print("  python updatedb_code.py")
        print("  python updatedb_docs.py")
        return True
    except Exception as e:
        print(f"\n✗ Error deleting database: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python db_tools_simple.py info")
        print("  python db_tools_simple.py remove <collection_name> [--force]")
        print("  python db_tools_simple.py clean [--force]     # Delete entire database")
        return 1

    command = sys.argv[1]

    if command == "info":
        show_info()

    elif command == "remove":
        if len(sys.argv) < 3:
            print("Error: collection_name required")
            print("Usage: python db_tools_simple.py remove <collection_name>")
            return 1

        collection_name = sys.argv[2]
        force = "--force" in sys.argv
        remove_collection_simple(collection_name, force)

    elif command == "clean":
        force = "--force" in sys.argv
        clean_database(force)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: info, remove, clean")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
