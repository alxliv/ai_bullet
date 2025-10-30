#!/usr/bin/env python3
"""
db_tools_simple.py
==================

Simple utility for managing ChromaDB collections using ChromaDB client API.

Usage:
    python db_tools_simple.py list
    python db_tools_simple.py remove <collection_name>
    python db_tools_simple.py info
"""

import os
import sys
from pathlib import Path

# Disable telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import chromadb

try:
    from config import CHROMA_DB_DIR
except:
    CHROMA_DB_DIR = "chroma_store_qwen3/"


def get_client():
    """Get ChromaDB client."""
    db_path = os.path.expanduser(CHROMA_DB_DIR)

    if not os.path.exists(db_path):
        print(f"Warning: Database directory does not exist: {db_path}")
        print("Creating directory...")
        os.makedirs(db_path, exist_ok=True)

    return chromadb.PersistentClient(path=db_path)


def list_collections_simple():
    """List all collections using ChromaDB client."""
    db_path = os.path.expanduser(CHROMA_DB_DIR)

    try:
        client = get_client()
        collections = client.list_collections()
    except Exception as e:
        print(f"Error accessing database with ChromaDB client: {e}")
        print("\nThis may indicate a database created with an older ChromaDB version.")
        print("Recommendation: Delete the database and recreate it:")
        print(f"  rm -rf {db_path}")
        print(f"  python updatedb_code.py")
        print(f"  python updatedb_docs.py")
        return

    print(f"\n{'='*60}")
    print(f"Collections in database: {db_path}")
    print(f"{'='*60}")

    if not collections:
        print("No collections found in the database.")
        print("\nTip: Run updatedb_code.py or updatedb_docs.py to create collections")
    else:
        for idx, collection in enumerate(collections, 1):
            try:
                count = collection.count()
                print(f"{idx}. {collection.name}")
                print(f"   └─ Documents: {count:,}")
            except Exception as e:
                print(f"{idx}. {collection.name}")
                print(f"   └─ Error getting count: {e}")

        print(f"{'='*60}")
        print(f"Total collections: {len(collections)}")

    print()


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

    # Get document count for confirmation
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
    except Exception as e:
        print(f"Warning: Could not get document count: {e}")
        count = "unknown"

    # Confirmation
    if not force:
        print(f"\n⚠️  WARNING: About to delete collection '{collection_name}'")
        if count != "unknown":
            print(f"   This collection contains {count:,} documents.")
        print(f"   This action cannot be undone!\n")

        response = input("Proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled")
            return False

    # Delete using ChromaDB API
    try:
        client.delete_collection(collection_name)
        if count != "unknown":
            print(f"\n✓ Successfully deleted collection '{collection_name}' ({count:,} documents)")
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

        if collections:
            total_docs = 0
            for collection in collections:
                try:
                    count = collection.count()
                    total_docs += count
                    print(f"  - {collection.name}: {count:,} documents")
                except Exception as e:
                    print(f"  - {collection.name}: Error getting count")

            if total_docs > 0:
                print(f"\nTotal documents: {total_docs:,}")
    except Exception as e:
        print(f"Error accessing collections: {e}")

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
        print("  python db_tools_simple.py list")
        print("  python db_tools_simple.py remove <collection_name> [--force]")
        print("  python db_tools_simple.py clean [--force]     # Delete entire database")
        print("  python db_tools_simple.py info")
        return 1

    command = sys.argv[1]

    if command == "list":
        list_collections_simple()

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

    elif command == "info":
        show_info()

    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, remove, clean, info")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
