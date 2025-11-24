#!/usr/bin/env python3
"""
migrate_paths.py
================
Migrate existing ChromaDB collections to use OS-agnostic path encoding.

This script updates all file_path metadata in existing ChromaDB collections
from absolute paths (or old hardcoded paths) to the new variable-based encoding
({DOCS}, {SRC}, {EXAMPLES}).

Also migrates legacy $VAR$ format to new {VAR} format.

Usage:
    python migrate_paths.py [--dry-run] [--collection COLLECTION_NAME]

Options:
    --dry-run           Show what would be changed without making changes
    --collection NAME   Only migrate specific collection (default: migrate all)
"""

import os
import argparse
from chromadb_shim import chromadb
from config import CHROMA_DB_DIR
from path_utils import encode_path, is_encoded_path, DOCS_ROOT, SRC_ROOT, EXAMPLES_ROOT, LEGACY_VARIABLES

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

# Old hardcoded paths from VPS deployment (for backward compatibility)
OLD_PATHS_MAP = {
    "/home/ubuntu/work/rag_data/bullet3/docs": DOCS_ROOT,
    "/home/ubuntu/work/rag_data/bullet3/src": SRC_ROOT,
    "/home/ubuntu/work/rag_data/bullet3/examples": EXAMPLES_ROOT,
}


def migrate_legacy_variable_format(path: str) -> str:
    """Convert legacy $VAR$ format to new {VAR} format."""
    for legacy_var, new_var in LEGACY_VARIABLES.items():
        if path.startswith(legacy_var + "/"):
            # Replace legacy variable with new format
            return new_var + path[len(legacy_var):]
    return path


def normalize_old_path(old_path: str) -> str:
    """Convert old hardcoded VPS paths to current system absolute paths."""
    for old_prefix, new_prefix in OLD_PATHS_MAP.items():
        if old_path.startswith(old_prefix):
            # Replace old prefix with new prefix
            rel_part = old_path[len(old_prefix):].lstrip('/')
            return os.path.join(new_prefix, rel_part)
    return old_path


def migrate_collection(collection, dry_run=False):
    """Migrate all documents in a collection to use encoded paths."""

    print(f"\n{'='*60}")
    print(f"Migrating collection: {collection.name}")
    print(f"{'='*60}")

    # Get all documents
    total = collection.count()
    if total == 0:
        print("  Collection is empty, nothing to migrate.")
        return

    print(f"  Total documents: {total}")

    # Fetch in batches to handle large collections
    BATCH_SIZE = 1000
    offset = 0
    total_migrated = 0
    total_already_encoded = 0
    total_unchanged = 0
    total_legacy_migrated = 0

    while offset < total:
        limit = min(BATCH_SIZE, total - offset)
        result = collection.get(
            include=["metadatas"],
            limit=limit,
            offset=offset
        )

        ids = result["ids"]
        metadatas = result["metadatas"]

        for doc_id, metadata in zip(ids, metadatas):
            if "file_path" not in metadata:
                total_unchanged += 1
                continue

            old_path = metadata["file_path"]

            # Check if it's in legacy $VAR$ format
            migrated_legacy = migrate_legacy_variable_format(old_path)
            if migrated_legacy != old_path:
                print(f"\n  Document: {doc_id}")
                print(f"    Old (legacy): {old_path}")
                print(f"    New:          {migrated_legacy}")

                if not dry_run:
                    metadata["file_path"] = migrated_legacy
                    collection.update(
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                total_legacy_migrated += 1
                total_migrated += 1
                continue

            # Check if already in new {VAR} format
            if is_encoded_path(old_path):
                total_already_encoded += 1
                continue

            # Normalize old hardcoded paths first
            normalized_path = normalize_old_path(old_path)

            # Encode the path
            new_path = encode_path(normalized_path)

            # Check if encoding actually changed the path
            if new_path == old_path:
                total_unchanged += 1
                continue

            print(f"\n  Document: {doc_id}")
            print(f"    Old: {old_path}")
            print(f"    New: {new_path}")

            if not dry_run:
                # Update the metadata
                metadata["file_path"] = new_path
                collection.update(
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                total_migrated += 1
            else:
                total_migrated += 1

        offset += limit

    print(f"\n  Summary for {collection.name}:")
    print(f"    Already encoded:  {total_already_encoded}")
    print(f"    Legacy migrated:  {total_legacy_migrated}")
    print(f"    Absolute migr.:   {total_migrated - total_legacy_migrated}")
    print(f"    Unchanged:        {total_unchanged}")
    print(f"    Total:            {total}")

    if dry_run:
        print(f"\n  [DRY RUN] No changes were made")


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB paths to OS-agnostic encoding")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--collection", type=str, help="Only migrate specific collection")
    args = parser.parse_args()

    print("ChromaDB Path Migration Tool")
    print("="*60)
    print(f"Database: {CHROMA_DB_FULL_PATH}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'MIGRATION'}")
    print()
    print("Path mappings:")
    print(f"  {{DOCS}}     -> {DOCS_ROOT}")
    print(f"  {{SRC}}      -> {SRC_ROOT}")
    print(f"  {{EXAMPLES}} -> {EXAMPLES_ROOT}")
    print()
    print("Legacy format migration:")
    print(f"  $DOCS$     -> {{DOCS}}")
    print(f"  $SRC$      -> {{SRC}}")
    print(f"  $EXAMPLES$ -> {{EXAMPLES}}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)

    # Get collections to migrate
    if args.collection:
        try:
            collections = [client.get_collection(args.collection)]
        except Exception as e:
            print(f"\nError: Collection '{args.collection}' not found: {e}")
            return 1
    else:
        collections = client.list_collections()

    if not collections:
        print("\nNo collections found to migrate.")
        return 0

    # Migrate each collection
    for collection in collections:
        migrate_collection(collection, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN COMPLETE - No changes were made")
        print("Run without --dry-run to apply changes")
    else:
        print("MIGRATION COMPLETE")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())
