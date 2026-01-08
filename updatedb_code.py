import os
import sys
from dotenv import load_dotenv
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

from chromadb_shim import chromadb
from config import (
    CHROMA_DB_DIR,
    IGNORE_FILES,
    IGNORE_FOLDERS,
    GLOBAL_RAGDATA_MAP,
    RAGType,
)
from path_utils import encode_path
from updatedb_helper import (
    uniquify_records,
    short_hash,
    get_existing_ids,
    embed_and_add,
)

load_dotenv()

# Initialize tree-sitter parser for C++
CPP_LANGUAGE = Language(tscpp.language())
cpp_parser = Parser(CPP_LANGUAGE)

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}



def _is_separator_comment(line: str) -> bool:
    """Check if a line is a separator comment like // -------- or // ========."""
    stripped = line.strip()
    if not stripped.startswith("//"):
        return False
    # Get content after //
    content = stripped[2:].strip()
    # Check if it's mostly separator characters (dashes, equals, asterisks, etc.)
    if len(content) >= 3 and all(ch in '-=*#~_' for ch in content):
        return True
    return False


def grab_leading_comment(lines, start_idx, max_gap=2):
    """
    Walk upward from start_idx-1 to collect a contiguous block of comments
    separated from code by <= max_gap blank lines.
    Ignores separator comments like // -------- or // ========.
    """
    i = start_idx - 1
    collected = []
    blanks = 0
    while i >= 0:
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped.startswith("//") or "/*" in stripped:
            # Skip separator comments
            if _is_separator_comment(line):
                i -= 1
                continue
            collected.append(lines[i])
            blanks = 0
        elif stripped == "":
            blanks += 1
            if blanks > max_gap:
                break
            collected.append(lines[i])
        else:
            break
        i -= 1
    collected.reverse()
    return collected

def slice_lines(lines, start, end):
    return "\n".join(lines[start-1:end])

def _is_license_comment_block(text: str) -> bool:
    """Check if a text block is a license/copyright header comment."""
    lower = text.lower()

    # Keywords that indicate a license/copyright block
    license_keywords = [
        'copyright', 'license', 'licensed', 'all rights reserved',
        'permission is hereby granted', 'permission is granted',
        'software is provided', 'provided "as is"', "provided 'as-is'",
        'without warranty', 'express or implied',
        'in no event shall', 'authors be held liable',
        'redistribute', 'modification', 'sublicense',
        'apache license', 'mit license', 'bsd license', 'gpl',
        'gnu general public', 'lgpl', 'mozilla public license',
        'creative commons', 'public domain',
        'this file is part of', 'originally written by',
        'spdx-license-identifier',
    ]

    # Count how many license keywords appear
    keyword_count = sum(1 for kw in license_keywords if kw in lower)

    # If multiple license keywords found, it's likely a license block
    if keyword_count >= 2:
        return True

    # Single 'copyright' with year pattern is also a license header
    if 'copyright' in lower:
        import re
        if re.search(r'copyright\s*(?:\(c\)|©)?\s*\d{4}', lower):
            return True

    return False


def _strip_license_blocks(block_text: str) -> str:
    """
    Remove entire license/copyright comment blocks from text.

    Detects and removes:
    - Block comments (/* ... */) containing license text
    - Consecutive line comments (// ...) forming a license header
    - Mixed comment styles at file start
    """
    if not block_text:
        return block_text

    lines = block_text.splitlines(keepends=True)
    if not lines:
        return block_text

    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for block comment start /*
        if stripped.startswith('/*'):
            # Collect entire block comment
            block_lines = [line]
            # Check if it closes on the same line
            if '*/' in stripped[2:]:
                block_text_chunk = ''.join(block_lines)
                if _is_license_comment_block(block_text_chunk):
                    i += 1
                    continue
                else:
                    result_lines.append(line)
                    i += 1
                    continue

            # Multi-line block comment
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if '*/' in lines[i]:
                    break
                i += 1

            block_text_chunk = ''.join(block_lines)
            if not _is_license_comment_block(block_text_chunk):
                result_lines.extend(block_lines)
            i += 1
            continue

        # Check for consecutive line comments that might be a license header
        if stripped.startswith('//'):
            comment_lines = [line]
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if next_stripped.startswith('//') or next_stripped == '':
                    comment_lines.append(lines[j])
                    j += 1
                else:
                    break

            comment_text = ''.join(comment_lines)
            if _is_license_comment_block(comment_text):
                i = j
                continue
            else:
                result_lines.append(line)
                i += 1
                continue

        # Check for separator lines (all dashes, equals, asterisks)
        if stripped and len(stripped) >= 3 and all(ch in '-=*#' for ch in stripped):
            # Skip separator lines only if we haven't added any real content yet
            if not any(l.strip() and not l.strip().startswith(('//', '/*', '*'))
                      for l in result_lines):
                i += 1
                continue

        result_lines.append(line)
        i += 1

    return ''.join(result_lines)


def build_leftover_chunks(lines, used_spans, file_path):
    """Anything not covered by functions becomes its own chunk (contiguous blocks)."""
    n = len(lines)
    covered = [False]*(n+1)
    for s,e in used_spans:
        for i in range(s, e+1):
            if 0 < i <= n:
                covered[i] = True
    chunks = []
    i = 1
    while i <= n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j <= n and not covered[j]:
            j += 1
        text = _strip_license_blocks(slice_lines(lines, i, j-1))
        # Remove comment lines with separator dashes (e.g., //----, /*----)
        text = '\n'.join(
            line for line in text.splitlines()
            if not (line.lstrip().startswith(('//', '/*')) and '----' in line)
        )
        if text.strip():
            cid = f"{Path(file_path).name}:{i}-{j-1}-{short_hash(text)}"
            chunks.append({
                "id": cid,
                "text": text,
                "metadata": {
                    "file_path": encode_path(file_path),
                    "start_line": i,
                    "end_line": j-1,
                    "node_type": "leftover_block"
                }
            })
        i = j
    return chunks

def get_node_name(node, source_bytes):
    """Extract the name identifier from a tree-sitter node."""
    # For function_definition, look for declarator -> identifier
    # For class/struct/enum, look for name child

    if node.type == 'function_definition':
        # Navigate: function_definition -> declarator -> ... -> identifier
        declarator = node.child_by_field_name('declarator')
        if declarator:
            return _find_identifier_in_declarator(declarator, source_bytes)

    elif node.type in ('class_specifier', 'struct_specifier'):
        # Look for name field
        name_node = node.child_by_field_name('name')
        if name_node:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='ignore')

    elif node.type == 'enum_specifier':
        # Look for name field
        name_node = node.child_by_field_name('name')
        if name_node:
            return source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='ignore')

    return None


def _find_identifier_in_declarator(node, source_bytes):
    """Recursively find the identifier name in a declarator."""
    if node.type == 'identifier':
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

    if node.type == 'field_identifier':
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

    # Handle qualified identifiers (e.g., ClassName::methodName)
    if node.type == 'qualified_identifier':
        name_node = node.child_by_field_name('name')
        if name_node:
            return source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

    # Handle function declarators
    if node.type == 'function_declarator':
        declarator = node.child_by_field_name('declarator')
        if declarator:
            return _find_identifier_in_declarator(declarator, source_bytes)

    # Handle pointer/reference declarators
    if node.type in ('pointer_declarator', 'reference_declarator'):
        declarator = node.child_by_field_name('declarator')
        if declarator:
            return _find_identifier_in_declarator(declarator, source_bytes)

    # Handle parenthesized declarators
    if node.type == 'parenthesized_declarator':
        for child in node.children:
            result = _find_identifier_in_declarator(child, source_bytes)
            if result:
                return result

    # Fallback: search children
    for child in node.children:
        result = _find_identifier_in_declarator(child, source_bytes)
        if result:
            return result

    return None


def get_function_parameters(node, source_bytes):
    """Extract parameter list from a function definition."""
    declarator = node.child_by_field_name('declarator')
    if not declarator:
        return []

    # Find the function_declarator
    func_decl = None
    if declarator.type == 'function_declarator':
        func_decl = declarator
    else:
        # Search for function_declarator in children
        for child in declarator.children:
            if child.type == 'function_declarator':
                func_decl = child
                break
            # Handle pointer/reference declarators
            if child.type in ('pointer_declarator', 'reference_declarator'):
                for subchild in child.children:
                    if subchild.type == 'function_declarator':
                        func_decl = subchild
                        break

    if not func_decl:
        return []

    params = func_decl.child_by_field_name('parameters')
    if not params:
        return []

    param_list = []
    for child in params.children:
        if child.type == 'parameter_declaration':
            # Get the declarator name if present
            decl = child.child_by_field_name('declarator')
            if decl:
                name = _find_identifier_in_declarator(decl, source_bytes)
                if name:
                    param_list.append(name)

    return param_list

def read_code_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        cleaned_lines = []
        seen_blank = False
        for line in lines:
            if line == "\n":
                if seen_blank:
                    continue
                seen_blank = True
            else:
                seen_blank = False
            cleaned_lines.append(line)
        while cleaned_lines and cleaned_lines[-1] == "\n":
            cleaned_lines.pop()
        if cleaned_lines:
            cleaned_lines[-1] = cleaned_lines[-1].rstrip("\n")
        lines = cleaned_lines
    return lines

def extract_chunks_with_treesitter(path, include_comments=True):
    """
    Return list[{id,text,metadata}] for a single file using tree-sitter.

    Extracts the following node types:
    - function_definition
    - class_specifier
    - struct_specifier
    - enum_specifier
    """
    TARGET_TYPES = {'function_definition', 'class_specifier', 'struct_specifier', 'enum_specifier'}

    lines = read_code_file(path)
    src = "".join(lines)
    source_bytes = src.encode('utf-8')

    try:
        tree = cpp_parser.parse(source_bytes)
    except Exception as e:
        print(f"[tree-sitter] Failed to parse {path}: {e}")
        return []

    chunks = []
    spans = []

    # Walk the tree to find target nodes
    def visit(node):
        if node.type in TARGET_TYPES:
            # Get line numbers (tree-sitter uses 0-based, convert to 1-based)
            sl = node.start_point[0] + 1
            el = node.end_point[0] + 1

            # Extract the node text
            body_lines = lines[sl-1:el]
            comment_lines = grab_leading_comment(lines, sl-1) if include_comments else []
            text = "".join(comment_lines + body_lines)

            # Get node name
            name = get_node_name(node, source_bytes) or "anonymous"

            # Map tree-sitter types to our node_type names
            node_type_map = {
                'function_definition': 'function',
                'class_specifier': 'class',
                'struct_specifier': 'struct',
                'enum_specifier': 'enum'
            }
            node_type = node_type_map.get(node.type, node.type)

            # Build metadata
            metadata = {
                "file_path": encode_path(path),
                "start_line": sl,
                "end_line": el,
                "name": name,
                "node_type": node_type
            }

            # Add function-specific metadata
            if node.type == 'function_definition':
                params = get_function_parameters(node, source_bytes)
                metadata["parameter_count"] = len(params)
                metadata["parameters"] = ",".join(params) if params else ""
                # Build long_name similar to lizard format
                metadata["long_name"] = f"{name}({', '.join(params)})"

            cid = f"{Path(path).name}:{sl}-{el}-{short_hash(text)}"
            chunks.append({
                "id": cid,
                "text": text,
                "metadata": metadata
            })
            spans.append((sl, el))

            # Don't recurse into this node's children for nested definitions
            # (e.g., methods inside classes are handled separately)
            return

        # Recurse into children
        for child in node.children:
            visit(child)

    visit(tree.root_node)

    # Add non-covered regions (global variables, typedefs, etc.)
    leftovers = build_leftover_chunks(lines, spans, path)
    chunks.extend(leftovers)

    return chunks

def should_ignore_folder(path: str, ignore_patterns) -> bool:
    """
    Check if a folder should be ignored based on ignore rules.
    """
    import fnmatch
    if not ignore_patterns:
        return False

    # Normalize path and split into components to check each parent folder
    path = os.path.normpath(path)
    parts = path.split(os.sep)

    for part in parts:
        if not part:
            continue

        if part in ignore_patterns:
            return True

        for pattern in ignore_patterns:
            if '*' in pattern or '?' in pattern:
                if fnmatch.fnmatch(part, pattern):
                    return True
    return False

def should_ignore_file(filename: str, ignore_patterns) -> bool:
    """
    Check if a file should be ignored based on ignore rules.

    Args:
        filename: Name of the file to check
        ignore_patterns: Set or iterable of ignore patterns (exact names or wildcards)

    Returns:
        True if file should be ignored, False otherwise

    Examples:
        >>> should_ignore_file("test.cpp", {"test.cpp"})
        True
        >>> should_ignore_file("landscapeData.h", {"landscapeData.h"})
        True
        >>> should_ignore_file("test_main.cpp", {"test_*.cpp"})
        True
    """
    import fnmatch

    if not ignore_patterns:
        return False

    # Check exact match first (faster)
    if filename in ignore_patterns:
        return True

    # Check wildcard patterns
    for pattern in ignore_patterns:
        if '*' in pattern or '?' in pattern:
            if fnmatch.fnmatch(filename, pattern):
                return True

    return False


def walk_repo_and_chunk(root_dir, ignore_files=IGNORE_FILES):
    """
    Walk directory tree and extract code chunks from C/C++ files.

    Args:
        root_dir: Root directory to search
        ignore_files: Set of filenames/patterns to skip. Supports:
                     - Exact names: {"landscapeData.h", "test.cpp"}
                     - Wildcards: {"test_*.cpp", "*_generated.h"}

    Returns:
        List of chunk dictionaries

    Examples:
        >>> chunks = walk_repo_and_chunk("./src", {"test_*.cpp", "generated.h"})
    """
    all_chunks = []
    num_folders = 0
    total_files_processed = 0
    total_files_skipped = 0
    total_folders_skipped = 0
    max_chunks = 0
    max_dirpath=''
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"Directory not found: {root_dir}")

    for dirpath, _, files in os.walk(root_dir):
        print(f"#{num_folders}. walk_repo_and_chunk() path={dirpath}")
        count = 0
        num_chunks=0

        if should_ignore_folder(dirpath, IGNORE_FOLDERS):
            print(f"Skipping ignored folder {dirpath}")
            total_folders_skipped += 1
            continue

        num_folders += 1

        for name in files:
            # Check if file should be ignored
            if should_ignore_file(name, ignore_files):
                print(f"\tSkipping ignored file: {name}")
                total_files_skipped += 1
                continue

            # Check if it's a C/C++ file
            if os.path.splitext(name)[1].lower() in CPP_EXTS:
                p = os.path.join(dirpath, name)
                try:
                    chunks = extract_chunks_with_treesitter(p)
                    all_chunks.extend(chunks)
                    count += 1
                    num_chunks += len(chunks)
                    total_files_processed += 1
                    print(f"\tFile: {name}, {count} files chunked in this folder {num_chunks} chunks)")
                    if num_chunks > max_chunks:
                        max_chunks = num_chunks
                        max_dirpath = dirpath
                        print(f"!New max_chunks {max_chunks} was made in {dirpath}")

                except Exception as e:
                    print(f"\t[ERROR] Failed to process {name}: {e}")
                    total_files_skipped += 1

    print(f"\n=== Summary ===")
    print(f"Folders processed: {num_folders}")
    print(f"Files processed: {total_files_processed}")
    print(f"Files skipped/ignored: {total_files_skipped}, Folders ignored: {total_folders_skipped}")
    print(f"Total chunks extracted: {len(all_chunks)}, max_chunks {max_chunks} is in {max_dirpath}")

    return all_chunks


# --- ID helpers --------------------------------------------------------------


def update_code_collection(db_client, name, full_path):
    print(f"Updating code collection {name}")
    collection = db_client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
)

    code_chunks = walk_repo_and_chunk(full_path)
    print(f"All {len(code_chunks)} chunks from {full_path} collected.")

    existing = get_existing_ids(collection)
    already_seen = set(existing)
    uniq_records = uniquify_records(code_chunks, already_seen=already_seen)

    embed_and_add(uniq_records, collection)
    print("Done.")

def main():
    valid_names = ", ".join(
        sorted(key for key, (_, entry_type) in GLOBAL_RAGDATA_MAP.items() if entry_type == RAGType.SRC)
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python updatedb_code.py <collection name>")
        print(f"  Valid names are: {valid_names}")
        cname = "BASECODE"
#        return
    else:
        cname = sys.argv[1]

    rag_entry = GLOBAL_RAGDATA_MAP.get(cname)
    if rag_entry is None:
        print(f"[ERROR] Unknown collection '{cname}'. Valid options are: {valid_names}")
        return

    doc_path, _ = rag_entry
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    update_code_collection(client, cname, doc_path)

if __name__ == "__main__":
    main()
