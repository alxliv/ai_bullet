# OS-Agnostic Path Encoding System

## Overview

The AI Bullet RAG system uses an OS-agnostic path encoding system that allows the ChromaDB database to be portable across Windows, Linux, and macOS platforms.

## Problem Solved

Previously, file paths were stored as absolute paths in ChromaDB metadata:
- Windows: `D:\Work22\bullet3\docs\manual.pdf`
- Linux: `/home/user/bullet3/docs/manual.pdf`
- VPS: `/home/ubuntu/work/rag_data/bullet3/docs/manual.pdf`

This made databases non-portable - a database created on one system couldn't be used on another.

## Solution

File paths are now encoded using variable-based prefixes with curly braces:

```
{DOCS}/manual.pdf
{SRC}/btRigidBody.cpp
{EXAMPLES}/HelloWorld/main.cpp
```

**Note:** The curly brace format `{VAR}` was chosen to avoid collisions with LaTeX math notation (`$...$`) commonly found in technical documentation.

These variables map to actual directories defined in `config.py`:

| Variable | Windows Example | Linux Example |
|----------|----------------|---------------|
| `{DOCS}` | `D:\Work22\bullet3\docs` | `/home/user/bullet3/docs` |
| `{SRC}` | `D:\Work22\bullet3\src` | `/home/user/bullet3/src` |
| `{EXAMPLES}` | `D:\Work22\bullet3\examples` | `/home/user/bullet3/examples` |

## How It Works

### 1. Encoding (Storage Phase)

When ingesting documents or code:

```python
from path_utils import encode_path

# Absolute path on your system
abs_path = "D:/Work22/bullet3/docs/manual.pdf"

# Encoded for storage
encoded = encode_path(abs_path)  # Returns: "{DOCS}/manual.pdf"

# Store in ChromaDB metadata
metadata = {"file_path": encoded, ...}
```

### 2. Decoding (Retrieval Phase)

When retrieving results:

```python
from path_utils import decode_path

# Encoded path from database
encoded = "{DOCS}/manual.pdf"

# Decoded to absolute path for current OS
abs_path = decode_path(encoded)  # Returns: "D:/Work22/bullet3/docs/manual.pdf"

# Also supports legacy $VAR$ format for backward compatibility
legacy = "$DOCS$/manual.pdf"
abs_path = decode_path(legacy)  # Also works!
```

### 3. Path Format

- Uses forward slashes (`/`) as separators (POSIX-style)
- OS-specific separators are applied during decoding
- Relative paths are preserved within the encoded format

## Files Modified

### Core Implementation
- **`path_utils.py`** - Path encoding/decoding utilities
  - `encode_path()` - Convert absolute → variable-based
  - `decode_path()` - Convert variable-based → absolute
  - `is_encoded_path()` - Check if path is encoded
  - `get_path_variable()` - Get variable for a path

### Database Ingestion
- **`updatedb_docs.py`** - Document ingestion (uses `encode_path()`)
- **`updatedb_code.py`** - Code ingestion (uses `encode_path()`)

### Retrieval & Display
- **`retriever.py`** - RAG retrieval (uses `decode_path()`)
- **`app.py`** - Web application (removed old path translation)

### Migration
- **`migrate_paths.py`** - Migrate existing databases

## Usage

### For New Databases

Just run the ingestion scripts - paths will be encoded automatically:

```bash
python updatedb_docs.py
python updatedb_code.py
```

### For Existing Databases

Migrate your existing database to use the new encoding:

```bash
# Preview changes
python migrate_paths.py --dry-run

# Apply migration
python migrate_paths.py

# Migrate specific collection
python migrate_paths.py --collection bullet_docs
```

### Testing

Test the encoding/decoding system:

```bash
python path_utils.py
```

Expected output:
```
Path Encoding/Decoding Demo
==================================================

Original:  D:\Work22\bullet3\docs\manual.pdf
Encoded:   {DOCS}/manual.pdf
Decoded:   D:\Work22\bullet3\docs\manual.pdf
Match:     OK
```

## Configuration

Set your path roots in `config.py`:

```python
# config.py (or config_win.py / config_posix.py)
DOCUMENTS_PATH = 'D:/Work22/bullet3/docs'
SOURCES_PATH = 'D:/Work22/bullet3/src'
EXAMPLES_PATH = 'D:/Work22/bullet3/examples'
```

The `path_utils.py` module automatically reads these settings.

## Benefits

1. **Portability**: Database works across different operating systems
2. **Flexibility**: Easy to relocate data directories by changing `config.py`
3. **Clarity**: Variable names make paths self-documenting
4. **Backward Compatible**: Migration script handles old formats (including legacy `$VAR$`)
5. **LaTeX Safe**: Curly brace format doesn't collide with LaTeX math notation (`$...$`)

## Implementation Details

### Path Matching

The encoding uses longest-prefix matching:

```python
# Given these roots:
DOCS_ROOT = "D:/Work/bullet3/docs"
SRC_ROOT = "D:/Work/bullet3/src"

# This path:
"D:/Work/bullet3/docs/api/manual.pdf"

# Matches DOCS_ROOT and becomes:
"{DOCS}/api/manual.pdf"
```

### Edge Cases

- **Unmatched paths**: Paths not under any root are stored unchanged
- **Cross-drive paths** (Windows): Paths on different drives won't match
- **Symlinks**: Resolved to actual paths before encoding

### Migration Logic

The migration script (`migrate_paths.py`):

1. Migrates legacy `$VAR$` format to new `{VAR}` format
2. Handles old VPS paths (`/home/ubuntu/work/rag_data/...`)
3. Normalizes them to current system paths
4. Encodes to variable format
5. Updates ChromaDB metadata in batches

### Legacy Format Support

For backward compatibility, the `decode_path()` function supports both:
- **Current format**: `{DOCS}/file.pdf` (preferred)
- **Legacy format**: `$DOCS$/file.pdf` (deprecated, but still works)

The migration script will automatically convert legacy format to the new format.

## Troubleshooting

### Path not encoding

Check that the path is under one of the configured roots:

```python
from path_utils import get_path_variable

path = "D:/Work22/bullet3/docs/file.pdf"
var = get_path_variable(path)
print(var)  # Should print: {DOCS}
```

### Decoding returns wrong path

Verify your `config.py` settings match your directory structure:

```python
from path_utils import DOCS_ROOT, SRC_ROOT, EXAMPLES_ROOT

print(f"DOCS: {DOCS_ROOT}")
print(f"SRC: {SRC_ROOT}")
print(f"EXAMPLES: {EXAMPLES_ROOT}")
```

### Migration doesn't change anything

The path might already be encoded. Check with:

```python
from path_utils import is_encoded_path

path = "{DOCS}/manual.pdf"
print(is_encoded_path(path))  # True

# Legacy format also returns True
legacy = "$DOCS$/manual.pdf"
print(is_encoded_path(legacy))  # True
```

## Design Decision: Why `{VAR}` Instead of `$VAR$`?

The original implementation used `$DOCS$`, `$SRC$`, and `$EXAMPLES$`, but this created a serious collision with LaTeX math notation:

- LaTeX uses `$...$` for inline math and `$$...$$` for display math
- Technical documentation frequently contains mathematical formulas
- Storing text like `$x = y + z$` in the database would be ambiguous

The curly brace format `{VAR}` avoids this collision while remaining:
- **Readable**: Clear variable substitution syntax
- **Common**: Used by many template systems (Mustache, Jinja2-style)
- **Safe**: No collision with LaTeX, markdown, or common programming syntax
- **Backward Compatible**: Legacy `$VAR$` format still works via decode_path()

## Future Enhancements

Potential improvements:

- Support for additional root variables (`{TESTS}`, `{DATA}`, etc.)
- Environment variable expansion in paths
- Relative path support for project-internal references
- Path validation during encoding/decoding
