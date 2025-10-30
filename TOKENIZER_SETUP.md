# Qwen3 Tokenizer Setup Guide

This project has been migrated from OpenAI's `tiktoken` to the HuggingFace `tokenizers` library using the Qwen3 tokenizer for offline/local operation.

## Quick Start

### 1. Install Required Package

```bash
pip install tokenizers>=0.13.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download Qwen3 Tokenizer

You have several options to obtain the tokenizer:

#### Option A: Download from HuggingFace (Recommended)

```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-4B-Instruct --include "tokenizer.json" --local-dir ./models/qwen3
```

Or download manually from: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/main

#### Option B: Use Existing Ollama Model Cache

If you already have Qwen3 installed via Ollama, you may find the tokenizer in:

**Linux/macOS:**
```bash
~/.ollama/models/
```

**Windows:**
```bash
%USERPROFILE%\.ollama\models\
```

Copy the `tokenizer.json` file to your project directory.

#### Option C: Extract from Model Directory

If you have the full Qwen3-4B-Instruct model downloaded locally, copy the `tokenizer.json` file from the model directory.

### 3. Configure Tokenizer Path

Set the `TOKENIZER_PATH` environment variable in your `.env` file:

```bash
# .env
TOKENIZER_PATH=/path/to/Qwen3-4B-Instruct/tokenizer.json
```

Or place `tokenizer.json` in one of these locations (searched in order):

1. `./models/qwen3/tokenizer.json`
2. `./models/tokenizer.json`
3. `./tokenizer.json`
4. `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-4B-Instruct/snapshots/*/tokenizer.json`

## Testing the Tokenizer

Run the tokenizer module directly to test:

```bash
python tokenizer_utils.py
```

Expected output:
```
Testing Qwen3 tokenizer...
Loaded Qwen3 tokenizer from: /path/to/tokenizer.json
Encoded tokens: [...]
Token count: XX
Decoded text: Hello, how are you? This is a test of the Qwen3 tokenizer.
...
Tokenizer test successful!
```

## Troubleshooting

### Error: "Tokenizer file not found"

**Solution:** Make sure you've downloaded `tokenizer.json` and set `TOKENIZER_PATH` correctly.

### Error: "tokenizers library not installed"

**Solution:** Install the package:
```bash
pip install tokenizers>=0.13.0
```

### Fallback Behavior

If the tokenizer cannot be loaded, the code will fall back to character-based estimation:
- **Token counting:** Assumes ~4 characters per token
- **Text splitting:** Uses character-based chunking

This fallback allows the code to run but may result in less accurate token counting.

## Migration from tiktoken

The following changes were made to migrate from `tiktoken` to `tokenizers`:

### Files Modified

1. **tokenizer_utils.py** (NEW) - Unified tokenizer interface
2. **retriever.py** - Updated to use `tokenizer_utils`
3. **updatedb_code.py** - Updated to use `tokenizer_utils`
4. **updatedb_docs.py** - Updated to use `tokenizer_utils`
5. **requirements.txt** - Replaced `tiktoken` with `tokenizers>=0.13.0`

### API Compatibility

The new `tokenizer_utils` module provides a similar API to `tiktoken`:

| Old (tiktoken) | New (tokenizer_utils) |
|----------------|----------------------|
| `tiktoken.get_encoding("cl100k_base")` | `tokenizer_utils.get_tokenizer()` |
| `enc.encode(text)` | `tokenizer_utils.encode(text)` |
| `enc.decode(ids)` | `tokenizer_utils.decode(ids)` |
| `len(enc.encode(text))` | `tokenizer_utils.count_tokens(text)` |
| N/A | `tokenizer_utils.truncate(text, max_tokens)` |
| N/A | `tokenizer_utils.split_by_tokens(text, max_tokens)` |

### Why Qwen3?

- **Offline operation:** No internet connection required after initial setup
- **Consistency:** Uses the same tokenizer as your local Qwen3:4b model
- **Compatibility:** Works with Ollama and other local LLM setups
- **Open source:** HuggingFace tokenizers library is well-maintained

## Advanced Configuration

### Custom Tokenizer

To use a different tokenizer, modify `tokenizer_utils.py`:

```python
# In _load_tokenizer() function
_tokenizer = Tokenizer.from_file("/path/to/your/tokenizer.json")
```

### Programmatic Loading

```python
from tokenizer_utils import get_tokenizer, count_tokens

# Load tokenizer
tokenizer = get_tokenizer()

# Count tokens
num_tokens = count_tokens("Your text here")
print(f"Token count: {num_tokens}")
```

## References

- HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
- Qwen Models: https://huggingface.co/Qwen
- Ollama: https://ollama.ai/

## Support

If you encounter issues:

1. Verify `tokenizer.json` exists and is readable
2. Check the `TOKENIZER_PATH` environment variable
3. Run `python tokenizer_utils.py` to test
4. Check logs for detailed error messages
