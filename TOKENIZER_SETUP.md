# Tokenizer Setup Guide

This project uses HuggingFace `transformers` library with AutoTokenizer for offline/local operation with Qwen3 models.

## Quick Start

### 1. Install Required Package

```bash
uv pip install transformers
```

Or install all dependencies:

```bash
uv pip install -r requirements.txt
```

### 2. Download Tokenizers (One-Time Setup)

Run the setup script while connected to the internet:

```bash
python setup_tokenizers.py
```

This will:
- Download the Qwen tokenizer from HuggingFace
- Save it to `./tokenizers/qwen/` directory
- Enable completely offline operation afterward

> [!IMPORTANT]
> You only need to run this **once** while online. After that, the tokenizer works completely offline.

### 3. Verify Installation

Test the tokenizer:

```bash
python tokenizer_utils.py
```

Expected output:
```
Testing tokenizer...
Loaded tokenizer from: ./tokenizers/qwen
Encoded tokens: [...]
Token count: XX
Decoded text: Hello, how are you? This is a test of the Qwen3 tokenizer.
...
Tokenizer test successful!
```

## Architecture Overview

### Tokenizer Storage Structure

```
tokenizers/
└── qwen/
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    └── special_tokens_map.json
```

### Model Mapping

The `tokenizer_utils.py` module maps model names to tokenizer directories:

```python
TOKENIZER_MAP = {
    "qwen3:4b-instruct": "./tokenizers/qwen",
    "qwen2.5": "./tokenizers/qwen",
    "qwen": "./tokenizers/qwen",
}
```

## Offline Operation

Once tokenizers are downloaded, the system operates completely offline:

- **`local_files_only=True`** - Prevents any internet access
- **`trust_remote_code=True`** - Safe offline (only local files executed)
- **Cached tokenizers** - LRU cache prevents reloading

## Usage in Code

### Basic Usage

```python
from tokenizer_utils import count_tokens, encode, decode

# Count tokens (default model: qwen3:4b-instruct)
num_tokens = count_tokens("Your text here")

# Encode text to token IDs
tokens = encode("Hello, world!")

# Decode token IDs back to text
text = decode(tokens)
```

### Specify Model

```python
# Use specific model
tokens = encode("Hello, world!", model="qwen2.5")
count = count_tokens("Your text", model="qwen")
```

### Advanced Features

```python
from tokenizer_utils import truncate, split_by_tokens

# Truncate to max tokens
truncated_text, actual_count = truncate("Long text...", max_tokens=100)

# Split into chunks
chunks = split_by_tokens("Very long text...", max_tokens=512)
```

## Manual Setup (Alternative)

If you prefer manual setup instead of using `setup_tokenizers.py`:

### Option A: Download with huggingface-cli

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --include "tokenizer*" --include "vocab.json" --include "special_tokens_map.json" \
  --local-dir ./tokenizers/qwen
```

### Option B: Python Script

```python
from transformers import AutoTokenizer

# Download and save
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)
tokenizer.save_pretrained("./tokenizers/qwen")
```

## Adding New Models

To add support for a new model:

1. **Download the tokenizer:**
   ```python
   tokenizer = AutoTokenizer.from_pretrained("model/name")
   tokenizer.save_pretrained("./tokenizers/model_name")
   ```

2. **Update TOKENIZER_MAP in tokenizer_utils.py:**
   ```python
   TOKENIZER_MAP = {
       "qwen3:4b-instruct": "./tokenizers/qwen",
       "new-model": "./tokenizers/model_name",  # Add this line
   }
   ```

3. **Test:**
   ```python
   from tokenizer_utils import count_tokens
   count_tokens("Test", model="new-model")
   ```

## Troubleshooting

### Error: "transformers library not installed"

**Solution:**
```bash
uv pip install transformers
```

### Error: "Tokenizer directory not found"

**Solution:** Run the setup script:
```bash
python setup_tokenizers.py
```

### Error: "Model 'xxx' not found in tokenizer map"

**Solution:** Add the model to `TOKENIZER_MAP` in `tokenizer_utils.py` or use one of the existing model names:
- `qwen3:4b-instruct` (default)
- `qwen2.5`
- `qwen`

### ImportError with trust_remote_code

**Solution:** This is normal if transformers isn't installed. Install it:
```bash
uv pip install transformers
```

## Migration from tiktoken

The project was migrated from OpenAI's `tiktoken` to `transformers`:

### Files Modified

1. **tokenizer_utils.py** - Rewritten to use `AutoTokenizer`
2. **retriever.py** - Updated to use new `tokenizer_utils`
3. **updatedb_code.py** - Updated to use new `tokenizer_utils`
4. **updatedb_docs.py** - Updated to use new `tokenizer_utils`
5. **requirements.txt** - Replaced `tiktoken` with `transformers`

### API Changes

| Old (tiktoken) | New (tokenizer_utils) |
|----------------|----------------------|
| `tiktoken.get_encoding("cl100k_base")` | `tokenizer_utils.get_tokenizer(model="qwen")` |
| `enc.encode(text)` | `tokenizer_utils.encode(text, model="qwen")` |
| `enc.decode(ids)` | `tokenizer_utils.decode(ids, model="qwen")` |
| `len(enc.encode(text))` | `tokenizer_utils.count_tokens(text, model="qwen")` |
| N/A | `tokenizer_utils.truncate(text, max_tokens, model="qwen")` |
| N/A | `tokenizer_utils.split_by_tokens(text, max_tokens, model="qwen")` |

### Why transformers + AutoTokenizer?

- **Offline operation:** No internet after initial setup
- **Consistency:** Uses same tokenizer as local Qwen3 models
- **Flexibility:** Easy to add new models
- **Caching:** Built-in LRU cache for performance
- **Standard:** Industry-standard library with wide support

### Why Qwen3 Tokenizer?

- **Large vocabulary:** ~150K tokens for efficient encoding
- **Multilingual:** Excellent Chinese and English support
- **Modern:** Optimized for latest LLM architectures
- **Compatible:** Works with Ollama, LM Studio, and other tools

## Security Considerations

### trust_remote_code Parameter

> [!NOTE]
> `trust_remote_code=True` is safe for offline usage because:
> - **No network access:** `local_files_only=True` prevents downloads
> - **Local execution only:** Code runs from your `tokenizers/` directory
> - **Controlled environment:** You control what's in the tokenizer files

The "remote code" becomes "local code" after the initial download.

### Best Practices

1. **Download from trusted sources** (official HuggingFace repositories)
2. **Verify checksums** if security is critical
3. **Review tokenizer files** before first use in production
4. **Keep transformers library updated** for security patches

## Performance

- **First load:** ~100-200ms (loads from disk)
- **Cached loads:** ~1ms (LRU cache hit)
- **Encoding:** ~0.1ms per 100 tokens
- **Memory usage:** ~50MB per loaded tokenizer

The `@lru_cache` decorator caches up to 5 tokenizers in memory for optimal performance.

## References

- HuggingFace Transformers: https://github.com/huggingface/transformers
- Qwen Models: https://huggingface.co/Qwen
- AutoTokenizer Docs: https://huggingface.co/docs/transformers/main_classes/tokenizer

## Support

If you encounter issues:

1. Verify transformers is installed: `pip list | grep transformers`
2. Check tokenizer directory exists: `ls tokenizers/qwen/`
3. Run test script: `python tokenizer_utils.py`
4. Check logs for detailed error messages
5. Try re-downloading: `python setup_tokenizers.py`
