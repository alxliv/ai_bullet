# Token Limits Quick Reference

## Qwen3-4B Model Stats
- **Context Window**: 256K tokens (262,144)
- **Parameters**: 4.0B
- **Recommended Context Usage**: 16K tokens (~6% of total)

## Current Configuration (Balanced)

### updatedb_docs.py
```python
MAX_ITEM_TOKENS = 2048        # Per chunk
MAX_REQUEST_TOKENS = 8192     # Per batch
DEFAULT_BATCH_LIMIT = 8192    # Request limit
```

### updatedb_code.py
```python
MAX_ITEM_TOKENS = 3000        # Per code chunk
MAX_REQUEST_TOKENS = 12000    # Per batch
```

### retriever.py (RetrieverConfig)
```python
max_context_tokens = 16000    # Context sent to LLM
max_snippets = 12             # Max retrieved chunks
```

## Quick Conversions

| Tokens | Approx. Words | Approx. Chars |
|--------|---------------|---------------|
| 512    | 380           | 2,048         |
| 1024   | 760           | 4,096         |
| 2048   | 1,520         | 8,192         |
| 3000   | 2,250         | 12,000        |
| 4096   | 3,072         | 16,384        |
| 8192   | 6,144         | 32,768        |
| 16000  | 12,000        | 64,000        |
| 32768  | 24,576        | 131,072       |

**Rule of Thumb**: 1 token ≈ 0.75 words ≈ 4 characters

## Alternative Configurations

### Conservative (8GB RAM)
```python
MAX_ITEM_TOKENS = 1024
MAX_REQUEST_TOKENS = 4096
max_context_tokens = 8000
```

### Aggressive (32GB+ RAM)
```python
MAX_ITEM_TOKENS = 4096
MAX_REQUEST_TOKENS = 16384
max_context_tokens = 32000
```

## When to Adjust

**Reduce limits if:**
- ❌ Out of memory errors
- ❌ Slow processing
- ❌ Poor retrieval relevance

**Increase limits if:**
- ✅ Concepts split across chunks
- ✅ Incomplete code functions
- ✅ Plenty of RAM available
- ✅ Need more context

## Testing Command
```bash
python tokenizer_utils.py
```

## Documentation
- Full guide: [TOKEN_LIMITS_GUIDE.md](TOKEN_LIMITS_GUIDE.md)
- Setup: [TOKENIZER_SETUP.md](TOKENIZER_SETUP.md)
- Migration: [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)
