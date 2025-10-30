# Token Limits Guide for Qwen3-4B

This guide explains how to set appropriate token limits for the Qwen3-4B model in the ai_bullet project.

## Qwen3-4B Model Specifications

Based on the Ollama model information:
- **Model Size**: 4.0B parameters
- **Context Length**: 262,144 tokens (256K)
- **Native Context**: 32,768 tokens (32K) - without extensions
- **Extended Context**: Up to 262K with optimizations

## Understanding Token Limits

### 1. Context Window vs Working Limits

The model's **context window** (256K tokens) is the absolute maximum, but you should use **working limits** that are much smaller for practical reasons:

- **Memory efficiency**: Smaller chunks use less RAM
- **Embedding quality**: Smaller chunks create more focused embeddings
- **Retrieval accuracy**: Smaller chunks are easier to match semantically
- **Processing speed**: Smaller chunks process faster

### 2. Recommended Token Limits

For the ai_bullet RAG system, here are the recommended limits:

#### For Documentation Processing (`updatedb_docs.py`)

```python
MAX_ITEM_TOKENS = 2048        # Maximum tokens per chunk (for embedding)
MAX_REQUEST_TOKENS = 8192     # Maximum tokens per batch request
DEFAULT_BATCH_LIMIT = 8192    # Total tokens in a single API call
```

**Rationale:**
- **2048 tokens per chunk**: Good balance between context and granularity
  - Approximately 1500-1600 words
  - Captures complete concepts without being too broad
  - Works well with RAG retrieval

- **8192 tokens per request**: Safe batch size for embedding
  - Well within the model's capabilities
  - Allows ~4 chunks per batch (efficient processing)
  - Leaves headroom for metadata and formatting

#### For Code Processing (`updatedb_code.py`)

```python
MAX_ITEM_TOKENS = 3000        # Maximum tokens per code chunk
MAX_REQUEST_TOKENS = 12000    # Maximum tokens per batch request
```

**Rationale:**
- **3000 tokens per chunk**: Handles larger functions/classes
  - Approximately 2200-2500 words of code
  - Captures complete functions with context
  - Includes leading comments and documentation

- **12000 tokens per request**: Larger batches for code
  - Code is typically more structured than prose
  - Allows ~4 code chunks per batch

#### For Retrieval Context (`retriever.py`)

```python
max_context_tokens = 16000    # Total context sent to LLM
max_snippets = 12             # Maximum number of retrieved chunks
```

**Rationale:**
- **16000 tokens for context**: Generous context for complex queries
  - Leaves ~240K tokens for the response
  - With 12 snippets, each can be ~1300 tokens
  - Provides rich context without overwhelming the model

### 3. Token Budget Breakdown

For a typical query to Qwen3-4B with 256K context:

```
Total Context Window:     262,144 tokens
├─ System Prompt:            ~500 tokens
├─ Retrieved Context:      16,000 tokens
├─ User Query:               ~100 tokens
├─ Conversation History:   ~2,000 tokens (optional)
├─ Available for Response: 243,544 tokens
└─ Reserved Headroom:        ~500 tokens
```

## Recommended Configuration by Use Case

### Conservative (Memory-Constrained Systems)

```python
# For systems with limited RAM (8GB or less)
MAX_ITEM_TOKENS = 1024
MAX_REQUEST_TOKENS = 4096
DEFAULT_BATCH_LIMIT = 4096
max_context_tokens = 8000
max_snippets = 8
```

### Balanced (Recommended)

```python
# Good balance for most systems (16GB+ RAM)
MAX_ITEM_TOKENS = 2048
MAX_REQUEST_TOKENS = 8192
DEFAULT_BATCH_LIMIT = 8192
max_context_tokens = 16000
max_snippets = 12
```

### Aggressive (High-Performance Systems)

```python
# For powerful systems (32GB+ RAM, dedicated GPU)
MAX_ITEM_TOKENS = 4096
MAX_REQUEST_TOKENS = 16384
DEFAULT_BATCH_LIMIT = 16384
max_context_tokens = 32000
max_snippets = 16
```

### Maximum (Experimental)

```python
# Push the limits (not recommended for production)
MAX_ITEM_TOKENS = 8192
MAX_REQUEST_TOKENS = 32768
DEFAULT_BATCH_LIMIT = 32768
max_context_tokens = 64000
max_snippets = 20
```

## How to Calculate Token Limits

### Rule of Thumb Formulas

1. **MAX_ITEM_TOKENS**:
   ```
   MAX_ITEM_TOKENS = Context_Window * 0.008
   For 256K: 256000 * 0.008 = 2048 tokens
   ```

2. **MAX_REQUEST_TOKENS**:
   ```
   MAX_REQUEST_TOKENS = MAX_ITEM_TOKENS * 4
   For 2048: 2048 * 4 = 8192 tokens
   ```

3. **max_context_tokens**:
   ```
   max_context_tokens = Context_Window * 0.06
   For 256K: 256000 * 0.06 = 15360 tokens (~16000)
   ```

### Testing Your Configuration

Run this test to verify your token limits:

```python
from tokenizer_utils import count_tokens

# Test text (approximately 1000 words)
test_text = "Your test content here..." * 100

token_count = count_tokens(test_text)
print(f"Token count: {token_count}")

# Check if it fits your limits
if token_count <= MAX_ITEM_TOKENS:
    print("✓ Fits within MAX_ITEM_TOKENS")
else:
    print(f"✗ Exceeds MAX_ITEM_TOKENS by {token_count - MAX_ITEM_TOKENS}")
```

## Performance Considerations

### Chunking Strategy

**Smaller chunks (1K-2K tokens):**
- ✓ Better retrieval precision
- ✓ Lower memory usage
- ✓ Faster embedding generation
- ✗ May split concepts across chunks
- ✗ More chunks to manage

**Larger chunks (3K-4K tokens):**
- ✓ Preserves more context
- ✓ Fewer total chunks
- ✓ Complete concepts in one chunk
- ✗ Less precise retrieval
- ✗ Higher memory usage

### Embedding Performance

For Ollama with Qwen3-4B on a typical system:

| Chunk Size | Chunks/Second | Memory Usage |
|------------|---------------|--------------|
| 1024 tokens | ~5-10 | Low (2-4GB) |
| 2048 tokens | ~3-7 | Medium (4-8GB) |
| 4096 tokens | ~1-3 | High (8-16GB) |

## Updating Configuration Files

### 1. Update `updatedb_docs.py`

```python
# Line 43-45
MAX_ITEM_TOKENS = 2048        # Recommended for Qwen3-4B
MAX_REQUEST_TOKENS = 8192
DEFAULT_BATCH_LIMIT = 8192
```

### 2. Update `updatedb_code.py`

```python
# Line 17-18
MAX_ITEM_TOKENS = 3000        # Slightly larger for code
MAX_REQUEST_TOKENS = 12000
```

### 3. Update `retriever.py` (RetrieverConfig)

```python
# Line 103-104
max_context_tokens: int = 16000,
max_snippets: int = 12,
```

## Monitoring and Optimization

### Signs Your Limits Are Too High

- Out of memory errors
- Slow embedding generation
- Poor retrieval relevance (chunks too broad)
- Long processing times

### Signs Your Limits Are Too Low

- Concepts split across multiple chunks
- Incomplete code functions
- Loss of important context
- Excessive number of chunks

### Optimization Tips

1. **Monitor token usage**: Log actual token counts during processing
2. **Adjust incrementally**: Change limits by 25-50% at a time
3. **Test retrieval quality**: Run sample queries and check relevance
4. **Profile memory usage**: Monitor RAM during batch processing
5. **Balance quality vs speed**: Find the sweet spot for your use case

## References

- Qwen3 Model Card: https://huggingface.co/Qwen/Qwen3-4B
- Ollama Documentation: https://ollama.ai/
- Token Counting: See `tokenizer_utils.py` for implementation

## Summary Table

| Configuration | MAX_ITEM_TOKENS | MAX_REQUEST_TOKENS | max_context_tokens | Use Case |
|---------------|-----------------|--------------------|--------------------|----------|
| Conservative  | 1024 | 4096 | 8000 | Low-end systems |
| **Balanced** (Recommended) | **2048** | **8192** | **16000** | **Most systems** |
| Aggressive    | 4096 | 16384 | 32000 | High-end systems |
| Maximum       | 8192 | 32768 | 64000 | Experimental |

Choose **Balanced** configuration for optimal results with Qwen3-4B!
