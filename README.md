# AI Bullet

An AI-powered question-answering system specifically designed for the Bullet Physics library. This project provides an intelligent assistant that can answer questions about Bullet3 physics engine documentation and source code using Retrieval-Augmented Generation (RAG).

## Project Goals

AI Bullet aims to make the Bullet Physics library more accessible by providing:

- **Intelligent Documentation Search**: Query Bullet3 documentation using natural language instead of manual searching
- **Code Understanding**: Ask questions about Bullet3 C++ source code with context-aware responses
- **Comprehensive Knowledge Base**: Combines documentation, source code, and examples into a unified searchable database
- **Developer Productivity**: Reduce time spent searching through documentation and source code
- **Learning Support**: Help developers learn Bullet Physics concepts through interactive Q&A

The system uses OpenAI's embedding models and ChromaDB for vector storage, enabling semantic search across multiple types of content including PDFs, documentation files, and C++ source code.

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

### Python Dependencies

1. **Clone the repository:**
```bash
git clone https://github.com/alxliv/ai_bullet.git
cd ai_bullet
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

The main dependencies include:
- `openai>=1.0.0` - OpenAI API client
- `chromadb` - Vector database for embeddings
- `fastapi` - Web framework for the UI
- `tiktoken` - Token counting for OpenAI models
- `tree_sitter` - C++ code parsing
- `pypdf` - PDF document processing
- `python-docx` - Word document processing

### Environment Setup

1. **Create environment file:**
```bash
cp .env_example .env
```

2. **Configure your settings in `.env`:**
```bash
# OpenAI API Configuration
OPENAI_API_KEY="your-openai-api-key-here"

# User database - format: username:password,username:password
USERS_DB=admin:12345,user01:demo

# ChromaDB Telemetry Suppression
ANONYMIZED_TELEMETRY=False
CHROMA_TELEMETRY=False
```

3. **Configure paths in `config.py`:**
```python
EMBEDDING_MODEL = 'text-embedding-3-small'  # Or text-embedding-ada-002
CHROMA_DB_DIR = 'chroma_store/'
DOCUMENTS_PATH = '~/work/rag_data/bullet3/docs'  # Path to Bullet3 docs
SOURCES_PATH = '~/work/rag_data/bullet3/src'     # Path to Bullet3 source
EXAMPLES_PATH = "~/work/rag_data/bullet3/examples"  # Path to examples
CHUNK_SIZE = 800  # tokens/words
CHUNK_OVERLAP = 50
```

### Platform-Specific Notes

**Windows:**
- Use forward slashes or escaped backslashes in path configurations
- Ensure Python is added to your PATH
- Consider using Windows Subsystem for Linux (WSL) for better compatibility

**macOS:**
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for Python if needed: `brew install python`

**Linux:**
- Install development tools: `sudo apt-get install build-essential`
- Ensure you have the latest pip: `python -m pip install --upgrade pip`

## Usage

### Quick Start

1. **Set up your Bullet3 data** (if you have access to Bullet3 repository):
```bash
# Clone Bullet3 (optional - for full functionality)
git clone https://github.com/bulletphysics/bullet3.git ~/work/rag_data/bullet3
```

2. **Update the knowledge base with documentation:**
```bash
python updatedb_docs.py
```

3. **Update the knowledge base with source code:**
```bash
python updatedb_code.py
```

4. **Start the web interface:**
```bash
python webgui.py
```

5. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

### Command-Line Examples

#### Database Population

**Index documentation files:**
```bash
# Process PDFs, DOCX, Markdown, and text files
python updatedb_docs.py
```

**Index C++ source code:**
```bash
# Parse and index C++ files with syntax awareness
python updatedb_code.py
```

#### Running the Web Interface

**Standard server:**
```bash
python webgui.py
```

**Using uvicorn directly:**
```bash
uvicorn webgui:app --host 0.0.0.0 --port 8501 --reload
```

**Debug server (for testing):**
```bash
python debug_server.py
```

#### Direct API Usage

**Using the retriever programmatically:**
```python
from retriever import create_retriever, ask_llm

# Create retriever instance
retriever = create_retriever()

# Ask a question
question = "How do I create a rigid body in Bullet3?"
answer, sources = ask_llm(question, retriever)
print(f"Answer: {answer}")
```

### Configuration Options

#### Web Interface Settings

- **Host/Port**: Modify in `webgui.py` or use uvicorn parameters
- **Authentication**: Configure users in `.env` file using `USERS_DB`
- **Session Management**: Automatic cleanup of old sessions

#### Retrieval Settings

Configure in `config.py`:
- `EMBEDDING_MODEL`: Choose OpenAI embedding model
- `CHUNK_SIZE`: Size of text chunks for processing
- `CHUNK_OVERLAP`: Overlap between consecutive chunks
- `CHROMA_DB_DIR`: Directory for vector database storage

#### Advanced Configuration

**Retriever parameters** (in `retriever.py`):
```python
config = RetrieverConfig(
    k_per_collection={"cpp_code": 12, "bullet_docs": 4},
    use_mmr=True,
    mmr_diversity_bias=0.1,
    context_budget_tokens=4000
)
```

### Basic Workflow

1. **Preparation Phase:**
   - Set up OpenAI API key
   - Configure paths to Bullet3 data
   - Install dependencies

2. **Indexing Phase:**
   - Run `updatedb_docs.py` to process documentation
   - Run `updatedb_code.py` to process source code
   - Verify ChromaDB collections are created

3. **Usage Phase:**
   - Start web interface with `python webgui.py`
   - Access at `http://localhost:8501`
   - Log in with configured credentials
   - Ask questions about Bullet Physics

### Example Questions

Once set up, you can ask questions like:
- "How do I create a rigid body in Bullet3?"
- "What are the basic data types in Bullet Physics?"
- "Explain the constraint solver implementation"
- "How does collision detection work in Bullet3?"
- "What examples are available for soft body dynamics?"

### Troubleshooting

**Common Issues:**

1. **ChromaDB errors**: Ensure `chroma_store/` directory has proper permissions
2. **OpenAI API errors**: Verify your API key and check rate limits
3. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
4. **Path issues**: Use absolute paths or ensure relative paths are correct
5. **Memory issues**: Reduce `CHUNK_SIZE` if processing large files

**Logs and Debugging:**
- Web interface logs appear in console when running `webgui.py`
- Check browser developer tools for frontend issues
- Use `debug_server.py` for isolated testing

For additional support, please check the project's issue tracker or create a new issue with detailed error information.