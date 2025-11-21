# GPTil: AI-Powered RAG Chatbot for Project Documentation

A versatile Retrieval-Augmented Generation (RAG) chatbot designed to document and answer questions about **any software project** that include source files and document collection of various types.
The [Bullet3 library](https://github.com/bulletphysics/bullet3) was used as a `testbed` - as an example and for a demo.

[!IMPORTANT]
Remember to copy config_win.py (or config_posix.py) to **_config.py_** and .env_example to **_.env_**. And modify those as needed for your setup.

[!NOTE]
I used **python 3.10.8** during development.

## Project Goals

AI Bullet aims to make your software project more accessible by providing:

- **Intelligent Documentation Search**: Query all sources and documentation using natural language instead of manual searching
- **Code Understanding**: Ask questions about your C++ source code with context-aware responses
- **Comprehensive Knowledge Base**: Combines documentation, source code, and examples into a unified searchable database
- **Developer Productivity**: Reduce time spent searching through documentation and source code
- **Learning Support**: Help developers to study your code base through interactive Q&A

The system uses OpenAI's embedding models and ChromaDB for vector storage, enabling semantic search across multiple types of content including PDFs, documentation files, and C++ source code.

## Features

- **Real-time streaming** of LLM output via Server-Sent Events (SSE)
- **Retrieval from multiple sources** (source code + documentation)
- **Score-based fusion** for combining retrieval results from multiple collections
- **Optional Maximal Marginal Relevance (MMR)** for result diversification
- **Optional LLM-based re-ranking** using GPT models
- **Token-budgeted context** building with configurable limits
- **Dual knowledge modes**: RAG-only (context-based) or full LLM knowledge
- **Per-user session management** with automatic chat history persistence
- **Automatic source citation** in Markdown with clickable file links
- **LaTeX math rendering** support with KaTeX (four delimiter types)
- **OS-agnostic path system** for cross-platform database portability
- **HTTP Basic authentication** with user/password management
- **Modern async architecture** with FastAPI and OpenAI SDK

---

## Project Structure

### Core Application Files

- **app.py**: Main FastAPI application serving the chat UI, managing sessions, streaming responses, and retrieving relevant snippets from ChromaDB
- **web/index.html**: Front-end UI (Jinja2 template) for chat interaction, streaming display, markdown rendering, LaTeX math, and source link handling
- **retriever.py**: RAG retrieval engine handling query embedding, multi-collection retrieval, fusion, optional MMR/LLM re-ranking, context construction, and message building
- **updatedb_code.py**: Script for ingesting C++ source code files into ChromaDB with syntax-aware parsing
- **updatedb_docs.py**: Script for ingesting documentation files (PDF, DOCX, Markdown, text) into ChromaDB
- **config.py**: Central configuration for paths, collections, model parameters, and telemetry settings
- **path_utils.py**: OS-agnostic path encoding/decoding system for cross-platform database portability
- **my_logger.py**: Centralized logging configuration
- **migrate_paths.py**: Migration tool for converting legacy databases to OS-agnostic path format

---

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
1. **Install `uv` if not already installed:**
```bash
pip install uv
```

2. **Install required packages:**
```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```
> **Tip:** Use `uv` istead of just `pip`, uv runs much faster.

### Environment Setup

1. **Create your private .env environment file:**
```bash
cp .env_example .env
```
> **Warning:** Do not commit your `.env` file to GitHub or any public repository. Keep it secret and out of version control.

2. **Configure your settings in `.env`:**
```bash
# OpenAI API Configuration
OPENAI_API_KEY="your-openai-api-key-here"

# User database - format: username:password,username:password
USERS_DB=admin:12345,user01:demo

# (Optional) Other environment overrides go here
```

3. **Configure paths in `config.py`:**
```python
EMBEDDING_MODEL = 'text-embedding-3-small'  # Or text-embedding-ada-002
CHROMA_DB_DIR = 'chroma_store/'
DOCUMENTS_PATH = '~/work/rag_data/bullet3/docs'   # Path to Bullet3 docs
SOURCES_PATH = '~/work/rag_data/bullet3/src'      # Path to Bullet3 source
EXAMPLES_PATH = "~/work/rag_data/bullet3/examples"  # Path to examples
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
# Telemetry flags are now controlled here instead of .env
ANONYMIZED_TELEMETRY = False
CHROMA_TELEMETRY = False
```
> **Note:** Ensure that `DOCUMENTS_PATH`, `SOURCES_PATH`, and `EXAMPLES_PATH` in `config.py` correctly correspond to the folders in your cloned Bullet3 repository. Use provided config_win.py (Windows) or config_posix.py (Linux) as template for your config.py.

4. **Set up your Bullet3 data** :
```bash
# Clone Bullet3 (optional - for full functionality)
git clone https://github.com/bulletphysics/bullet3.git ~/work/rag_data/bullet3
```

#### Database Population

**Process documentation files:**
```bash
# Process PDFs, DOCX, Markdown, and text files
python updatedb_docs.py
```

**Process C++ source code:**
```bash
# Parse and index C++ files with syntax awareness
python updatedb_code.py
```

**Migrate existing database to OS-agnostic paths (if needed):**
```bash
# Preview changes before applying
python migrate_paths.py --dry-run

# Apply migration to convert legacy paths to OS-agnostic format
python migrate_paths.py
```

**Test retrieval standalone:**
```bash
# Test retrieval without running the web server
python retriever.py "How do I create a rigid body in Bullet3?"
```

#### Running the Web Interface

**Standard server (recommended):**
```bash
python app.py
```

**Using uvicorn directly:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Or for localhost testing:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Access the application:**
- Open your browser to `http://localhost:8000`
- Log in with credentials configured in `.env` file (USERS_DB)
- Start asking questions about your project

#### Direct API Usage

**Using the retriever programmatically:**
```python
from retriever import create_retriever

# Create retriever instance with default configuration
retriever = create_retriever()

# Retrieve relevant context for a question
question = "How do I create a rigid body in Bullet3?"
hits = retriever.retrieve(question)
context, sources = retriever.build_context(hits)

# Build messages for OpenAI API
messages = retriever.build_messages(question, context, use_full_knowledge=False)
print(f"Context: {context}")
print(f"Sources: {sources}")
```

### Configuration Options

#### Web Interface Settings

- **Host/Port**: Modify in `app.py` or use uvicorn parameters (default: `0.0.0.0:8000`)
- **Authentication**: Configure users in `.env` file using `USERS_DB` (format: `username:password,username:password`)
- **Session Management**: Automatic per-user session tracking with in-memory storage
- **Session Cleanup**: Automatically removes old sessions when count exceeds 100 (keeps 50 most recent)
- **Chat History**: All conversations auto-saved to `saved_chats/` directory with metadata

#### Retrieval Settings (RetrieverConfig in retriever.py)

- `use_mmr`: Enable Maximal Marginal Relevance for diversity (default: `True`)
- `use_llm_rerank`: Use LLM for re-ranking instead of MMR (default: `False`)
- `max_context_tokens`: Token budget for retrieved context (default: `6000`)
- `max_snippets`: Maximum number of snippets to include (default: `12`)
- `system_template`: System prompt for RAG-only mode (context-based responses)
- `system_template_full`: System prompt for full knowledge mode (unrestricted LLM)

#### Database Settings (config.py)

- `EMBEDDING_MODEL`: Choose OpenAI embedding model (e.g., `text-embedding-3-small`)
- `CHUNK_SIZE`: Size of text chunks for processing (default: `800` tokens)
- `CHUNK_OVERLAP`: Overlap between consecutive chunks (default: `50` tokens)
- `CHROMA_DB_DIR`: Directory for ChromaDB vector database storage
- `DOCUMENTS_PATH`: Path to documentation files for indexing
- `SOURCES_PATH`: Path to source code files for indexing
- `EXAMPLES_PATH`: Path to example files for indexing
- `ANONYMIZED_TELEMETRY`: Disable OpenAI telemetry (default: `False`)
- `CHROMA_TELEMETRY`: Disable ChromaDB telemetry (default: `False`)

### API Endpoints

#### GET /
Main chat interface with authentication required. Returns HTML page with chat UI.

Query parameters:
- `session_id` (optional): Resume existing session

#### POST /chat
Streaming chat endpoint using Server-Sent Events (SSE).

Request body:
```json
{
  "message": "Your question here",
  "model": "gpt-4o-mini",
  "use_full_knowledge": false
}
```

Query parameters:
- `session_id` (optional): Session identifier for conversation continuity

Response: SSE stream with JSON chunks containing `content` or `error` fields

#### POST /sessions/new
Create a new chat session for the current user.

Response:
```json
{
  "session_id": "username_abc12345",
  "chat_history": [],
  "model": "gpt-4o-mini"
}
```

#### GET /health
Health check endpoint.

Response:
```json
{
  "status": "healthy",
  "openai_configured": true,
  "version": "0.1.2"
}
```

#### GET /models
List available OpenAI models.

Response:
```json
{
  "models": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-5-mini"],
  "default_model": "gpt-4o-mini"
}
```

### Knowledge Modes

The application supports two distinct modes for generating responses:

#### RAG-Only Mode (default, use_full_knowledge=false)
- Responses generated **only** from retrieved documentation and code context
- System prompt restricts LLM to use only provided CONTEXT
- Best for ensuring factual accuracy from indexed materials
- LLM will say "I don't know" if information is not in the context

#### Full Knowledge Mode (use_full_knowledge=true)
- LLM can use its full training knowledge in addition to retrieved context
- Useful for general questions or when RAG context is insufficient
- May provide broader answers but with potential for hallucination
- Context still provided for grounding, but not strictly required

### OS-Agnostic Path System

AI Bullet uses an innovative path encoding system that makes ChromaDB databases portable across different operating systems (Windows, Linux, macOS).

#### How It Works

**Path Encoding**: Absolute paths are converted to variable-based format when storing in ChromaDB:
- Windows: `D:\Work\rag_data\bullet3\docs\file.pdf` → `{DOCS}/file.pdf`
- Linux: `/home/user/rag_data/bullet3/docs/file.pdf` → `{DOCS}/file.pdf`

**Path Decoding**: Variable-based paths are converted back to absolute paths at runtime based on your `config.py` settings.

**Supported Variables**:
- `{DOCS}` - Maps to `DOCUMENTS_PATH` in config.py
- `{SRC}` - Maps to `SOURCES_PATH` in config.py
- `{EXAMPLES}` - Maps to `EXAMPLES_PATH` in config.py

**Benefits**:
- Share ChromaDB databases between Windows and Linux systems
- Move databases without re-indexing all documents
- Collaborate across different development environments

**Legacy Format**: The old `$VAR$` format is still supported for backward compatibility, but new databases use `{VAR}` format to avoid collisions with LaTeX math notation.

#### Migrating Existing Databases

If you have an existing ChromaDB database with hardcoded absolute paths:

```bash
# Preview what will be changed
python migrate_paths.py --dry-run

# Apply the migration
python migrate_paths.py
```

This will convert all absolute paths in your database to the OS-agnostic format.

### Basic Workflow

1. **Preparation Phase:**
   - Set up OpenAI API key in `.env` file
   - Configure paths to your project data in `config.py`
   - Install dependencies with `uv pip install -r requirements.txt`
   - Configure user authentication in `.env` (USERS_DB)

2. **Indexing Phase:**
   - Run `python updatedb_docs.py` to process documentation
   - Run `python updatedb_code.py` to process source code
   - Verify ChromaDB collections are created in `chroma_store/` directory
   - (Optional) Run `python migrate_paths.py` if migrating existing database

3. **Usage Phase:**
   - (Optional) Run `python retriever.py "your question"` for standalone testing
   - Start web interface with `python app.py`
   - Access at `http://localhost:8000`
   - Log in with configured credentials from `.env`
   - Select knowledge mode (RAG-only or Full Knowledge)
   - Ask questions about your project

### Example Questions

Once set up, you can ask questions like:
- "How do I create a rigid body in Bullet3?"
- "What are the basic data types in Bullet Physics?"
- "Explain the constraint solver implementation"
- "How does collision detection work in Bullet3?"
- "What examples are available for soft body dynamics?"
- "Where is class btMotionState defined?"
- "Describe DeformableDemo"
- "What value of timeStep is recommended for the integration?"

### Session Management

- **Per-User Sessions**: Each user maintains their own chat history
- **Session Persistence**: Conversations are automatically saved to `saved_chats/` directory
- **Session Format**: `username_DD_Mon_YYYY_HH_MM_SS.json`
- **Session Metadata**: Includes creation time, last update time, model used, and `use_full_knowledge` flag for each message
- **New Chat**: Click "New Chat" button to start a fresh conversation
- **Session Cleanup**: Old sessions automatically cleaned up when count exceeds 100

### Saved Chat Format

Chat histories are saved in JSON format with the following structure:

```json
{
  "created_on": "12-Oct-2025",
  "last_update_on": "12-Oct-2025 15:32:04",
  "session_id": "admin_p6gukreq",
  "username": "admin",
  "messages": [
    {
      "role": "system",
      "content": "You are a precise assistant..."
    },
    {
      "role": "user",
      "content": "Describe DeformableDemo"
    },
    {
      "role": "assistant",
      "model": "gpt-4o-mini",
      "use_full_knowledge": false,
      "content": "### DeformableDemo Overview\n\n..."
    }
  ]
}
```

**Logs and Debugging:**
- Web interface logs appear in console when running `app.py`
- All chat conversations are recorded in the `saved_chats/` folder
- Use `python retriever.py "question"` for isolated retrieval testing
- Check `/health` endpoint to verify OpenAI configuration

For additional support, please check the project's issue tracker or create a new issue with detailed error information.

## License

This project is licensed under the [MIT License](LICENSE).

