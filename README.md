# AI Bullet: AI-Powered RAG Chat UI for Project Documentation

A versatile Retrieval-Augmented Generation (RAG) chatbot designed to document and answer questions about **any software project** that includes source files and document collection of various types.
The [Bullet3 library](https://github.com/bulletphysics/bullet3) was used as a `testbed` - as an example and for a demo.

## Project Goals

AI Bullet aims to make your software project more accessible by providing:

- **Intelligent Documentation Search**: Query all sources and documentation using natural language instead of manual searching
- **Code Understanding**: Ask questions about your C++ source code with context-aware responses
- **Comprehensive Knowledge Base**: Combines documentation, source code, and examples into a unified searchable database
- **Developer Productivity**: Reduce time spent searching through documentation and source code
- **Learning Support**: Help developers to study your code base through interactive Q&A

The system uses OpenAI's embedding models and ChromaDB for vector storage, enabling semantic search across multiple types of content including PDFs, documentation files, and C++ source code.

## Features

- **Real-time streaming** of LLM output via SSE (Work in progress)
- **Retrieval from multiple sources** (source code + documentation)
- **Reciprocal Rank Fusion (RRF)** + optional LLM re-rank (Currently score-based fusion is implemented, no RRF)
- **Token-budgeted context**
- **Configurable system prompts** (context-only vs. full-knowledge)
- **Automatic source citation** in Markdown
- **Simple web-based chat UI**

---

## Project Structure

- **webgui.py**: FastAPI backend serving the chat UI, managing sessions, streaming responses, and retrieving relevant snippets from ChromaDB.
- **chat.html**: Front-end UI (HTML + vanilla JS) for chat interaction, streaming display, markdown rendering, and source link handling.
- **retriever.py**: Handles query embedding, multi-collection retrieval, fusion, context construction, and message building.
- **updatadb_code.py**: Script for ingesting source code files into ChromaDB.
- **updatedb_docs.py**: Script for ingesting documentation files into ChromaDB.
- **config.py**: Central configuration for paths, collections, and model/API parameters.

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

# ChromaDB Telemetry Suppression
ANONYMIZED_TELEMETRY=False
CHROMA_TELEMETRY=False
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

** You can also run retriever.py as standalone:
```bash
# Parse and index C++ files with syntax awareness
python retriever.py
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
Or for localhost testing:
```bash
uvicorn webgui:app --host 127.0.0.1 --port 8000 --reload
```

**Debug server (for testing math LaTex display):**
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
   - Run retriever.py as standalone, for testing
   - Start local web interface with `uvicorn webgui:app --host 127.0.0.1 --port 8000`
   - Access at `http://localhost:8000`
   - Log in with configured credentials
   - Ask questions about Bullet Physics

### Example Questions

Once set up, you can ask questions like:
- "How do I create a rigid body in Bullet3?"
- "What are the basic data types in Bullet Physics?"
- "Explain the constraint solver implementation"
- "How does collision detection work in Bullet3?"
- "What examples are available for soft body dynamics?"
- "Where class btMotionState is defined?"


**Logs and Debugging:**
- Web interface logs appear in console when running `webgui.py`
- Use `debug_server.py` for isolated testing (Only tot test math formulas display)
- Messages for all sessions are recorded in the `saved_chats` folder

For additional support, please check the project's issue tracker or create a new issue with detailed error information.

## License

This project is licensed under the [MIT License](LICENSE).

