# AI Bullet: AI-Powered RAG Chatbot for Project Documentation

A versatile Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about **one or many collections of your documents and source files**.

The [Bullet3 library](https://github.com/bulletphysics/bullet3) was used as a testbed for demonstration purposes.

AI Bullet can operate in either **online** or **offline** (local) mode.\
For simplicity, we use the OpenAI "GPT-4o-mini" model for online mode.\
For offline (local) mode, we use Ollama with the qwen3:4b-instruct-2507-fp16 LLM.

### ⚠️ Important
Examine **config.py** - all important settings are located there.\
The **USE_OPENAI** flag controls online or offline mode.\
You can set your preferred LLM models and specify your documents and their locations in **GLOBAL_RAGDATA_MAP**.

### ⚠️ Important
Remember to copy and modify .env_example to **.env**. It should contain your private data (like OpenAI API key and usernames/passwords).

### ℹ️ Note
We used **Python 3.10.8 and 3.11.9** during development.

## Project Goals

AI Bullet aims to make your software project more accessible by providing:

- **Intelligent Documentation Search**: Query all sources and documentation using natural language instead of manual searching
- **Code Understanding**: Ask questions about your C++ source code with context-aware responses
- **Comprehensive Knowledge Base**: Combines documentation, source code, and examples into a unified searchable database
- **Developer Productivity**: Reduce time spent searching through documentation and source code
- **Learning Support**: Help developers to study your code base through interactive Q&A

The system uses OpenAI or Qwen embedding models and ChromaDB for vector storage, enabling semantic search across multiple types of content including PDFs, documentation files, and C++ source code.

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
- **updatedb_all.py**: Batch script for updating all collections defined in GLOBAL_RAGDATA_MAP, processing both code and documentation
- **db_tools_simple.py**: Database management utility for viewing collection info, removing collections, and cleaning up ChromaDB
- **config.py**: Central configuration for paths, collections, model parameters, and telemetry settings
- **path_utils.py**: OS-agnostic path encoding/decoding system for cross-platform database portability
- **my_logger.py**: Centralized logging configuration
---

## Installation

### Prerequisites

- If used online with OpenAI, you will need an API key

### Python Dependencies

1. **Clone the repository:**
```bash
git clone https://github.com/alxliv/ai_bullet.git
cd ai_bullet
```

2. **Install `uv` if not already installed:**
```bash
pip install uv
```

3. **Install required packages:**
```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```
> **Tip:** Use `uv` instead of just `pip`, uv runs much faster.

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

3. **Configure `config.py`:**
   - Set **USE_OPENAI** flag for online/offline mode
   - Configure **GLOBAL_RAGDATA_MAP** with your document locations
   - Set embedding and LLM models according to your needs
   - See the provided config.py as an example

#### Database Population

**Process documentation files:**

Configure your **SYMBOLIC_NAMES** in **config.py** **GLOBAL_RAGDATA_MAP**.\
See default config.py provided as an example.

```bash
# Process PDFs, DOCX, Markdown, and text files
python updatedb_docs.py <SYMBOLIC_NAME>
```

**Process C/C++ source code:**
```bash
# Parse and index C++ files with syntax awareness
python updatedb_code.py <SYMBOLIC_NAME>
```

**Test retrieval standalone:**
```bash
# Test retrieval without running the web server
python retriever.py "How do I create a rigid body in Bullet3?"
```

#### Running the Web Interface

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

**Benefits**:
- Share ChromaDB databases between Windows and Linux systems
- Move databases without re-indexing all documents
- Collaborate across different development environments

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

3. **Usage Phase:**
   - (Optional) Run `python retriever.py "your question"` for standalone testing
   - Start web interface with `uvicorn app:app --host 0.0.0.0 --port 8000`
   - Access at `http://localhost:8000`
   - Log in with configured credentials from `.env`
   - Ask questions about your project

---

## Database Maintenance

### Managing Collections with db_tools_simple.py

The `db_tools_simple.py` utility provides commands for inspecting and managing your ChromaDB collections:

**View database information and all collections:**
```bash
python db_tools_simple.py info
```
This displays:
- Database path and size
- List of all collections with their types (CODE/DOC)
- Number of records in each collection
- Total records across all collections

**Remove a specific collection:**
```bash
# Interactive mode with confirmation prompt
python db_tools_simple.py remove <COLLECTION_NAME>

# Force removal without confirmation
python db_tools_simple.py remove <COLLECTION_NAME> --force
```

**Clean entire database:**
```bash
# Interactive mode - requires typing 'DELETE ALL' to confirm
python db_tools_simple.py clean

# Force clean without confirmation
python db_tools_simple.py clean --force
```

### Adding Documents to Existing Collections

To add new documents or folders to an existing collection, simply run the appropriate update script:

**Add documentation files:**
```bash
# Re-run for the same SYMBOLIC_NAME - only new files will be added
python updatedb_docs.py <SYMBOLIC_NAME>
```

The script automatically:
- Scans the configured path for files
- Computes content hashes to detect new or modified documents
- Skips unchanged documents (de-duplication)
- Adds only new chunks to the collection

**Add source code files:**
```bash
# Re-run for the same SYMBOLIC_NAME - only new files will be added
python updatedb_code.py <SYMBOLIC_NAME>
```

The script automatically:
- Parses C/C++ files in the configured directory
- Detects new or modified source files
- Skips unchanged code (de-duplication)
- Indexes only new code chunks

### Updating Collections After Changes

**When you modify documents or code:**

Simply re-run the appropriate update script. The system will:
1. Compute hashes for all files
2. Compare with existing records in ChromaDB
3. Add only new or modified content
4. Preserve unchanged records

**Batch update all collections:**
```bash
# Update all collections defined in GLOBAL_RAGDATA_MAP
python updatedb_all.py
```

### Removing Documents from Collections

**To remove specific documents:**

Currently, the best approach is to:
1. Remove the collection: `python db_tools_simple.py remove <COLLECTION_NAME>`
2. Remove unwanted files/folders from your source directory
3. Rebuild the collection: `python updatedb_docs.py <SYMBOLIC_NAME>` or `python updatedb_code.py <SYMBOLIC_NAME>`

**To reorganize collections:**

1. Update paths in `GLOBAL_RAGDATA_MAP` in `config.py`
2. Remove old collection if needed
3. Run update scripts to create new collections with updated paths

### Collection Types

Collections are automatically identified by type:
- **CODE**: Source code collections (C/C++, Python, etc.) - parsed with syntax awareness
- **DOC**: Documentation collections (PDF, DOCX, Markdown, text) - processed as text documents

The type is determined by the `RAGType` specified in `GLOBAL_RAGDATA_MAP`:
- `RAGType.SRC` → CODE collection
- `RAGType.DOC` → DOC collection

---

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
