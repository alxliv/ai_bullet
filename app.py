#!/usr/bin/env python3
"""
OpenAI Chat Client with Math and Markdown Rendering

A FastAPI application that provides a web interface for mathematical conversations
with AI, featuring LaTeX rendering and streaming responses.

Required packages: pip install fastapi uvicorn openai python-dotenv
"""

import os
from pathlib import Path
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from dotenv import load_dotenv
from my_logger import setup_logger;
from retriever import create_retriever, ask_llm
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH
import re, random, string, json

version = "0.1.1"

logger = setup_logger()
# Example usage
logger.debug("Debugging message")
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error!")

load_dotenv()

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-5-mini"
]

DEFAULT_MODEL = "gpt-4o-mini"

logger.info(f"Available models: {AVAILABLE_MODELS}")
logger.info(f"Default model: {DEFAULT_MODEL}")

SYSTEM_PROMPT = (
    "You are a helpful mathematics tutor. Use LaTeX notation for all "
    "mathematical expressions. For inline math use $...$ and for display "
    "math use $$...$$.  Provide clear, step-by-step explanations."
)
# Template configuration
templates = Jinja2Templates(directory="web")
HTML_FILE_PATH = Path("web") / "index.html"

# Global client instance
_openai_client: Optional[AsyncOpenAI] = None

retriever = create_retriever()

class Config:
    """Application configuration"""

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)


def get_openai_client() -> Optional[AsyncOpenAI]:
    """Get the OpenAI client instance"""
    return _openai_client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager"""
    global _openai_client
    config = Config()

    # Initialize OpenAI client if API key is available
    if config.has_openai_key:
        _openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OpenAI API key not found - client not initialized")

    yield

    # Cleanup
    if _openai_client:
        await _openai_client.close()
        logger.info("OpenAI client closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="OpenAI Math Chat",
    description="A web interface for mathematical conversations with AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Restrict to local origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
)

DOCS_ROOT = os.path.expanduser(DOCUMENTS_PATH)
SRC_ROOT = os.path.expanduser(SOURCES_PATH)
EXAMPLES_ROOT = os.path.expanduser(EXAMPLES_PATH)

# FIXME!!! should fix how path are stored in ChromaDB (I use chromadb that was created on Oracle VPS, so there is /home/ubuntu/work...)
# Should fix the way location of data is stored in CHROMADB!!!
print("Remember to fix how data references are stored in DB!")

DOCS_ROOT_REF = "/home/ubuntu/work/rag_data/bullet3/docs" # os.path.expanduser(DOCUMENTS_PATH)
SRC_ROOT_REF = "/home/ubuntu/work/rag_data/bullet3/src"  # os.path.expanduser(SOURCES_PATH)
EXAMPLES_ROOT_REF = "/home/ubuntu/work/rag_data/bullet3/examples" # os.path.expanduser(EXAMPLES_PATH)

REPLACEMENTS = {}
for root, alias in (
    (DOCS_ROOT_REF,    "/docs"),
    (SRC_ROOT_REF,     "/src"),
    (EXAMPLES_ROOT_REF, "/examples"),
):
    # map both the absolute path and the same string without leading slash
    REPLACEMENTS[root] = alias
    REPLACEMENTS[root.lstrip(os.sep)] = alias

# compile a single pattern matching any of those keys
RE_ROOTS_PATTERN = re.compile("|".join(re.escape(k) for k in REPLACEMENTS))

# Mount static files
app.mount("/static", StaticFiles(directory="web"), name="static")

app.mount("/docs", StaticFiles(directory=DOCS_ROOT), name="docs")
app.mount("/src", StaticFiles(directory=SRC_ROOT), name="src")
app.mount("/examples", StaticFiles(directory=EXAMPLES_ROOT), name="examples")

# _username and _session_id are globals for now, but will be replaced by proper user/session magagement later
_username = ""
_session_id = ""

# Session-based message storage - each client gets their own chat history
user_sessions = {}  # {session_id: {"messages": [...], "model": "..."}}
current_model = ""  # Will be set to first available model

def generate_short_id(length=8):
    """Generate a random identifier of given length."""
    # Create a sequence of uppercase letters and digits
    characters = string.ascii_lowercase + string.digits
    # Choose 'length' characters randomly
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id


def get_or_create_session(username, session_id=None):
    """Get or create a user session"""

    if session_id not in user_sessions:
        session_id = username+'_'+generate_short_id()
        system_prompt = retriever.cfg.system_template
        user_sessions[session_id] = {
            "messages": [{"role": "system", "content": system_prompt}],
            "model": current_model,
            "username": username
        }
    return session_id

def save_chat_history(session_id):
    """Save current chat history to a JSON file"""
    if session_id not in user_sessions:
        return None

    session_data = user_sessions[session_id]
    messages = session_data["messages"]

    if len(messages) <= 1:  # Only system message, nothing to save
        return None

    # Create chats directory if it doesn't exist
    os.makedirs("saved_chats", exist_ok=True)
    filename = f"saved_chats/{session_id}.json"

    # Prepare chat data
    chat_data = {
        "timestamp": datetime.now().strftime("%H:%M:%S GMT"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "session_id": session_id,
        "message_count": len(messages) - 1,  # Exclude system message
        "messages": messages
    }

    # Save to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return None

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="The user's message")
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI model to use for the response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the derivative of x^2?",
                "model": "gpt-4o-mini"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")


class StreamResponse(BaseModel):
    """Stream response model"""
    content: Optional[str] = Field(None, description="Response content")
    error: Optional[str] = Field(None, description="Error message")


class ModelsResponse(BaseModel):
    """Response model for available OpenAI models"""
    models: list[str] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model to use")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Get or create session
    session_id = get_or_create_session(_username, _session_id)

    # Get recent chat history if it exists (exclude system messages)
    session_data = user_sessions[session_id]
    all_messages = session_data["messages"]
    user_messages = [msg for msg in all_messages if msg["role"] != "system"]
    chat_history = user_messages[-10:] if len(user_messages) > 0 else []
    examples = "What is Jacobi solver?<br>" +\
        "Describe DeformableDemo<br>" +\
        "What value of timeStep is recommended for the integration?<br>" +\
        "Explain struct LuaPhysicsSetup<br>" +\
        "How collisions are calculated?"

    """Serve the main chat interface"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "version": version,
            "username": _username
        })
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load frontend"
        )

async def stream_openai_response(message: str, model: str) -> AsyncGenerator[str, None]:
    """Stream responses from OpenAI API using Server-Sent Events format"""
    client = get_openai_client()

    if not client:
        error_response = ErrorResponse(error="OpenAI API key not configured")
        yield f"data: {error_response.model_dump_json()}\n\n"
        return

    try:
        hits = retriever.retrieve(message)
        ctx, sources = retriever.build_context(hits)
        messages = retriever.build_messages(message, ctx)

        # Use max_completion_tokens for GPT-5 models, max_tokens for others
        token_param = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
        max_tokens = 8000 if model.startswith("gpt-5") else 3000
        stream_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            token_param: max_tokens
        }

        stream = await client.chat.completions.create(**stream_params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response = StreamResponse(content=content, error=None)
                # await asyncio.sleep(0.2)  # Simulate network delay
                yield f"data: {response.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
        logger.info("Response streaming completed successfully")

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        error_response = ErrorResponse(error=f"API error: {str(e)}")
        yield f"data: {error_response.model_dump_json()}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle streaming chat requests with OpenAI API"""

   # ensure we have a session
    session_id   = get_or_create_session(_username, _session_id)
    session_data = user_sessions[session_id]

    client = get_openai_client()

    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured"
        )

    if request.model not in AVAILABLE_MODELS:
        logger.warning(f"Unsupported model requested: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not supported"
        )

    # record user turn
    session_data["messages"].append({"role": "user", "content": request.message})

    logger.info(f"Chat request: model={request.model}, message_length={len(request.message)}")

    return StreamingResponse(
        stream_openai_response(request.message, request.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    client = get_openai_client()
    return {
        "status": "healthy",
        "openai_configured": client is not None,
        "version": "1.0.0"
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """Return the list of available OpenAI models"""
    if not AVAILABLE_MODELS:
        logger.error("No OpenAI models configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models configured"
        )

    if DEFAULT_MODEL not in AVAILABLE_MODELS:
        logger.warning("DEFAULT_MODEL is not in AVAILABLE_MODELS; using default fallback")

    sorted_models = sorted(AVAILABLE_MODELS)

    return ModelsResponse(
        models=sorted_models,
        default_model=DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODELS else sorted_models[0]
    )

def print_startup_info(config: Config):
    """Print application startup information"""
    print("=" * 60)
    print("OpenAI Math Chat Server")
    print("=" * 60)

    if not config.has_openai_key:
        print("WARNING: OPENAI_API_KEY not found!")
        print("Create a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("The server will run but chat functionality will be disabled.")
    else:
        print("OpenAI API key loaded")

    print(f"Starting server at http://{config.host}:{config.port}")
    print("Health check: /health")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Example queries
    print("\nTry asking:")
    print('- "Explain Laplace transform"')
    print('- "What is Heaviside step function?"')
    print('- "Derive the quadratic formula"')
    print('- "What is harmonic oscillator?"')
    # what is the Lorentzian function?
    # Explain The Cauchy probability distribution with examples

if __name__ == "__main__":
    import uvicorn

    config = Config()
    print_startup_info(config)

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped gracefully")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"\nError: {e}")
        print(f"Try: uvicorn app:app --host {config.host} --port {config.port}")
