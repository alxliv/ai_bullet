#!/usr/bin/env python3
"""
OpenAI Chat Client with Math and Markdown Rendering

A FastAPI application that provides a web interface for mathematical conversations
with AI, featuring LaTeX rendering and streaming responses.

Required packages: pip install fastapi uvicorn openai python-dotenv
"""
import os, time, random, string, json, re
import secrets

from pathlib import Path
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Request, Depends, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from dotenv import load_dotenv
from my_logger import setup_logger;
from retriever import create_retriever
from path_utils import DOCS_ROOT, SRC_ROOT, EXAMPLES_ROOT


version = "0.1.5"
title="AI Bullet: AI-Powered Q & A"

logger = setup_logger()
# Example usage
# logger.debug("Debugging message")
# logger.info("Information message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical error!")

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

# Add after the existing imports and before app = FastAPI()
security = HTTPBasic()

def load_users_from_env():
    """Load users from environment variable"""
    users_str = os.getenv("USERS_DB", "")
    users = {}

    if users_str:
        try:
            # Parse format: username:password,username:password
            for user_pair in users_str.split(','):
                if ':' in user_pair:
                    username, password = user_pair.split(':', 1)  # Split only on first ':'
                    users[username.strip()] = password.strip()
        except Exception as e:
            logger.error(f"Error parsing USERS_DB: {e}")
    return users

# Load users from environment
USERS = load_users_from_env()

if (len(USERS)==0):
    logger.error("No users are defined. Please add at least one user to the .env file.")
    exit(-2)
logger.info(f"Loaded {len(USERS)} users from environment")

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify user credentials"""
    username = credentials.username
    password = credentials.password
    if username in USERS and secrets.compare_digest(password, USERS[username]):
        logger.info(f"Auth OK for: {username}")
        return username
    logger.warning(f"Not authorized: {username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

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
    title=title,
    description="A web interface for mathematical conversations with AI",
    version=version,
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

# Path roots are now imported from path_utils (DOCS_ROOT, SRC_ROOT, EXAMPLES_ROOT)
# Paths in ChromaDB are now stored using OS-agnostic variable encoding ($DOCS$, $SRC$, $EXAMPLES$)
# and decoded at runtime using path_utils.decode_path()

# Mount static files
app.mount("/static", StaticFiles(directory="web"), name="static")

app.mount("/docs", StaticFiles(directory=DOCS_ROOT), name="docs")
app.mount("/src", StaticFiles(directory=SRC_ROOT), name="src")
app.mount("/examples", StaticFiles(directory=EXAMPLES_ROOT), name="examples")

# _username and _session_id are globals for now, but will be replaced by proper user/session magagement later
_username = "demo"
_session_id = ""

# Session-based message storage - each client gets their own chat history
user_sessions = {}  # {session_id: {"messages": [...], "model": "...", "username": "..."}}
user_last_session: dict[str, str] = {}  # Track most recent session per username
current_model = DEFAULT_MODEL  # Track most recently used model globally

def generate_short_id(length=8):
    """Generate a random identifier of given length."""
    # Create a sequence of uppercase letters and digits
    characters = string.ascii_lowercase + string.digits
    # Choose 'length' characters randomly
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id


def build_chat_filename(username: str, dt: Optional[datetime] = None) -> str:
    """Return a filesystem-safe path for storing a chat transcript."""
    dt = dt or datetime.now()
    timestamp = dt.strftime("%d_%b_%Y_%H_%M_%S").lower()
    safe_username = re.sub(r"[^\w.-]", "_", username)
    return str(Path("saved_chats") / f"{safe_username}_{timestamp}.json")


def get_or_create_session(
    username: str,
    session_id: Optional[str] = None,
    *,
    force_new: bool = False
) -> str:
    """
    Fetch an existing session for a user or create a brand-new one.

    Args:
        username: Authenticated username.
        session_id: Optional session identifier supplied by the client.
        force_new: When True, always generate a fresh session for the user.

    Returns:
        A session identifier that is guaranteed to belong to the user.
    """
    if session_id and session_id in user_sessions:
        existing = user_sessions[session_id]
        if existing.get("username") == username:
            user_last_session[username] = session_id
            return session_id
        logger.warning(
            "Session %s does not belong to user %s; creating a new session",
            session_id,
            username,
        )
        session_id = None

    if not force_new:
        prior_session = user_last_session.get(username)
        if prior_session and prior_session in user_sessions:
            return prior_session

    if not session_id or session_id not in user_sessions or force_new:
        session_id = f"{username}_{generate_short_id()}"
        system_prompt = retriever.cfg.system_template
        os.makedirs("saved_chats", exist_ok=True)
        save_path = build_chat_filename(username)
        Path(save_path).touch(exist_ok=True)
        now = datetime.now()
        user_sessions[session_id] = {
            "messages": [{"role": "system", "content": system_prompt}],
            "model": current_model or DEFAULT_MODEL,
            "username": username,
            "save_path": save_path,
            "created_at": now,
            "last_updated_at": now,
        }
        user_last_session[username] = session_id

    return session_id


def get_session_messages(session_id: str) -> list[dict]:
    """Return non-system messages for a session."""
    session_data = user_sessions.get(session_id, {})
    messages = session_data.get("messages", [])
    return [msg for msg in messages if msg.get("role") != "system"]

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

    save_path = session_data.get("save_path")
    if not save_path:
        logger.error("Session %s missing save_path; cannot persist chat history", session_id)
        return None
    save_path = str(save_path)

    # Prepare chat data
    created_at = session_data.get("created_at")
    if not isinstance(created_at, datetime):
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        session_data["created_at"] = created_at

    last_updated_at = datetime.now()
    session_data["last_updated_at"] = last_updated_at

    chat_data = {
        "created_on": created_at.strftime("%d-%b-%Y"),
        "last_update_on": last_updated_at.strftime("%d-%b-%Y %H:%M:%S"),
        "session_id": session_id,
        "username": session_data.get("username"),
        "messages": messages
    }

    # Save to file
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        return save_path
    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return None


def normalize_repo_links(markdown_text: str) -> str:
    """Ensure markdown links display the same repo-relative path as their destination."""

    def _replace(match: re.Match) -> str:
        text = match.group(1)
        url = match.group(2)

        if text.strip() == url.strip():
            return match.group(0)

        if url.startswith("/"):
            return f"[{text.strip() or url}]({url})"

        normalized_text = text.strip()
        normalized_url = url.strip()

        if normalized_text.startswith("/"):
            return f"[{normalized_text}]({normalized_text})"

        return match.group(0)

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace, markdown_text)


def save_response(session_id: str, model: str, user_message: str, assistant_message: str, use_full_knowledge: bool = False) -> None:
    """Persist a completed user/assistant exchange to disk."""
    session_data = user_sessions.get(session_id)

    if not session_data:
        logger.warning(f"Cannot save response; session {session_id} not found")
        return

    session_messages = session_data["messages"]

    if session_messages and session_messages[-1].get("role") == "user":
        # Ensure latest user message is up to date before saving
        session_messages[-1]["content"] = user_message
    else:
        session_messages.append({"role": "user", "content": user_message})

    session_messages.append({
        "role": "assistant",
        "model": model,
        "use_full_knowledge": use_full_knowledge,
        "content": assistant_message
    })

    session_data["model"] = model

    saved_path = save_chat_history(session_id)

    if saved_path:
        logger.info(f"Saved chat history to {saved_path}")
    else:
        logger.warning(f"Failed to persist chat history for session {session_id}")

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="The user's message")
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI model to use for the response"
    )
    use_full_knowledge: bool = Field(
        default=False,
        description="Use full LLM knowledge instead of only RAG context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the derivative of x^2?",
                "model": "gpt-4o-mini",
                "use_full_knowledge": False
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
async def index(
    request: Request,
    session_id: Optional[str] = Query(default=None),
    username: str = Depends(authenticate_user),
):
    """Serve the main chat interface."""

    session_id = get_or_create_session(username, session_id)

    session_data = user_sessions.get(session_id)
    if not session_data:
        logger.error("Failed to load session %s for user %s", session_id, username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load chat session",
        )

    user_messages = get_session_messages(session_id)
    chat_history = user_messages[-10:] if user_messages else []
    examples = [
        "Describe DeformableDemo",
        "What value of timeStep is recommended for the integration?",
        "What is the Jacobi solver implementation?",
        "Explain btMotionState class definition",
        "How does collision detection work in Bullet3?",
        "Describe the constraint solver architecture",
        "Explain struct LuaPhysicsSetup",
        "How to compute the object AABBs?",
        "What types of constraints are available in Bullet3 and how do I create a hinge joint?"
    ]

    try:
        return templates.TemplateResponse("index.html", {
            "title": title,
            "request": request,
            "version": version,
            "username": username,
            "chat_history": chat_history,
            "selected_model": session_data["model"],
            "session_id": session_id,
            "username": username,
            "example_questions": examples
        })
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load frontend"
        )

def complete_response(combined_content: str, model: str, message: str, session_id: Optional[str] = None, use_full_knowledge: bool = False):
    """
    Called when the complete response has been received and accumulated.
    This function receives the full combined content from all chunks.

    Args:
        combined_content: The complete response text from all chunks combined
        model: The model used for generation
        message: The original user message/query
        use_full_knowledge: Whether full knowledge mode was used

    Returns:
        Processed content (with paths converted to web URLs)
    """
    logger.info(f"Complete response received ({len(combined_content)} chars)")
    logger.debug(f"Model: {model}")
    logger.debug(f"Original query: {message[:100]}...")  # First 100 chars of query
    logger.debug(f"Response preview: {combined_content[:200]}...")  # First 200 chars of response

    normalized_content = normalize_repo_links(combined_content)

    if session_id:
        save_response(session_id, model, message, normalized_content, use_full_knowledge)
    else:
        logger.warning("Session ID missing; skipping chat persistence")
        logger.debug("Normalized content preview: %s", normalized_content[:200])

    return normalized_content

    # You can add any post-processing here:
    # - Store complete response in database
    # - Perform analytics on the full response
    # - Extract structured information
    # - Generate summaries or metadata
    # - Trigger notifications or webhooks
    #
    # Note: Path conversion from {DOCS}/file.pdf to /docs/file.pdf
    # is now done in retriever.py at source_label() generation time,
    # so the LLM receives proper web URLs from the start

async def stream_openai_response(message: str, model: str, use_full_knowledge: bool = False, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Stream responses from OpenAI API using Server-Sent Events format"""
    client = get_openai_client()

    if not client:
        error_response = ErrorResponse(error="OpenAI API key not configured")
        yield f"data: {error_response.model_dump_json()}\n\n"
        return

    try:
        hits = retriever.retrieve(message)
        ctx, sources = retriever.build_context(hits)
        messages = retriever.build_messages(message, ctx, use_full_knowledge=use_full_knowledge)

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

        logger.info("Response stream opened.")
        start_time = time.perf_counter()

        # Accumulate all chunks into combined content
        combined_content = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

                # Accumulate content from this chunk
                combined_content += content

                response = StreamResponse(content=content, error=None)
                # await asyncio.sleep(0.2)  # Simulate network delay
                yield f"data: {response.model_dump_json()}\n\n"

        # Call complete_response with the accumulated content
        # This can be used for logging, analytics, storage, etc.
        if combined_content:
            complete_response(combined_content, model, message, session_id=session_id, use_full_knowledge=use_full_knowledge)

        yield "data: [DONE]\n\n"
        elapsed_sec = time.perf_counter() - start_time
        logger.info(f"Response streaming completed successfully in {elapsed_sec:.2f} sec")

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        error_response = ErrorResponse(error=f"API error: {str(e)}")
        yield f"data: {error_response.model_dump_json()}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest, session_id: Optional[str] = Query(default=None), username: str = Depends(authenticate_user)):
    """Handle streaming chat requests with OpenAI API"""

    # ensure we have a session
    session_id = get_or_create_session(username, session_id)
    session_data = user_sessions.get(session_id)

    if not session_data:
        logger.error("Session %s could not be created for user %s", session_id, username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to initialize chat session",
        )

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

    logger.info(f"Chat request: model={request.model}, message_length={len(request.message)}, use_full_knowledge={request.use_full_knowledge}")

    return StreamingResponse(
        stream_openai_response(
            request.message,
            request.model,
            request.use_full_knowledge,
            session_id=session_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.post("/sessions/new")
async def create_new_session(username: str = Depends(authenticate_user)):
    """Create a brand new chat session for the current user."""
    session_id = get_or_create_session(username, None, force_new=True)
    session_data = user_sessions[session_id]
    logger.info(f"Created new session for {username}, session_id={session_id}")
    return {
        "session_id": session_id,
        "chat_history": get_session_messages(session_id),
        "model": session_data.get("model", DEFAULT_MODEL),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    client = get_openai_client()
    return {
        "status": "healthy",
        "openai_configured": client is not None,
        "version": version
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
