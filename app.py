#!/usr/bin/env python3
"""
LLM Chat Client with Math and Markdown Rendering

A FastAPI application that provides a web interface for mathematical conversations
with AI, featuring LaTeX rendering, RAG context, and streaming responses.
"""
import os, time, random, string, json, re
import secrets

from pathlib import Path
from typing import AsyncGenerator, Optional, Any, Union, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Request, Depends, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

from my_logger import setup_logger;
from retriever import create_retriever
from config import (
    GLOBAL_RAGDATA_MAP,
    QUERY_EXAMPLES,
    LLM_DEFAULT_MODEL,
    USE_OPENAI,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
)
if USE_OPENAI:
    from openai import AsyncOpenAI

from fastapi.middleware.trustedhost import TrustedHostMiddleware


version = "0.2.2"
title="AI-Powered Q & A"

logger = setup_logger()
# Example usage
# logger.debug("Debugging message")
# logger.info("Information message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical error!")

load_dotenv()

DEFAULT_MODEL = LLM_DEFAULT_MODEL
AVAILABLE_MODELS = [DEFAULT_MODEL]

logger.info(f"Available models: {AVAILABLE_MODELS}")
logger.info(f"Default model: {LLM_DEFAULT_MODEL}")

# Template configuration
templates = Jinja2Templates(directory="web")
HTML_FILE_PATH = Path("web") / "index.html"

# Global client instance
_llm_client: Optional[Union["OllamaChatClient", "OpenAIChatClient"]] = None

retriever = create_retriever()

# Add after the existing imports and before app = FastAPI()
security = HTTPBasic()


class OllamaChatClient:
    """Thin async wrapper around the local Ollama HTTP API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

    async def close(self) -> None:
        """Tear down the underlying HTTP client."""
        await self._client.aclose()

    async def chat_stream(
        self,
        *,
        model: str,
        messages: Sequence[Any],
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream chat completions from Ollama.

        Yields the raw JSON payloads delivered by Ollama's streaming API.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON stream fragment from Ollama: %s", line)
                    continue
                yield data


class OpenAIChatClient:
    """Async wrapper for OpenAI chat completions with streaming support."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        if base_url:
            self._client: Any = AsyncOpenAI(base_url=base_url)
        else:
            self._client = AsyncOpenAI()

    async def close(self) -> None:
        await self._client.close()

    async def chat_stream(
        self,
        *,
        model: str,
        messages: Sequence[Any],
        **extra: Any,
    ) -> AsyncGenerator[Any, None]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **extra,
        )
        async for chunk in stream:
            yield chunk

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
        self.use_openai = USE_OPENAI
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).rstrip("/")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))

    @property
    def has_ollama_endpoint(self) -> bool:
        return bool(self.ollama_base_url)

    @property
    def has_openai_credentials(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


def get_llm_client() -> Optional[Union[OllamaChatClient, OpenAIChatClient]]:
    """Return whichever LLM client has been configured for this deployment."""
    return _llm_client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan manager"""
    global _llm_client
    config = Config()

    try:
        if config.use_openai:
            if not config.has_openai_credentials:
                raise RuntimeError("OPENAI_API_KEY not configured")
            _llm_client = OpenAIChatClient(base_url=config.openai_base_url)
            logger.info("OpenAI client initialized OK")
        elif config.has_ollama_endpoint:
            _llm_client = OllamaChatClient(config.ollama_base_url)
            logger.info("Ollama client initialized at %s", config.ollama_base_url)
        else:
            raise RuntimeError("OLLAMA_BASE_URL not configured")
    except Exception as exc:
        _llm_client = None
        logger.error("Failed to initialize LLM client: %s", exc)

    yield

    # Cleanup
    if _llm_client:
        await _llm_client.close()
        logger.info("LLM client closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=title,
    description="A web interface for mathematical conversations with AI",
    version=version,
    lifespan=lifespan
)

# CORS: configure for prod via ALLOWED_ORIGINS env (comma-separated)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
    if not allow_origins:
        allow_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Cache-Control", "X-Requested-With"],
    expose_headers=["Content-Type"],
    max_age=86400,
)

# Trusted Host protection (prevents Host header attacks)
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["chat.alexlabs.net", "www.chat.alexlabs.net", "localhost", "127.0.0.1"]

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# Mount static files (web UI plus every configured RAG folder)
app.mount("/static", StaticFiles(directory="web"), name="static")

for name, (raw_path, _rag_type) in GLOBAL_RAGDATA_MAP.items():
    fs_path = os.path.normpath(os.path.expanduser(raw_path))
    if not os.path.isdir(fs_path):
        logger.warning("Skipping mount for %s (%s not found)", name, fs_path)
        continue

    mount_path = f"/{name.lower()}"
    try:
        app.mount(mount_path, StaticFiles(directory=fs_path), name=name.lower())
        print(f"app.mount({mount_path}, directory={fs_path}, name={name.lower()})")

    except RuntimeError as exc:
        logger.error("Failed to mount %s at %s: %s", fs_path, mount_path, exc)

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
            "model": current_model or LLM_DEFAULT_MODEL,
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


def save_response(session_id: str, model: str, user_message: str, assistant_message: str, use_full_knowledge: bool) -> None:
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
    timestamp = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    session_messages.append({
        "timestamp": timestamp,
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
        description="Ollama model to use for the response"
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
    """Response model for available Ollama models"""
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
            "example_questions": QUERY_EXAMPLES
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

    return normalized_content

async def stream_ollama_response(
    message: str,
    model: str,
    client: OllamaChatClient,
    use_full_knowledge: bool = False,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream responses from the local Ollama API using Server-Sent Events format."""

    try:
        hits = retriever.retrieve(message)
        ctx, _sources = retriever.build_context(hits)
        messages = retriever.build_messages(message, ctx, use_full_knowledge=use_full_knowledge)

        logger.info("Opening Ollama response stream for model %s", model)
        start_time = time.perf_counter()
        combined_content = ""

        temp = 1.0 if any(tag in model for tag in ("-mini", "-nano")) else 0.0

        async for chunk in client.chat_stream(  # type: ignore[call-arg]
            model=model,
            messages=messages
        ):
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])

            content_piece = ""
            if "message" in chunk:
                content_piece = chunk["message"].get("content", "") or ""
            elif "response" in chunk:
                # Some Ollama versions stream under 'response'
                content_piece = chunk.get("response") or ""

            if content_piece:
                combined_content += content_piece
                response = StreamResponse(content=content_piece, error=None)
                yield f"data: {response.model_dump_json()}\n\n"

            if chunk.get("done"):
                break

        if combined_content:
            normalized_content = complete_response(
                combined_content,
                model,
                message,
                session_id=session_id,
                use_full_knowledge=use_full_knowledge,
            )
            final_response = StreamResponse(
                content=f"\n\n[NORMALIZED_CONTENT]\n{normalized_content}",
                error=None,
            )
            yield f"data: {final_response.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
        elapsed_sec = time.perf_counter() - start_time
        logger.info("Response streaming completed in %.2f sec", elapsed_sec)

    except httpx.HTTPError as exc:
        logger.error("Ollama HTTP error: %s", exc)
        error_response = ErrorResponse(error=f"Ollama HTTP error: {exc}")
        yield f"data: {error_response.model_dump_json()}\n\n"
    except Exception as exc:
        logger.error("Ollama streaming error: %s", exc)
        error_response = ErrorResponse(error=f"Ollama error: {exc}")
        yield f"data: {error_response.model_dump_json()}\n\n"


async def stream_openai_response(
    message: str,
    model: str,
    client: OpenAIChatClient,
    use_full_knowledge: bool = False,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream responses from OpenAI chat completions via SSE."""

    try:
        hits = retriever.retrieve(message)
        ctx, _sources = retriever.build_context(hits)
        messages = retriever.build_messages(message, ctx, use_full_knowledge=use_full_knowledge)

        logger.info("Opening OpenAI response stream for model %s", model)
        start_time = time.perf_counter()
        combined_content = ""

        async for chunk in client.chat_stream(model=model, messages=messages):
            chunk_content = ""
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                delta_content = getattr(delta, "content", None)
                if isinstance(delta_content, str):
                    chunk_content += delta_content
                elif isinstance(delta_content, list):
                    for part in delta_content:
                        text = None
                        if isinstance(part, dict):
                            text = part.get("text")
                        else:
                            text = getattr(part, "text", None)
                        if text:
                            chunk_content += text
            if not chunk_content:
                continue

            combined_content += chunk_content
            response = StreamResponse(content=chunk_content, error=None)
            yield f"data: {response.model_dump_json()}\n\n"

        if combined_content:
            normalized_content = complete_response(
                combined_content,
                model,
                message,
                session_id=session_id,
                use_full_knowledge=use_full_knowledge,
            )
            final_response = StreamResponse(
                content=f"\n\n[NORMALIZED_CONTENT]\n{normalized_content}",
                error=None,
            )
            yield f"data: {final_response.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
        elapsed_sec = time.perf_counter() - start_time
        logger.info("OpenAI response streaming completed in %.2f sec", elapsed_sec)

    except Exception as exc:
        logger.error("OpenAI streaming error: %s", exc)
        error_response = ErrorResponse(error=f"OpenAI error: {exc}")
        yield f"data: {error_response.model_dump_json()}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest, session_id: Optional[str] = Query(default=None), username: str = Depends(authenticate_user)):
    """Handle streaming chat requests using the configured LLM backend."""

    # ensure we have a session
    session_id = get_or_create_session(username, session_id)
    session_data = user_sessions.get(session_id)

    if not session_data:
        logger.error("Session %s could not be created for user %s", session_id, username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to initialize chat session",
        )

    client = get_llm_client()

    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama endpoint not configured"
        )

    if request.model not in AVAILABLE_MODELS:
        logger.warning(f"Unsupported model requested: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not supported"
        )

    # record user turn
    timestamp = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    session_data["messages"].append({"timestamp": timestamp, "role": "user", "content": request.message})

    logger.info(f"Chat request: model={request.model}, message_length={len(request.message)}, use_full_knowledge={request.use_full_knowledge}")

    if isinstance(client, OpenAIChatClient):
        stream_gen = stream_openai_response(
            message=request.message,
            model=request.model,
            client=client,
            use_full_knowledge=request.use_full_knowledge,
            session_id=session_id,
        )
    elif isinstance(client, OllamaChatClient):
        stream_gen = stream_ollama_response(
            message=request.message,
            model=request.model,
            client=client,
            use_full_knowledge=request.use_full_knowledge,
            session_id=session_id,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unknown LLM client",
        )

    return StreamingResponse(
        stream_gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
            # Let CORSMiddleware set Access-Control-Allow-Origin
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
    client = get_llm_client()
    backend = (
        "openai" if isinstance(client, OpenAIChatClient)
        else "ollama" if isinstance(client, OllamaChatClient)
        else None
    )
    return {
        "status": "healthy",
        "backend": backend,
        "ollama_base_url": OLLAMA_BASE_URL if backend == "ollama" else None,
        "version": version
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """Return available chat models for the configured backend"""
    if not AVAILABLE_MODELS:
        logger.error("No Ollama models configured")
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
    print("Local Math Chat Server")
    print("=" * 60)

    if config.use_openai:
        if not config.has_openai_credentials:
            print("WARNING: OPENAI_API_KEY not configured!")
        else:
            print("Backend: OpenAI API")
    else:
        if not config.has_ollama_endpoint:
            print("WARNING: OLLAMA_BASE_URL not configured!")
            print("Create a .env file with:")
            print("OLLAMA_BASE_URL=http://127.0.0.1:11434")
            print("Ensure Ollama is running locally with the desired model pulled.")
        else:
            print(f"Ollama endpoint: {config.ollama_base_url}")

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
