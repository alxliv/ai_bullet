import os
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import random
import string
import logging
import secrets
from retriever import Retriever, create_retriever, ask_llm
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH
from typing import Tuple
from fastapi import Query

version = "1.18.1"
print(f"Version: {version}")

# ANSI escape codes for colors
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"

class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: COLOR_CYAN,
        logging.INFO: COLOR_GREEN,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_RED,
        logging.CRITICAL: COLOR_RED
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, COLOR_RESET)
        record.levelname = f"{color}{record.levelname}{COLOR_RESET}"
        return super().format(record)

# Setup logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()

# Include timestamp and colored levelname
formatter = ColorFormatter(
    fmt='%(asctime)s %(levelname)-16s: %(message)s',
    datefmt='%d-%m %H:%M:%S'
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# Configure uvicorn loggers
if True:
    uvicorn_logger = logging.getLogger("uvicorn")

    # Use your existing ColorFormatter for uvicorn logs too
    uvicorn_logger.handlers.clear()

    # Add your colored formatter to uvicorn
    uvicorn_handler = logging.StreamHandler()
    uvicorn_handler.setFormatter(formatter)  # Your existing ColorFormatter
    uvicorn_logger.addHandler(uvicorn_handler)

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.INFO)
    uvicorn_access.handlers.clear()
    uvicorn_access.addHandler(uvicorn_handler)


# Example usage
#logger.debug("Debugging message")
#logger.info("Information message")
#logger.warning("Warning message")
#logger.error("Error message")
#logger.critical("Critical error!")

load_dotenv()

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
    logger.error("No USERS defined")
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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# Add CORS middleware to allow streaming
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_ROOT = os.path.expanduser(DOCUMENTS_PATH)
SRC_ROOT = os.path.expanduser(SOURCES_PATH)
EXAMPLES_ROOT = os.path.expanduser(EXAMPLES_PATH)

REPLACEMENTS = {}
for root, alias in (
    (DOCS_ROOT,    "/docs"),
    (SRC_ROOT,     "/src"),
    (EXAMPLES_ROOT, "/examples"),
):
    # map both the absolute path and the same string without leading slash
    REPLACEMENTS[root] = alias
    REPLACEMENTS[root.lstrip(os.sep)] = alias

# compile a single pattern matching any of those keys
RE_ROOTS_PATTERN = re.compile("|".join(re.escape(k) for k in REPLACEMENTS))


app.mount("/static", StaticFiles(directory="web"), name="static")
app.mount("/docs", StaticFiles(directory=DOCS_ROOT), name="docs")
app.mount("/src", StaticFiles(directory=SRC_ROOT), name="src")
app.mount("/examples", StaticFiles(directory=EXAMPLES_ROOT), name="examples")

templates = Jinja2Templates(directory="web")

# Session-based message storage - each client gets their own chat history
user_sessions = {}  # {session_id: {"messages": [...], "model": "..."}}
available_models = []  # Will be populated on startup
current_model = ""  # Will be set to first available model

def generate_short_id(length=8):
    """Generate a random identifier of given length."""
    # Create a sequence of uppercase letters and digits
    characters = string.ascii_lowercase + string.digits
    # Choose 'length' characters randomly
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id

# system_prompt = '''
# You are CodeEngineerGPT, an expert programming and systems-engineering assistant. Your specialties are:
#   • Python scripting and application development
#   • C and C++ system-level programming, performance optimization, and memory management

# When responding:
#   1. Strive for clear, concise, and correct answers, grounded in best practices.
#   2. Provide example code snippets, explanations of key concepts, and relevant references (e.g. official docs) when helpful.
#   3. Format your responses in markdown, and use LaTeX math notation (enclosed in $...$) for mathematical expressions and formulas.
#   4. To render math, use $ ... $ for inline math or $$ ... $$ for block math.
#   5. If you’re not sure or you lack sufficient information to answer confidently, say “I don’t know” or “I’m not sure” rather than guessing.
#   6. If the user’s question is ambiguous, too broad, or missing necessary details, ask a follow-up question to clarify before answering.
#   7. Avoid unnecessary jargon; when using technical terms, define them briefly.

# Always aim to be a reliable partner in their development workflow.
# '''

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

def get_available_models():
    return ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4.1', 'o4-mini']


# Initialize available models on startup
available_models = get_available_models()
current_model = available_models[0] if available_models else "gpt-3.5-turbo"

logger.info(f"Available models: {available_models}")
logger.info(f"Default model: {current_model}")

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

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request, session_id = None, username: str = Depends(authenticate_user)):
    # Get session_id from query parameter if not provided as path parameter
    if not session_id:
        session_id = request.query_params.get('session_id')

    # Get or create session
    session_id = get_or_create_session(username, session_id)

    # Get recent chat history if it exists (exclude system messages)
    session_data = user_sessions[session_id]
    all_messages = session_data["messages"]
    user_messages = [msg for msg in all_messages if msg["role"] != "system"]
    chat_history = user_messages[-10:] if len(user_messages) > 0 else []

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_history,
        "selected_model": session_data["model"],
        "available_models": available_models,
        "session_id": session_id,
        "username": username,
        "version": version
    })

@app.post("/", response_class=HTMLResponse)
async def chat_submit(request: Request, message: str = Form(...), model: str = Form(None), session_id: str = Form(None), username: str = Depends(authenticate_user)):
    pass

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon file"""
    from fastapi.responses import FileResponse
    return FileResponse("web/favicon.ico")

@app.get("/favicon.svg")
async def favicon_svg():
    """Serve SVG favicon file"""
    from fastapi.responses import FileResponse
    return FileResponse("web/favicon.svg")

@app.get("/styles.css")
async def get_styles():
    """Serve CSS file directly"""
    from fastapi.responses import FileResponse
    return FileResponse("web/styles.css", media_type="text/css")

@app.get("/marked.min.js")
async def get_marked_js():
    """Serve marked.js library"""
    from fastapi.responses import FileResponse
    return FileResponse("web/marked.min.js", media_type="application/javascript")

@app.post("/new-chat")
async def new_chat(request: Request, session_id: str = Form(None), username: str = Depends(authenticate_user)):
    """Save current chat and start a new one"""
    # Get or create session
    session_id = get_or_create_session(username, session_id='')

    # Reset messages to just the system message
    system_prompt = retriever.cfg.system_template
    user_sessions[session_id]["messages"] = [{"role": "system", "content": system_prompt}]

    # Redirect back to main page with session_id
    return RedirectResponse(url=f"/?session_id={session_id}", status_code=303)

@app.get("/saved-chats")
async def list_saved_chats():
    """List all saved chat files"""
    try:
        if not os.path.exists("saved_chats"):
            return {"chats": []}

        chat_files = []
        for filename in os.listdir("saved_chats"):
            if filename.endswith(".json"):
                filepath = os.path.join("saved_chats", filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                    chat_files.append({
                        "filename": filename,
                        "timestamp": chat_data.get("timestamp", "Unknown"),
                        "date": chat_data.get("date", "Unknown"),
                        "session_id": chat_data.get("session_id", "Unknown"),
                        "message_count": chat_data.get("message_count", 0)
                    })
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        # Sort by filename (which contains YYYYMMDD_HHMMSS timestamp) - newest first
        chat_files.sort(key=lambda x: x["filename"], reverse=True)
        return {"chats": chat_files}
    except Exception as e:
        return {"error": str(e), "chats": []}


@app.get("/session-info")
async def session_info(session_id=None):
    """Get session information"""
    if not session_id or session_id not in user_sessions:
        return {"error": "No active session"}

    session_data = user_sessions[session_id]
    return {
        "session_id": session_id,
        "model": session_data["model"],
        "message_count": len(session_data["messages"]),
        "total_sessions": len(user_sessions)
    }

@app.post("/api/chat")
async def api_chat(request: Request, username: str = Depends(authenticate_user)):
    pass

def parse_source_line(src: str) -> Tuple[str, str]:
    """
    Given a string like:
      "/home/…/Bullet_User_Manual.pdf : p 23, 1-17"
    return:
      ("/home/…/Bullet_User_Manual.pdf", "p 23, 1-17")
    """
    if ":" in src:
        file_path, page_line = src.split(":", 1)
    else:
        page_line=""
        file_path = src
    return file_path.strip(), page_line.strip()

from fastapi import Query

@app.get("/api/chat-stream")
async def api_chat_stream(
    session_id: str = Query(None),
    message: str   = Query(...),
    model:   str   = Query(None),
    use_full_knowledge:  bool = Query(False),
    username: str  = Depends(authenticate_user),
):
    """
    Streaming chat endpoint using Server-Sent Events.
    Expects: /api/chat-stream?session_id=…&message=…&model=…
    """
    # ensure we have a session
    session_id   = get_or_create_session(username, session_id)
    session_data = user_sessions[session_id]

    # select model
    if model and model in available_models:
        session_data["model"] = model
    else:
        model = session_data["model"]

    # record user turn
    session_data["messages"].append({"role": "user", "content": message})

    async def event_generator():
        try:
            # filter system messages for o1 models
            if model.startswith("o1"):
                msgs = [m for m in session_data["messages"] if m["role"] != "system"]
            else:
                msgs = session_data["messages"]

            # kickoff retrieval + streaming LLM
            stream, sources = ask_llm(message, retriever,
                    model=model, use_full_knowledge=use_full_knowledge, streaming=True)

            # initial SSE event
            yield f"data: {json.dumps({'type':'start', 'session_id':session_id, 'model':model})}\n\n"

            full_response = ""
            # stream chunks
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    full_response += delta
#                    print(f"Delta SSE received {delta}")
                    yield f"data: {json.dumps({'type':'content','content':delta})}\n\n"

            full_response = full_response.replace("](…/", "](/")

            # post‐process the complete answer
            full_response = RE_ROOTS_PATTERN.sub(lambda m: REPLACEMENTS[m.group(0)], full_response)
            # only display sources whose score is above your relevance threshold

            # filter by score
            MIN_SOURCE_SCORE = 0.25
            VERY_MIN_SOURCE_SCORE = 0.08
            relevant = [s for s in sources if s["score"] >= MIN_SOURCE_SCORE]
            if not relevant:
                relevant = [s for s in sources if s["score"] >= VERY_MIN_SOURCE_SCORE]
                if relevant:
                    relevant = relevant[:3]

            # append a SOURCES section
            if relevant:
                src_md = "\n\n**SOURCES:**\n"
                for src in relevant:
                    s = src["source"]
                    s = RE_ROOTS_PATTERN.sub(lambda m: REPLACEMENTS[m.group(0)], s)

                    url, loc = parse_source_line(s)
                    name = os.path.basename(url)
                    src_md += f"- [{name}]({url})"
                    if loc:
                        src_md += f" : {loc}"
                    src_md += f" <score: {src['score']:.3f}>"
                    src_md += "\n"
                full_response += src_md

            # save assistant turn
            session_data["messages"].append({"role":"assistant", "model": model, "content": full_response})
            save_chat_history(session_id)
            cleanup_old_sessions()

            # final SSE event
            print("Final SSE")
            yield f"data: {json.dumps({'type':'done','full_response': full_response})}\n\n"

        except Exception as e:
            err = f"Error with {model}: {e}"
            session_data["messages"].append({"role":"assistant","content": err})
            yield f"data: {json.dumps({'type':'error','error': err})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":            "no-cache",
            "Connection":               "keep-alive",
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

def cleanup_old_sessions():
    """Clean up sessions older than 24 hours (basic cleanup)"""
    # This is a simple implementation - in production you'd want more sophisticated cleanup
    if len(user_sessions) > 100:  # Simple threshold
        # Keep only the 50 most recent sessions
        session_items = list(user_sessions.items())
        for session_id, data in session_items[:-50]:
            del user_sessions[session_id]
        logger.info(f"Cleaned up old sessions, now have {len(user_sessions)} active sessions")


if __name__ == "__main__":
    import uvicorn
    print("before unicorn run")
    uvicorn.run("webgui:app", host="0.0.0.0", port=8501, reload=False)

# To run: uvicorn webgui:app --host 0.0.0.0 --port 8501
