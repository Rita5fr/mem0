import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from mem0 import Memory

# ==============================================================
# 🔧 LOGGING SETUP
# ==============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================
# ⚙️ LOAD ENVIRONMENT VARIABLES
# ==============================================================
load_dotenv()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# ==============================================================
# 🧠 MEM0 CONFIGURATION
# ==============================================================
DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
            "sslmode": "require"
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "temperature": 0.2,
            "model": "gpt-4.1-nano-2025-04-14"
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

# ==============================================================
# 🧩 MEMORY INITIALIZATION (with safe reconnect)
# ==============================================================
def create_memory_instance():
    """Initialize Mem0 with robust Neon connection handling."""
    try:
        logging.info("Initializing Mem0 memory instance...")
        memory = Memory.from_config(DEFAULT_CONFIG)
        logging.info("✅ Mem0 initialized successfully.")
        return memory
    except Exception as e:
        logging.error(f"❌ Failed to initialize Mem0: {e}")
        raise

MEMORY_INSTANCE = create_memory_instance()

def get_memory_instance():
    """Reconnects to Neon automatically if the connection is dropped."""
    global MEMORY_INSTANCE
    try:
        MEMORY_INSTANCE.get_all(user_id="health_check")
        return MEMORY_INSTANCE
    except Exception as e:
        logging.warning(f"🔄 Reconnecting to Neon Postgres: {e}")
        MEMORY_INSTANCE = create_memory_instance()
        return MEMORY_INSTANCE

# ==============================================================
# 🚀 FASTAPI APP
# ==============================================================
app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.1.0",
)

# ==============================================================
# 📘 DATA MODELS
# ==============================================================
class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")

class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

# ==============================================================
# 🌐 REST ROUTES
# ==============================================================
@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}

@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    memory = get_memory_instance()
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier required.")
    try:
        params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
        response = memory.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", summary="Get memories")
def get_all_memories(user_id: Optional[str] = None, run_id: Optional[str] = None, agent_id: Optional[str] = None):
    memory = get_memory_instance()
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier required.")
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
        return memory.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    memory = get_memory_instance()
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return memory.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(user_id: Optional[str] = None, run_id: Optional[str] = None, agent_id: Optional[str] = None):
    memory = get_memory_instance()
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier required.")
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
        memory.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Redirect to API docs", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")
