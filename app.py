import hmac
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from assemble_graph import graph

load_dotenv()

# --- Langfuse v3 ---
try:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler

    _langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    if _langfuse.auth_check():
        LANGFUSE_AVAILABLE = True
        print("[INFO] Langfuse connecté avec succès.")
    else:
        LANGFUSE_AVAILABLE = False
        _langfuse = None
        print("[WARN] Langfuse : auth_check() échoué — clés invalides ou projet introuvable.")
except Exception as e:
    print(f"[WARN] Langfuse non disponible: {e}")
    _langfuse = None
    LANGFUSE_AVAILABLE = False


def get_langfuse_handler():
    if not LANGFUSE_AVAILABLE:
        return None
    try:
        return CallbackHandler()  # utilise la config globale initialisée par Langfuse()
    except Exception as e:
        print(f"Langfuse handler error: {e}")
        return None


app = FastAPI(title="RAG Digestats API")

APP_PASSWORD = os.environ.get("APP_PASSWORD")

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    password: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str]


@app.get("/api/health")
async def health():
    """Vérification de l'état de l'API pour Cloud Run."""
    return {"status": "ok", "langfuse": "connected" if LANGFUSE_AVAILABLE else "unavailable"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint principal :
    1. Vérifie le mot de passe.
    2. Configure le tracing Langfuse.
    3. Exécute le graphe LangGraph.
    """

    # 1. Vérification du mot de passe (timing-safe, fail-closed)
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="Authentification non configurée côté serveur")
    if not hmac.compare_digest(req.password or "", APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Mot de passe incorrect")

    # 2. Préparation de la session (Thread ID)
    thread_id = req.thread_id or f"chat_{uuid.uuid4().hex[:8]}"

    # 3. Configuration de Langfuse (Callbacks)
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    # 4. Configuration de LangGraph
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
    }

    inputs = {"messages": [("user", req.question)]}
    answer = ""
    sources = []

    # 5. Exécution du flux RAG
    ctx = propagate_attributes(session_id=thread_id, user_id="user") if LANGFUSE_AVAILABLE else None
    try:
        if ctx:
            ctx.__enter__()
        for chunk in graph.stream(inputs, config=config):
            for node, update in chunk.items():
                if node == "retrieve" and "messages" in update:
                    last_msg = update["messages"][-1]
                    if hasattr(last_msg, "artifact") and last_msg.artifact:
                        for doc in last_msg.artifact:
                            src = doc.metadata.get("source", "Source inconnue")
                            if src not in sources:
                                sources.append(src)

                if node == "search_web" and "web_search" not in sources:
                    sources.append("Recherche Web")

                if node == "generate_answer" and "messages" in update:
                    answer = update["messages"][-1].content

        if not answer:
            answer = "Désolé, je n'ai pas pu générer de réponse."

    except Exception as e:
        print(f"Erreur lors du streaming LangGraph : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du moteur RAG")
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
        if _langfuse:
            _langfuse.flush()

    return ChatResponse(answer=answer, thread_id=thread_id, sources=sources)


@app.get("/", response_class=FileResponse)
async def chatbot_ui():
    return FileResponse(STATIC_DIR / "index.html")
