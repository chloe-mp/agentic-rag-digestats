import hmac
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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

# Statuts affichés dans l'UI pendant que le graphe travaille (latence perçue)
NODE_STATUS = {
    "generate_query_or_respond": "Analyse de la question…",
    "retrieve": "Recherche dans la base réglementaire…",
    "rewrite_question": "Reformulation de la question…",
    "search_web": "Recherche web complémentaire…",
    "generate_answer": "Rédaction de la réponse…",
}


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    password: str | None = None


def _sse(event: str, data: dict) -> str:
    """Formate un événement Server-Sent Events."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text(content) -> str:
    """Normalise le contenu d'un chunk LLM (str ou liste de blocs) en texte."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


@app.get("/api/health")
async def health():
    """Vérification de l'état de l'API pour Cloud Run."""
    return {"status": "ok", "langfuse": "connected" if LANGFUSE_AVAILABLE else "unavailable"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Endpoint principal (Server-Sent Events) :
    1. Vérifie le mot de passe (fail-closed).
    2. Stream les statuts de progression, les sources, puis les tokens
       de la réponse au fil de la génération (latence perçue minimale).
    """
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="Authentification non configurée côté serveur")
    if not hmac.compare_digest(req.password or "", APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Mot de passe incorrect")

    thread_id = req.thread_id or f"chat_{uuid.uuid4().hex[:8]}"
    langfuse_handler = get_langfuse_handler()
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler] if langfuse_handler else [],
    }
    inputs = {"messages": [("user", req.question)]}

    def event_stream():
        sources = []
        sources_sent = False
        streamed_any_token = False
        direct_answer = ""
        final_answer = ""
        ctx = (
            propagate_attributes(session_id=thread_id, user_id="user")
            if LANGFUSE_AVAILABLE
            else None
        )
        try:
            if ctx:
                ctx.__enter__()
            yield _sse("meta", {"thread_id": thread_id})

            for mode, payload in graph.stream(inputs, config=config, stream_mode=["updates", "messages"]):
                if mode == "updates":
                    for node, update in payload.items():
                        if node in NODE_STATUS:
                            yield _sse("status", {"message": NODE_STATUS[node]})

                        if node == "retrieve" and "messages" in update:
                            last_msg = update["messages"][-1]
                            if getattr(last_msg, "artifact", None):
                                for doc in last_msg.artifact:
                                    src = doc.metadata.get("source", "Source inconnue")
                                    if src not in sources:
                                        sources.append(src)

                        if node == "search_web" and "Recherche Web" not in sources:
                            sources.append("Recherche Web")

                        # Réponse directe du planner (sans appel d'outil) :
                        # son message EST la réponse finale.
                        if node == "generate_query_or_respond" and "messages" in update:
                            msg = update["messages"][-1]
                            if not getattr(msg, "tool_calls", None):
                                direct_answer = _text(getattr(msg, "content", ""))

                        if node == "generate_answer" and "messages" in update:
                            final_answer = _text(update["messages"][-1].content)
                            if not sources_sent:
                                yield _sse("sources", {"sources": sources})
                                sources_sent = True

                elif mode == "messages":
                    chunk, metadata = payload
                    if metadata.get("langgraph_node") == "generate_answer":
                        text = _text(chunk.content)
                        if text:
                            streamed_any_token = True
                            yield _sse("token", {"content": text})

            # Fallbacks : réponse directe du planner, ou réponse générée
            # non streamée (le stream de tokens n'a rien produit).
            if not streamed_any_token:
                fallback = direct_answer or final_answer
                if fallback:
                    yield _sse("token", {"content": fallback})

            if not sources_sent:
                yield _sse("sources", {"sources": sources})
            yield _sse("done", {})

        except Exception as e:
            print(f"Erreur lors du streaming LangGraph : {e}")
            yield _sse("error", {"detail": "Erreur interne du moteur RAG"})
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            if _langfuse:
                _langfuse.flush()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/", response_class=FileResponse)
async def chatbot_ui():
    return FileResponse(STATIC_DIR / "index.html")
