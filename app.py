import os
import uuid
import hmac

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from assemble_graph import graph
from dotenv import load_dotenv

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

    # 1. Vérification du mot de passe (timing-safe)
    if not hmac.compare_digest(req.password or "", APP_PASSWORD or ""):
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
                if node == "retrieve":
                    if "messages" in update:
                        last_msg = update["messages"][-1]
                        if hasattr(last_msg, "artifact") and last_msg.artifact:
                            for doc in last_msg.artifact:
                                src = doc.metadata.get("source", "Source inconnue")
                                if src not in sources:
                                    sources.append(src)

                if node == "search_web":
                    if "web_search" not in sources:
                        sources.append("Recherche Web")

                if node == "generate_answer":
                    if "messages" in update:
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


CHATBOT_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Digestats — Assistant Réglementaire</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, system-ui, sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; }

        /* Login */
        #login-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #f0f2f5; display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .login-box { background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 90%; max-width: 400px; text-align: center; }
        .login-box h2 { margin-bottom: 16px; color: #1a5c2e; }
        .login-box input { width: 100%; padding: 12px; margin-bottom: 16px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
        .login-box button { width: 100%; padding: 12px; background: #1a5c2e; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }

        /* Layout */
        header { background: #1a5c2e; color: white; padding: 16px 24px; }
        header h1 { font-size: 18px; }
        #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
        #input-bar { padding: 16px; background: white; border-top: 1px solid #ddd; display: flex; gap: 12px; }
        #input-bar input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; outline: none; font-size: 14px; }
        #input-bar button { padding: 12px 20px; background: #1a5c2e; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
        #input-bar button:disabled { background: #ccc; cursor: not-allowed; }

        /* Messages */
        .message { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.7; }
        .user-msg { align-self: flex-end; background: #1a5c2e; color: white; border-bottom-right-radius: 2px; white-space: pre-wrap; }
        .bot-msg { align-self: flex-start; background: white; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }

        /* Contenu formaté du bot */
        .bot-msg strong { font-weight: 700; color: #1a3a20; }
        .bot-msg ul { margin: 6px 0 6px 8px; padding-left: 18px; }
        .bot-msg ul li { margin-bottom: 4px; }
        .bot-msg p { margin-bottom: 8px; }
        .bot-msg p:last-child { margin-bottom: 0; }
        .bot-msg hr { border: none; border-top: 1px solid #e0e0e0; margin: 10px 0; }

        /* Sources */
        .sources { font-size: 11px; color: #888; margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee; }

        /* Indicateur de chargement */
        .loader { align-self: flex-start; background: white; border-radius: 12px; border-bottom-left-radius: 2px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); display: flex; gap: 5px; align-items: center; }
        .loader span { width: 7px; height: 7px; background: #1a5c2e; border-radius: 50%; animation: bounce 1.2s infinite ease-in-out; }
        .loader span:nth-child(2) { animation-delay: 0.2s; }
        .loader span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
    </style>
</head>
<body>
    <div id="login-overlay">
        <div class="login-box">
            <h2>Accès Restreint</h2>
            <p>Veuillez entrer le mot de passe pour accéder à l'Assistant.</p>
            <input type="password" id="password-input" placeholder="Mot de passe..." autofocus />
            <button onclick="checkPassword()">Accéder</button>
        </div>
    </div>

    <header><h1>Assistant Réglementaire Digestats</h1></header>
    <div id="chat-container"></div>
    <div id="input-bar">
        <input type="text" id="user-input" placeholder="Votre question..." />
        <button id="send-btn" onclick="sendMessage()">Envoyer</button>
    </div>

    <script>
        let threadId = null;
        let currentPassword = "";
        const container = document.getElementById('chat-container');
        const input = document.getElementById('user-input');
        const btn = document.getElementById('send-btn');

        function checkPassword() {
            const pwd = document.getElementById('password-input').value;
            if (pwd) { currentPassword = pwd; document.getElementById('login-overlay').style.display = 'none'; }
        }

        // Convertit le markdown simplifié du LLM en HTML
        function renderMarkdown(text) {
            const lines = text.split('\\n');
            let html = '';
            let inList = false;

            for (const raw of lines) {
                const line = raw
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')  // échappement XSS
                    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');                 // **gras**

                if (/^[-•*]\\s+/.test(raw) || /^\\d+\\.\\s+/.test(raw)) {
                    if (!inList) { html += '<ul>'; inList = true; }
                    html += '<li>' + line.replace(/^[-•*]\\s+/, '').replace(/^\\d+\\.\\s+/, '') + '</li>';
                } else {
                    if (inList) { html += '</ul>'; inList = false; }
                    if (line.trim() === '' || line.trim() === '---') {
                        html += '<hr>';
                    } else {
                        html += '<p>' + line + '</p>';
                    }
                }
            }
            if (inList) html += '</ul>';
            return html;
        }

        function addLoader() {
            const d = document.createElement('div');
            d.className = 'loader'; d.id = 'loader';
            d.innerHTML = '<span></span><span></span><span></span>';
            container.appendChild(d);
            container.scrollTop = container.scrollHeight;
        }

        function removeLoader() {
            const l = document.getElementById('loader');
            if (l) l.remove();
        }

        function addMsg(txt, cls, sources = []) {
            const d = document.createElement('div');
            d.className = 'message ' + cls;
            if (cls === 'bot-msg') {
                d.innerHTML = renderMarkdown(txt);
            } else {
                d.textContent = txt;
            }
            if (sources.length > 0) {
                const s = document.createElement('div');
                s.className = 'sources';
                s.textContent = 'Sources : ' + sources.join(', ');
                d.appendChild(s);
            }
            container.appendChild(d);
            container.scrollTop = container.scrollHeight;
        }

        async function sendMessage() {
            const question = input.value.trim();
            if (!question) return;

            addMsg(question, 'user-msg');
            input.value = ''; btn.disabled = true;
            addLoader();

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, thread_id: threadId, password: currentPassword })
                });

                removeLoader();

                if (res.status === 401) {
                    alert("Session expirée ou mauvais mot de passe.");
                    location.reload(); return;
                }

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    addMsg(`Erreur serveur (${res.status}) : ${err.detail || 'réponse inattendue'}`, 'bot-msg');
                    return;
                }

                const data = await res.json();
                threadId = data.thread_id;
                addMsg(data.answer || 'Réponse vide.', 'bot-msg', data.sources);
            } catch (err) {
                removeLoader();
                addMsg(`Erreur de connexion : ${err.message || 'réseau indisponible'}`, 'bot-msg');
            } finally {
                btn.disabled = false;
            }
        }

        document.getElementById('password-input').onkeydown = e => { if (e.key === 'Enter') checkPassword(); };
        input.onkeydown = e => { if (e.key === 'Enter' && !btn.disabled) sendMessage(); };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def chatbot_ui():
    return CHATBOT_HTML
