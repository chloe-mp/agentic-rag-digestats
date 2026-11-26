from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from models import gemini_lite, text_of

REWRITE_PROMPT = (
    "Reformule cette question en une seule phrase technique pour une "
    "recherche documentaire réglementaire.\n"
    "Réponds UNIQUEMENT par la phrase reformulée : pas d'introduction, pas "
    "d'options, pas de liste, pas de guillemets.\n\n"
    "Question : {question}"
)

# Clé où l'on conserve la question telle que l'utilisateur l'a posée :
# generate_answer doit répondre à CETTE question, pas à la reformulation.
ORIGINAL_QUESTION_KEY = "original_question"


def _first_sentence(text: str) -> str:
    """Première ligne utile, sans puces / citations / gras / guillemets."""
    for line in text.splitlines():
        line = line.strip().lstrip("*->•").strip().strip("*").strip()
        line = line.strip("«»\"'").strip()
        if line and not line.endswith(":"):
            return line
    return ""


def rewrite_question(state: MessagesState, config: RunnableConfig):
    # On reformule le dernier message HUMAIN — pas messages[-1], qui après
    # le nœud retrieve est un ToolMessage (le contexte récupéré, pas la question).
    last_human_message = next(
        m for m in reversed(state["messages"]) if m.type == "human"
    )
    original = last_human_message.additional_kwargs.get(
        ORIGINAL_QUESTION_KEY
    ) or text_of(last_human_message.content)

    prompt = REWRITE_PROMPT.format(question=original)
    response = gemini_lite.invoke([{"role": "user", "content": prompt}], config=config)
    # Flash Lite 3.5 ignore parfois la consigne (« Voici plusieurs
    # formulations… ») : on ne garde qu'une phrase, sinon la question d'origine.
    rewritten = _first_sentence(text_of(response.content)) or original

    # ASTUCE : On remplace le message humain par la version optimisée
    # au lieu d'en ajouter un nouveau, pour ne pas perdre l'agent.
    # Les IDs étant inchangés, add_messages met à jour en place.
    last_human_message.content = rewritten
    last_human_message.additional_kwargs[ORIGINAL_QUESTION_KEY] = original
    return {"messages": state["messages"]}
