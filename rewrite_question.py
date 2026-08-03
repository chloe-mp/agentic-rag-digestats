from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from models import gemini

REWRITE_PROMPT = (
    "Reformule cette question en une seule phrase technique pour une "
    "recherche documentaire : {question}"
)


def rewrite_question(state: MessagesState, config: RunnableConfig):
    # On reformule le dernier message HUMAIN — pas messages[-1], qui après
    # le nœud retrieve est un ToolMessage (le contexte récupéré, pas la question).
    last_human_message = next(
        m for m in reversed(state["messages"]) if m.type == "human"
    )

    prompt = REWRITE_PROMPT.format(question=last_human_message.content)
    response = gemini.invoke([{"role": "user", "content": prompt}], config=config)

    # ASTUCE : On remplace le message humain par la version optimisée
    # au lieu d'en ajouter un nouveau, pour ne pas perdre l'agent.
    # Les IDs étant inchangés, add_messages met à jour en place.
    last_human_message.content = response.content
    return {"messages": state["messages"]}
