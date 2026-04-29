from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from models import gemini

REWRITE_PROMPT = (
    "Tu es un ingénieur expert en réglementation agricole. "
    "Reformule la question suivante pour optimiser une recherche dans une base de données de règlements (Règlement 1069/2009, 142/2011, etc.).\n\n"
    "QUESTION ORIGINALE : {question}\n\n"
    "CONSIGNE : Génère une requête technique concise incluant les mots-clés juridiques pertinents."
)


def rewrite_question(state: MessagesState, config: RunnableConfig):
    last_human_message = state["messages"][-1].content

    # On demande une reformulation SIMPLE, pas un dictionnaire de mots-clés
    REWRITE_PROMPT = "Reformule cette question en une seule phrase technique pour une recherche documentaire : {question}"

    prompt = REWRITE_PROMPT.format(question=last_human_message)
    response = gemini.invoke([{"role": "user", "content": prompt}], config=config)

    # ASTUCE : On remplace le dernier message par la version optimisée
    # au lieu d'en ajouter un nouveau, pour ne pas perdre l'agent.
    state["messages"][-1].content = response.content
    return {"messages": state["messages"]}
