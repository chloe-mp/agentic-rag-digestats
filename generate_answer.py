from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from models import gemini

GENERATE_PROMPT = (
    "Tu es un expert en réglementation des digestats et engrais agricoles. "
    "Tu ne réponds uniquement qu'à des questions sur la réglementation des digestats, engrais, "
    "fertilisants, sous-produits animaux et méthanisation.\n\n"
    "Réponds à la question en te basant UNIQUEMENT sur le contexte documentaire fourni.\n\n"
    "Si l'information n'est pas dans le contexte fourni, dis-le explicitement.\n\n"
    "STRUCTURE DE RÉPONSE (STRICTE) :\n"
    "1. **Réponse directe** : Commence par répondre précisément à la question posée "
    "(chiffre, condition, oui/non) en 1-3 phrases maximum.\n"
    "2. **Cadre réglementaire détaillé** : Détaille les étapes, démarches et procédures "
    "à suivre de tous les textes de loi et articles applicables. Pour chaque étape, cite les textes de loi et articles applicables.\n"
    "   - Mentionne TOUJOURS l'agrément sanitaire (**Préfet/DDPP**) pour les matières animales.\n"
    "   - Distingue les produits transformés (NF U, fertilisant UE) des matières brutes.\n"
    "   - Cite les chiffres et documents officiels (**Annexe I**, **Art. 48**, etc.).\n"
    "3. **Obligatoire vs en discussion** : Sépare clairement ce qui est "
    "**obligatoire aujourd'hui** de ce qui est **en cours de discussion**.\n\n"
    "RÈGLES DE FORMAT :\n"
    "- Commence DIRECTEMENT par la réponse (pas d'intro).\n"
    "- Utilise des titres (ex: **Étape 1**) et des listes à puces.\n"
    "- Utilise le gras (**) pour les autorités et documents clés.\n\n"
    "CONTEXTE :\n{context}\n\n"
    "QUESTION : {question}"
)


def generate_answer(state: MessagesState, config: RunnableConfig):
    """Génère une réponse structurée en utilisant le contexte récupéré."""
    # 1. Extraction de la question
    # On cherche le dernier message humain
    question = "Pas de question trouvée"
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) or m.type == "human":
            question = m.content
            break

    # 2. Extraction du contexte
    # ATTENTION : On filtre pour ne prendre QUE les messages des outils.
    # Si on prend le dernier message (state["messages"][-1]),
    # on risque de prendre le verdict du "grader" (ex: 'pertinent') au lieu des PDF !
    tool_messages = [m.content for m in state["messages"] if m.type == "tool"]
    context = (
        "\n\n---\n\n".join(tool_messages) if tool_messages else "Aucun contexte trouvé."
    )

    # 3. Préparation du prompt
    full_prompt = GENERATE_PROMPT.format(question=question, context=context)

    # 4. Invocation
    # On utilise SystemMessage pour les instructions et HumanMessage pour déclencher
    response = gemini.invoke(
        [
            SystemMessage(content="Tu es un assistant juridique spécialisé."),
            HumanMessage(content=full_prompt),
        ],
        config=config,
    )

    # 5. RETOUR OBLIGATOIRE : Un dictionnaire
    # On s'assure que c'est bien une liste de messages
    return {"messages": [AIMessage(content=response.content)]}
