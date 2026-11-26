import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    # gemini-2.5-* ne sont plus servis aux comptes passés en prépaiement
    # (404 « no longer available to new users », nov. 2026).
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Modèle économique pour les nœuds intermédiaires (planification, grading,
# reformulation) : ~10x moins cher et plus rapide que Flash. Seule la
# génération finale (generate_answer) a besoin de la qualité de Flash.
def text_of(content) -> str:
    """Texte d'un contenu de message LangChain.

    Gemini 3 renvoie une liste de blocs ({"type": "text", "text": ..., "extras":
    {"signature": ...}}) au lieu d'une chaîne : un f-string ou str() injecterait
    la représentation Python brute (signature comprise) dans les prompts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


gemini_lite = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
