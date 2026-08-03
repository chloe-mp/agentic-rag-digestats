import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Modèle économique pour les nœuds intermédiaires (planification, grading,
# reformulation) : ~10x moins cher et plus rapide que Flash. Seule la
# génération finale (generate_answer) a besoin de la qualité de Flash.
gemini_lite = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
