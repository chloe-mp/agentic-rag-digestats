from dotenv import load_dotenv
from langchain.tools import tool

# On utilise l'import communautaire qui est le plus stable
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# On initialise l'outil
# Si tu as une erreur ici, c'est que la clé API est manquante dans ton .env
tavily_tool = TavilySearchResults(k=3)


@tool
def search_web(query: str):
    """
    Recherche des informations réglementaires actualisées sur le web (ex: AFSCA, NVWA).
    """
    return tavily_tool.invoke({"query": query})
