from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

# Si tu as une erreur ici, c'est que la clé API est manquante dans ton .env
tavily_tool = TavilySearch(max_results=3)


@tool
def search_web(query: str):
    """
    Recherche des informations réglementaires actualisées sur le web (ex: AFSCA, NVWA).
    """
    return tavily_tool.invoke({"query": query})
