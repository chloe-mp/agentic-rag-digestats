"""
Configuration pytest : mock des dépendances lourdes (ML, DB) avant tout import.
Ces patches s'appliquent au niveau sys.modules et s'exécutent avant la collecte
des tests, évitant ainsi le chargement des modèles HuggingFace et de Qdrant.
"""
import os
import sys
from unittest.mock import MagicMock

# --- Clés API fictives pour éviter les erreurs d'initialisation ---
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key")
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")


def _mock(name: str, **attrs) -> MagicMock:
    """Enregistre un MagicMock dans sys.modules si le module n'est pas déjà importé."""
    m = MagicMock(**attrs)
    sys.modules.setdefault(name, m)
    return m


# --- torch (évite la détection CUDA/MPS et le chargement de transformers) ---
_mock_torch = _mock("torch")
_mock_torch.backends.mps.is_available.return_value = False
_mock_torch.cuda.is_available.return_value = False
_mock("torch.nn")
_mock("torch.cuda")
_mock("torch.backends")

# --- transformers (évite l'initialisation lourde des tokenizers) ---
_mock("transformers")

# --- langchain_google_genai (non installé dans certains envs) ---
_mock_gemini_instance = MagicMock()
_mock_google_genai = _mock("langchain_google_genai")
_mock_google_genai.ChatGoogleGenerativeAI.return_value = _mock_gemini_instance

# --- HuggingFace Embeddings (évite le téléchargement du modèle) ---
_mock_embeddings = MagicMock()
_mock_embeddings.embed_query.return_value = [0.1] * 1024
_mock_embeddings.embed_documents.return_value = [[0.1] * 1024]
_mock_hf = _mock("langchain_huggingface")
_mock_hf.HuggingFaceEmbeddings.return_value = _mock_embeddings

# --- sentence_transformers CrossEncoder ---
_mock_reranker = MagicMock()
_mock_reranker.predict.return_value = []
_mock_st = _mock("sentence_transformers")
_mock_st.CrossEncoder.return_value = _mock_reranker

# --- FlagEmbedding (évite le crash à l'import causé par torch mocké) ---
_mock_flag = _mock("FlagEmbedding")
_mock_flag.BGEM3FlagModel = MagicMock()

# --- Qdrant (évite la connexion à la base locale) ---
_mock_qdrant_client = MagicMock()
_mock_qdrant_client.get_collections.return_value.collections = []
_mock_qdrant_client.scroll.return_value = ([], None)
_mock_qdrant_module = _mock("qdrant_client")
_mock_qdrant_module.QdrantClient.return_value = _mock_qdrant_client
_mock("qdrant_client.models")
_mock("langchain_qdrant")

# --- Tavily (évite la validation de la clé API au démarrage) ---
_mock("langchain_community.tools.tavily_search")
