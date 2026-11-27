"""
Tests de l'API FastAPI (auth fail-closed + health).

Ne teste PAS le chemin nominal du streaming : le graphe et le LLM sont
mockés, un test de bout en bout n'aurait pas de valeur ici. L'essentiel
est que le contrôle d'accès refuse TOUJOURS avant de toucher au graphe.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app import app

    return TestClient(app)


class TestHealth:
    def test_health_repond_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


class TestAuthFailClosed:
    """Régression : l'auth était fail-open si APP_PASSWORD était absent."""

    def test_mauvais_mot_de_passe_refuse(self, client):
        res = client.post("/api/chat", json={"question": "q", "password": "faux"})
        assert res.status_code == 401

    def test_mot_de_passe_absent_refuse(self, client):
        res = client.post("/api/chat", json={"question": "q"})
        assert res.status_code == 401

    def test_serveur_sans_password_configure_refuse_tout(self, client):
        # Si APP_PASSWORD n'est pas défini côté serveur, on doit obtenir 500,
        # JAMAIS un accès (fail-closed, pas fail-open).
        with patch("app.APP_PASSWORD", None):
            res = client.post("/api/chat", json={"question": "q", "password": ""})
        assert res.status_code == 500


class TestSources:
    def test_recherches_paralleles_gardent_les_sources_documentaires(self, client):
        # Régression (trace du 27/11) : retrieve_context et search_web lancés en
        # parallèle → seul le dernier ToolMessage était lu, l'UI n'affichait
        # que « Recherche Web » alors que la réponse citait le 2019/1009.
        import json

        from langchain_core.documents import Document
        from langchain_core.messages import AIMessage, ToolMessage

        doc_msg = ToolMessage(
            content="ctx", tool_call_id="1", name="retrieve_context",
            artifact=[Document(page_content="x", metadata={"source": "Reglement_UE_2019_1009.pdf"})],
        )
        web_msg = ToolMessage(content="web", tool_call_id="2", name="search_web")
        events = [
            ("updates", {"retrieve": {"messages": [doc_msg, web_msg]}}),
            ("updates", {"generate_answer": {"messages": [AIMessage(content="Réponse")]}}),
        ]

        with patch("app.APP_PASSWORD", "pw"), patch("app.graph") as graph:
            graph.stream.return_value = iter(events)
            res = client.post("/api/chat", json={"question": "q", "password": "pw"})

        sources = next(
            json.loads(block.split("data: ", 1)[1])["sources"]
            for block in res.text.split("\n\n") if block.startswith("event: sources")
        )
        assert sources == ["Reglement_UE_2019_1009.pdf", "Recherche Web"]
