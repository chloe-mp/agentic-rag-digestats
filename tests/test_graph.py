"""
Tests unitaires pour les nœuds du graphe RAG Digestats.

Couvre :
- grade_documents  : routage après récupération de contexte
- generate_answer  : génération de réponse à partir du contexte
- rewrite_question : reformulation de la question
- generate_query_or_respond : planification des outils
- structure du graphe : nœuds et compilation
"""
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

EMPTY_CONFIG: RunnableConfig = {"configurable": {}, "callbacks": []}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_state(question: str, tool_contents: list[str] | None = None) -> dict:
    """
    Construit un MessagesState minimal pour les tests.

    Args:
        question: La question posée par l'utilisateur.
        tool_contents: Contenus des ToolMessages simulant des résultats de retrieval.
    """
    messages = [HumanMessage(content=question)]
    for i, content in enumerate(tool_contents or []):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": str(i),
                    "name": "retrieve_context",
                    "args": {"query": question},
                }],
            )
        )
        messages.append(ToolMessage(content=content, tool_call_id=str(i)))
    return {"messages": messages}


# ──────────────────────────────────────────────
# grade_documents
# ──────────────────────────────────────────────

class TestGradeDocuments:
    """Vérifie le routage selon la pertinence du document et le nombre de tentatives."""

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_doc_pertinent_route_vers_generate_answer(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="yes")

        state = make_state(
            "Quels sont les critères DIG ?",
            ["Les critères DIG incluent être une structure publique."],
        )
        assert grade_documents(state) == "generate_answer"

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_doc_non_pertinent_premiere_tentative_route_vers_rewrite(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="no")

        state = make_state(
            "Quels sont les critères DIG ?",
            ["Document sans rapport avec la question."],
        )
        assert grade_documents(state) == "rewrite_question"

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_doc_non_pertinent_apres_deux_tentatives_route_vers_search_web(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="no")

        # 2 ToolMessages = 2 tentatives de retrieval
        state = make_state(
            "Quels sont les critères DIG ?",
            ["Doc sans rapport (1ère tentative).", "Doc sans rapport (2ème tentative)."],
        )
        assert grade_documents(state) == "search_web"

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_comptage_tool_attempts_exact(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="no")

        # Exactement 2 ToolMessages → seuil atteint → search_web
        state = make_state("Question", ["ctx1", "ctx2"])
        assert grade_documents(state) == "search_web"

        # Exactement 1 ToolMessage → seuil non atteint → rewrite_question
        state = make_state("Question", ["ctx1"])
        assert grade_documents(state) == "rewrite_question"


# ──────────────────────────────────────────────
# generate_answer
# ──────────────────────────────────────────────

class TestGenerateAnswer:
    """Vérifie la construction de la réponse finale."""

    @patch("generate_answer.gemini")
    def test_retourne_un_ai_message(self, mock_gemini):
        from generate_answer import generate_answer

        mock_gemini.invoke.return_value = MagicMock(content="Réponse réglementaire.")

        result = generate_answer(make_state("Seuil d'épandage ?", ["Seuil : 170 kg N/ha."]), EMPTY_CONFIG)

        assert "messages" in result
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "Réponse réglementaire."

    @patch("generate_answer.gemini")
    def test_question_incluse_dans_le_prompt(self, mock_gemini):
        from generate_answer import generate_answer

        mock_gemini.invoke.return_value = MagicMock(content="Réponse.")

        generate_answer(make_state("Ma question précise", ["Contexte doc"]), EMPTY_CONFIG)

        call_args = mock_gemini.invoke.call_args[0][0]
        full_prompt = " ".join(m.content for m in call_args)
        assert "Ma question précise" in full_prompt

    @patch("generate_answer.gemini")
    def test_contexte_des_tool_messages_inclus(self, mock_gemini):
        from generate_answer import generate_answer

        mock_gemini.invoke.return_value = MagicMock(content="Réponse.")

        generate_answer(make_state("Question", ["Extrait du règlement 1069/2009."]), EMPTY_CONFIG)

        call_args = mock_gemini.invoke.call_args[0][0]
        full_prompt = " ".join(m.content for m in call_args)
        assert "Extrait du règlement 1069/2009." in full_prompt

    @patch("generate_answer.gemini")
    def test_fallback_si_aucun_contexte(self, mock_gemini):
        from generate_answer import generate_answer

        mock_gemini.invoke.return_value = MagicMock(content="Réponse.")

        # Aucun ToolMessage → contexte vide
        generate_answer(make_state("Question sans contexte"), EMPTY_CONFIG)

        call_args = mock_gemini.invoke.call_args[0][0]
        full_prompt = " ".join(m.content for m in call_args)
        assert "Aucun contexte trouvé" in full_prompt

    @patch("generate_answer.gemini")
    def test_plusieurs_tool_messages_concatenes(self, mock_gemini):
        from generate_answer import generate_answer

        mock_gemini.invoke.return_value = MagicMock(content="Réponse.")

        generate_answer(make_state("Question", ["Premier extrait.", "Deuxième extrait."]), EMPTY_CONFIG)

        call_args = mock_gemini.invoke.call_args[0][0]
        full_prompt = " ".join(m.content for m in call_args)
        assert "Premier extrait." in full_prompt
        assert "Deuxième extrait." in full_prompt


# ──────────────────────────────────────────────
# rewrite_question
# ──────────────────────────────────────────────

class TestRewriteQuestion:
    """Vérifie la reformulation de la question utilisateur."""

    @patch("rewrite_question.gemini")
    def test_reformule_le_dernier_message(self, mock_gemini):
        from rewrite_question import rewrite_question

        mock_gemini.invoke.return_value = MagicMock(
            content="Critères éligibilité DIG réglementation agricole"
        )

        result = rewrite_question(make_state("Critères DIG ?"), EMPTY_CONFIG)

        assert result["messages"][-1].content == "Critères éligibilité DIG réglementation agricole"

    @patch("rewrite_question.gemini")
    def test_retourne_un_dict_avec_messages(self, mock_gemini):
        from rewrite_question import rewrite_question

        mock_gemini.invoke.return_value = MagicMock(content="Question reformulée")

        result = rewrite_question(make_state("Question vague"), EMPTY_CONFIG)

        assert isinstance(result, dict)
        assert "messages" in result

    @patch("rewrite_question.gemini")
    def test_conserve_le_nombre_de_messages(self, mock_gemini):
        from rewrite_question import rewrite_question

        mock_gemini.invoke.return_value = MagicMock(content="Version reformulée")

        state = make_state("Question", ["Contexte"])
        nb_avant = len(state["messages"])
        result = rewrite_question(state, EMPTY_CONFIG)

        assert len(result["messages"]) == nb_avant


# ──────────────────────────────────────────────
# generate_query_or_respond
# ──────────────────────────────────────────────

class TestGenerateQueryOrRespond:
    """Vérifie la planification des appels d'outils."""

    @patch("query_pipeline.llm_with_tools")
    def test_retourne_un_ai_message(self, mock_llm):
        from query_pipeline import generate_query_or_respond

        mock_llm.invoke.return_value = AIMessage(content="Je vais utiliser retrieve_context.")

        result = generate_query_or_respond(make_state("Critères DIG ?"), EMPTY_CONFIG)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @patch("query_pipeline.llm_with_tools")
    def test_system_prompt_inclus_en_premier(self, mock_llm):
        from query_pipeline import generate_query_or_respond, SYSTEM_PROMPT

        mock_llm.invoke.return_value = AIMessage(content="Réponse")

        generate_query_or_respond(make_state("Question"), EMPTY_CONFIG)

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == SYSTEM_PROMPT

    @patch("query_pipeline.llm_with_tools")
    def test_messages_utilisateur_transmis_au_llm(self, mock_llm):
        from query_pipeline import generate_query_or_respond

        mock_llm.invoke.return_value = AIMessage(content="Réponse")

        generate_query_or_respond(make_state("Ma question spécifique"), EMPTY_CONFIG)

        call_args = mock_llm.invoke.call_args[0][0]
        # Le 2e élément de la liste doit être le HumanMessage
        user_messages = [m for m in call_args if isinstance(m, HumanMessage)]
        assert any("Ma question spécifique" in m.content for m in user_messages)


# ──────────────────────────────────────────────
# Structure du graphe
# ──────────────────────────────────────────────

class TestGraphStructure:
    """Vérifie la topologie du graphe sans appels LLM."""

    def test_noeuds_attendus_presents(self):
        from assemble_graph import workflow

        node_names = set(workflow.nodes.keys())
        expected = {
            "generate_query_or_respond",
            "retrieve",
            "rewrite_question",
            "generate_answer",
            "search_web",
        }
        assert expected.issubset(node_names), (
            f"Nœuds manquants : {expected - node_names}"
        )

    def test_graphe_compile_avec_succes(self):
        from assemble_graph import graph

        assert graph is not None

    def test_graphe_possede_un_checkpointer(self):
        from assemble_graph import graph

        assert graph.checkpointer is not None


# ──────────────────────────────────────────────
# Régressions multi-tour (bugs corrigés)
# ──────────────────────────────────────────────

def make_multiturn_state() -> dict:
    """Simule un 2e échange dans un thread persistant (MemorySaver)."""
    return {
        "messages": [
            HumanMessage(content="ANCIENNE question du tour 1"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "retrieve_context", "args": {"query": "x"}}],
            ),
            ToolMessage(content="contexte tour 1", tool_call_id="1"),
            AIMessage(content="réponse du tour 1"),
            HumanMessage(content="QUESTION du tour courant"),
            AIMessage(
                content="",
                tool_calls=[{"id": "2", "name": "retrieve_context", "args": {"query": "y"}}],
            ),
            ToolMessage(content="contexte tour 2", tool_call_id="2"),
        ]
    }


class TestGradeDocumentsMultiTour:
    """Le grader doit raisonner sur le tour courant, pas sur le début du thread."""

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_utilise_la_question_du_tour_courant(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="yes")

        grade_documents(make_multiturn_state())

        prompt = mock_gemini.with_structured_output.return_value.invoke \
            .call_args[0][0][0]["content"]
        assert "QUESTION du tour courant" in prompt
        assert "ANCIENNE question du tour 1" not in prompt

    @patch("document_grader.get_config", return_value={"configurable": {}, "callbacks": []})
    @patch("document_grader.gemini")
    def test_compteur_tentatives_limite_au_tour_courant(self, mock_gemini, _):
        from document_grader import grade_documents, GradeDocuments

        mock_gemini.with_structured_output.return_value.invoke.return_value = \
            GradeDocuments(binary_score="no")

        # 1 seul ToolMessage au tour courant (les 2 du tour 1 ne comptent pas)
        # → rewrite_question, pas search_web
        assert grade_documents(make_multiturn_state()) == "rewrite_question"


class TestRewriteQuestionMultiTour:
    """La reformulation cible le dernier message humain, pas le ToolMessage."""

    @patch("rewrite_question.gemini")
    def test_reformule_le_message_humain_pas_le_tool_message(self, mock_gemini):
        from rewrite_question import rewrite_question

        mock_gemini.invoke.return_value = MagicMock(content="question reformulée")

        state = make_state("Question originale", ["CONTEXTE_TOOL_A_PRESERVER"])
        result = rewrite_question(state, EMPTY_CONFIG)

        contents = [m.content for m in result["messages"]]
        assert "question reformulée" in contents
        assert "CONTEXTE_TOOL_A_PRESERVER" in contents  # le ToolMessage n'est plus écrasé
        assert "Question originale" not in contents

    @patch("rewrite_question.gemini")
    def test_reformule_la_question_du_tour_courant(self, mock_gemini):
        from rewrite_question import rewrite_question

        mock_gemini.invoke.return_value = MagicMock(content="reformulée")

        result = rewrite_question(make_multiturn_state(), EMPTY_CONFIG)

        contents = [m.content for m in result["messages"]]
        assert "ANCIENNE question du tour 1" in contents  # tour 1 inchangé
        assert "QUESTION du tour courant" not in contents
        assert "reformulée" in contents


class TestWebSearchNode:
    """Le fallback web doit chercher la question courante, avec un ID unique."""

    @patch("assemble_graph.search_web")
    def test_utilise_la_derniere_question_humaine(self, mock_search):
        from assemble_graph import web_search_node

        mock_search.invoke.return_value = "résultats web"

        web_search_node(make_multiturn_state())

        assert mock_search.invoke.call_args[0][0] == {"query": "QUESTION du tour courant"}

    @patch("assemble_graph.search_web")
    def test_tool_call_id_unique_entre_deux_recherches(self, mock_search):
        from assemble_graph import web_search_node

        mock_search.invoke.return_value = "r"
        state = {"messages": [HumanMessage(content="q")]}

        id1 = web_search_node(state)["messages"][0].tool_call_id
        id2 = web_search_node(state)["messages"][0].tool_call_id

        assert id1 != id2
