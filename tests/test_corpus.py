"""
Tests de la préparation du corpus (build_chunks) et de la sélection des
résultats (indexing_pipeline.select_top). Aucun appel réseau ni modèle.
"""
from langchain_core.documents import Document

PAGE_TYPE_7 = """# 5 Spécifications

| N° | Dénomination du type | Mode d'obtention | Teneur minimale N+P2O5+K2O |
| --- | --- | --- | --- |
| 6a | Engrais NP issu de lisier ⁽b⁾ | Phase solide des lisiers | 7 % |
| 7 | Fumier de volaille | Fiente de volaille et sa litière ⁽c⁾ compostée, ≥ 50 % MS | 7 % |

⁽c⁾ Litière : matière végétale utilisée pour l'élevage des animaux et recevant leurs déjections.
"""


class TestSplitMarkdown:
    def test_une_ligne_de_tableau_par_chunk_avec_entete_et_titre(self):
        from build_chunks import build_chunks

        chunks = build_chunks([("NF U 42-001-2.pdf", 18, "vision", PAGE_TYPE_7)])
        rows = [c for c in chunks if c["metadata"]["kind"] == "table_row"]

        assert len(rows) == 2
        type7 = next(c["page_content"] for c in rows if "Fumier de volaille" in c["page_content"])
        # Le chunk se suffit à lui-même : norme, section, colonnes, ligne
        assert type7.startswith("Norme NF U 42-001-2")
        assert "5 Spécifications" in type7
        assert "Teneur minimale N+P2O5+K2O" in type7
        assert "6a" not in type7  # plus mélangé avec les types voisins

    def test_notes_du_tableau_gardees_en_texte(self):
        from build_chunks import build_chunks

        chunks = build_chunks([("NF U 42-001-2.pdf", 18, "vision", PAGE_TYPE_7)])

        texts = [c["page_content"] for c in chunks if c["metadata"]["kind"] == "text"]
        assert any("Litière : matière végétale" in t for t in texts)

    def test_mention_afnor_retiree(self):
        from build_chunks import build_chunks

        page = (
            "Avant-propos\nL'attention est appelée sur le fait que certains des éléments du "
            "présent document peuvent faire l'objet de droits de propriété industrielle ou "
            "de droits analogues.\nLes amendements organiques doivent respecter "
            "N + P2O5 + K2O < 7 % sur MB, critère applicable à tous les types."
        )
        chunks = build_chunks([("norme NF U 44-051 2006+A1_2010_.pdf", 5, "text", page)])

        joined = " ".join(c["page_content"] for c in chunks)
        assert "droits de propriété" not in joined
        assert "N + P2O5 + K2O < 7 %" in joined  # le contenu utile reste


class TestSkipPage:
    def test_pages_de_garde_sommaire_bibliographie_exclues(self):
        from build_chunks import SKIP_PAGE

        assert SKIP_PAGE.search("FA167575 ISSN 0335-3931 NF U 42-001-2 Octobre 2020")
        assert SKIP_PAGE.search("NF U 42-001-2\nSommaire Page\nAvant-propos ....... 6")
        assert SKIP_PAGE.search("NF U42-001-2 Bibliographie [1] Règlement (CE) n°1069/2009")

    def test_avant_propos_avec_regles_conserve(self):
        # L'avant-propos de la NF U 44-051 contient une vraie règle (< 7 % sur MB)
        from build_chunks import SKIP_PAGE

        assert not SKIP_PAGE.search(
            "NF U 44-051 Avant-propos La présente norme s'applique aux produits répondant "
            "à la définition générale des amendements organiques"
        )


class TestVisionFallback:
    def test_transcription_vide_retombe_sur_le_texte(self):
        from build_chunks import _vision_or_text

        raw = "Article 3 " * 100
        assert _vision_or_text("x.pdf", 1, "", raw) == ("x.pdf", 1, "text", raw)

    def test_phrase_dintroduction_supprimee(self):
        from build_chunks import _vision_or_text

        md = "Voici la transcription fidèle du document :\n# Titre\nContenu de la page."
        assert _vision_or_text("x.pdf", 1, md, "court")[3] == "# Titre\nContenu de la page."


class TestSelectTop:
    def test_meme_passage_de_deux_documents_compte_une_fois(self):
        from indexing_pipeline import select_top

        a = Document(page_content="Norme NF U 42-001-2\n| 7 | Fumier de volaille | 7 % |")
        a_bis = Document(page_content="Autre titre\n| 7 | Fumier de volaille | 7 % |")
        b = Document(page_content="Règlement 142/2011\nLisier non transformé…")

        assert select_top([a, a_bis, b], 5) == [a, b]

    def test_limite_respectee(self):
        from indexing_pipeline import select_top

        docs = [Document(page_content=f"T\npassage {i}") for i in range(8)]
        assert len(select_top(docs, 5)) == 5
