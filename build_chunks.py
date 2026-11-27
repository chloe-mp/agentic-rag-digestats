"""
Préparation du corpus : PDF → chunks.jsonl (aucun modèle local, tourne sur le Mac).

Pourquoi (analyse des traces Langfuse du 26/11/2026) :
- pdfplumber lit mal les tableaux des normes NF U (colonnes en biais, filigrane
  « Tous droits réservés » mêlé au texte) et indexait chaque page deux fois
  (texte brut + version TABLEAU) ;
- le type 7 « Fumier de volaille » partageait un chunk de 1500 caractères avec
  les types 5 et 6, et perdait face aux pages de garde / sommaires / bibliographies ;
- deux PDF étaient des doublons d'autres documents du corpus.

Ce script :
1. écarte les doublons (EXCLUDED) et les pages sans contenu utile
   (garde, sommaire, bibliographie) ;
2. fait transcrire par Gemini (vision) toutes les pages des normes NF U et les
   pages à tableaux des autres textes — une seule fois, mis en cache dans parsed/ ;
3. découpe selon la structure : une ligne de tableau = un chunk (avec l'en-tête
   du tableau), le texte par article / paragraphe ; chaque chunk est préfixé
   par le titre du document.

Usage :
    python build_chunks.py                 # transcrit ce qui manque, écrit chunks.jsonl
    python build_chunks.py --only-cached   # n'appelle pas Gemini (dry-run)
"""

import argparse
import base64
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pdfplumber
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "docs"
CACHE_DIR = ROOT / "parsed"
OUT_PATH = ROOT / "chunks.jsonl"

# Titre lisible ajouté en tête de chaque chunk : le contexte du document aide
# l'embedding et le reranker (« type 7 » seul ne dit pas de quelle norme il s'agit).
DOCS = {
    "NF U 42-001-2.pdf": "Norme NF U 42-001-2 (octobre 2020) — Engrais organiques",
    "NF U 42-001- 3 - novembre 2020.pdf": "Norme NF U 42-001-3 (novembre 2020) — Engrais organo-minéraux",
    "norme NF U 44-051 2006+A1_2010_.pdf": "Norme NF U 44-051 (2006+A1 2010) — Amendements organiques",
    "1069_2009.pdf": "Règlement (CE) n° 1069/2009 — Sous-produits animaux",
    "CE_142_2011.pdf": "Règlement (UE) n° 142/2011 — Application du règlement (CE) n° 1069/2009",
    "Reglement_UE_2019_1009.pdf": "Règlement (UE) 2019/1009 — Fertilisants UE (version consolidée 2022)",
    "Legifrance_2009.pdf": "Arrêté du 8 décembre 2011 — Règles sanitaires sous-produits animaux",
    "Texte_legifrance_DIG.pdf": "Arrêté du 22 octobre 2020 — Cahier des charges digestats (CDC DIG)",
}
EXCLUDED = {
    # Même édition que « NF U 42-001-2.pdf » (84/94 chunks quasi identiques),
    # avec un filigrane qui corrompt le texte (« Dénomoination », « voléaille »).
    "NF U 42-001-2 - octobre 2020.pdf",
    # Deuxième copie du règlement 2019/1009 ; 90 pages sur 114 sans texte
    # extractible. La version consolidée 2022 est dans Reglement_UE_2019_1009.pdf.
    "Règlement_CE.pdf",
}
# Normes : filigrane + tableaux en biais → toutes les pages passent en vision.
VISION_ALL_PAGES = {name for name in DOCS if "NF U" in name}

SKIP_PAGE = re.compile(
    r"ISSN 0335-3931"                       # page de garde AFNOR
    r"|^\W*(?:NF U[\w\s:+.-]*)?Sommaire\b"  # sommaire
    r"|\bBibliographie\s*(?:R\s*)?\[1\]",   # bibliographie numérotée
    re.I | re.M,
)
# Paragraphe standard présent dans chaque norme : il matchait toute requête « norme ».
AFNOR_DISCLAIMER = re.compile(
    r"L'attention est appelée sur le fait que certains des éléments.*?"
    r"(?:analogues|identifié tout ou partie de ces droits)\.?",
    re.S,
)

VISION_MODEL = "gemini-3.5-flash"
VISION_PROMPT = (
    "Transcris fidèlement cette page de texte réglementaire en Markdown.\n"
    "- Reproduis TOUT le contenu, mot pour mot. N'invente, ne résume et ne corrige rien.\n"
    "- Tableaux : tableau Markdown dont l'en-tête tient sur UNE SEULE ligne (combine les "
    "en-têtes à plusieurs niveaux, ex. « Teneur minimale N+P2O5+K2O ») ; puis une ligne "
    "par entrée du tableau (fusionne les cellules d'une même entrée réparties sur "
    "plusieurs lignes visuelles). Les notes du tableau ((a), (b)…) vont en texte SOUS "
    "le tableau, pas dans le tableau.\n"
    "- JAMAIS de LaTeX ni de $ : écris en caractères normaux P2O5, K2O, ≥, ≤, %, ●, ○. "
    "Appels de note en exposant : (a), (b), (c).\n"
    "- Titres de section : lignes commençant par #.\n"
    "- Ignore le filigrane (lettres éparses, « Tous droits réservés »), les en-têtes et "
    "pieds de page répétés et les numéros de page.\n"
    "Réponds uniquement par la transcription."
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    # Articles et paragraphes numérotés d'abord, puis paragraphes, phrases.
    separators=["\nArticle ", "\n#", "\n\n", "\n", ". ", " ", ""],
    keep_separator="start",
)


def _page_png(page) -> bytes:
    buf = io.BytesIO()
    page.to_image(resolution=150).original.save(buf, format="PNG")
    return buf.getvalue()


def _transcribe(png: bytes) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    from models import text_of

    # thinking_level="low" : ~800 tokens/page au lieu de ~6 000 (raisonnement),
    # même transcription une fois la consigne explicite (en-tête sur une ligne, pas de LaTeX).
    llm = ChatGoogleGenerativeAI(model=VISION_MODEL, temperature=0, thinking_level="low")
    msg = HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
    ])
    # Gemini renvoie parfois une réponse vide (vu sur 10 pages du 2019/1009) :
    # une seconde tentative suffit en général.
    for _ in range(2):
        md = INTRO_LINE.sub("", text_of(llm.invoke([msg]).content)).strip()
        if md:
            return md
    return ""


# « Voici la transcription fidèle… : » malgré la consigne.
INTRO_LINE = re.compile(r"\A\s*Voici[^\n]*:\s*\n")
# Une transcription bien plus courte que le texte extractible = page tronquée
# ou vide côté vision : on garde alors le texte pdfplumber.
MIN_VISION_RATIO = 0.5


def _cache_path(source: str, page: int) -> Path:
    return CACHE_DIR / Path(source).stem / f"{page:04d}.md"


def _vision_or_text(source: str, page: int, md: str, raw: str):
    md = INTRO_LINE.sub("", md).strip()
    if len(md) < MIN_VISION_RATIO * len(raw.strip()):
        print(f"[FALLBACK] {source} p{page} : vision {len(md)} car. < texte {len(raw.strip())} car.")
        return source, page, "text", raw
    return source, page, "vision", md


def collect_pages(only_cached: bool):
    """Liste (source, page, mode, texte) ; transcrit en vision ce qui manque."""
    pages, to_transcribe = [], []
    for source in sorted(DOCS):
        with pdfplumber.open(PDF_DIR / source) as pdf:
            for i, page in enumerate(pdf.pages):
                raw = page.extract_text() or ""
                if SKIP_PAGE.search(raw):
                    continue
                vision = source in VISION_ALL_PAGES or bool(page.find_tables())
                if not vision:
                    if len(raw.strip()) >= 50:
                        pages.append((source, i, "text", raw))
                    continue
                cache = _cache_path(source, i)
                if cache.exists():
                    pages.append(_vision_or_text(source, i, cache.read_text(), raw))
                elif only_cached:
                    print(f"[DRY] pas de transcription en cache : {source} p{i}")
                else:
                    to_transcribe.append((source, i, _page_png(page), raw))

    def work(item):
        source, i, png, raw = item
        md = _transcribe(png)
        if md:  # jamais de réponse vide en cache : elle serait réutilisée
            cache = _cache_path(source, i)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(md)
        return _vision_or_text(source, i, md, raw)

    if to_transcribe:
        print(f"[VISION] {len(to_transcribe)} pages à transcrire avec {VISION_MODEL}…")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for n, res in enumerate(pool.map(work, to_transcribe), 1):
                pages.append(res)
                print(f"  {n}/{len(to_transcribe)} {res[0]} p{res[1]}", flush=True)
    return sorted(pages, key=lambda p: (p[0], p[1]))


def _split_markdown(md: str):
    """Sépare une page Markdown en blocs texte et tableaux (en-tête, lignes)."""
    blocks, text, table = [], [], []
    heading = ""

    def flush_table():
        if len(table) >= 2:
            header, rows = table[0], [r for r in table[1:] if not re.fullmatch(r"[\s|:\-]+", r)]
            blocks.append(("table", heading, header, rows))
        elif table:
            text.extend(table)
        table.clear()

    for line in md.splitlines():
        if line.lstrip().startswith("|"):
            if text:
                blocks.append(("text", "\n".join(text)))
                text = []
            table.append(line.strip())
            continue
        flush_table()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        text.append(line)
    flush_table()
    if text:
        blocks.append(("text", "\n".join(text)))
    return blocks


def build_chunks(pages):
    chunks = []
    for source, page, mode, content in pages:
        title = DOCS[source]
        meta = {"source": source, "page": page, "title": title}
        content = AFNOR_DISCLAIMER.sub("", content)
        blocks = _split_markdown(content) if mode == "vision" else [("text", content)]
        for block in blocks:
            if block[0] == "table":
                _, heading, header, rows = block
                for row in rows:
                    ctx = f"{title}\n{heading}\n" if heading else f"{title}\n"
                    chunks.append({
                        "page_content": f"{ctx}{header}\n{row}",
                        "metadata": {**meta, "kind": "table_row"},
                    })
            else:
                text = block[1].strip()
                if len(text) < 50:
                    continue
                for piece in splitter.split_text(text):
                    if len(piece.strip()) >= 50:
                        chunks.append({
                            "page_content": f"{title}\n{piece.strip()}",
                            "metadata": {**meta, "kind": "text"},
                        })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-cached", action="store_true")
    args = parser.parse_args()

    missing = [p for p in DOCS if not (PDF_DIR / p).exists()]
    if missing:
        sys.exit(f"PDF manquants dans {PDF_DIR} : {missing}")
    ignored = sorted(p.name for p in PDF_DIR.glob("*.pdf") if p.name not in DOCS)
    for name in ignored:
        why = "doublon écarté" if name in EXCLUDED else "NON RÉFÉRENCÉ dans DOCS"
        print(f"[SKIP] {name} ({why})")

    pages = collect_pages(args.only_cached)
    chunks = build_chunks(pages)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    kinds = {}
    for c in chunks:
        key = (c["metadata"]["source"], c["metadata"]["kind"])
        kinds[key] = kinds.get(key, 0) + 1
    for (src, kind), n in sorted(kinds.items()):
        print(f"  {n:5} {kind:9} {src}")
    print(f"[OK] {len(chunks)} chunks → {OUT_PATH}")


if __name__ == "__main__":
    main()
