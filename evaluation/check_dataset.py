import json
from pathlib import Path

# Chemins de configuration
BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "evaluation/question_answers.json"
DOCS_DIR = BASE_DIR / "docs"


def check_dataset_health():
    if not JSON_PATH.exists():
        print(f"Erreur : Le fichier {JSON_PATH} est introuvable.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Extraction des sources uniques
    sources_in_json = set()
    for item in dataset:
        source = item.get("source_document")
        if source:
            sources_in_json.add(source)

    print(f"Analyse de {len(dataset)} questions...")
    print(f"Sources uniques détectées dans le JSON : {len(sources_in_json)}\n")

    missing_files = []
    found_files = []
    virtual_sources = []  # Pour legifrance_live ou les URLs

    for source in sources_in_json:
        # Cas des sources qui ne sont pas des fichiers locaux
        if source.startswith("http") or source == "legifrance_live":
            virtual_sources.append(source)
            continue

        # Vérification des fichiers PDF
        file_path = DOCS_DIR / source
        if file_path.exists():
            found_files.append(source)
        else:
            missing_files.append(source)

    # --- AFFICHAGE DU BILAN ---
    if found_files:
        print("Fichiers présents dans /docs :")
        for f in found_files:
            print(f"  - {f}")

    if virtual_sources:
        print("\nSources externes (Web/API) :")
        for s in virtual_sources:
            print(f"  - {s}")

    if missing_files:
        print("\nALERTE : Fichiers manquants dans /docs !")
        print("Le RAG ne pourra pas trouver ces contextes en local :")
        for m in missing_files:
            print(f"  - {m}")
    else:
        print("\nParfait ! Tous les fichiers locaux sont prêts pour l'indexation.")


if __name__ == "__main__":
    check_dataset_health()
