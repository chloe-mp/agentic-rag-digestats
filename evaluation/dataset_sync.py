import json
from pathlib import Path
import os

# Chemins
JSON_PATH = str(Path(__file__).resolve().parent / "question_answers.json")

# Dictionnaire de correspondance (Ancien nom -> Nouveau nom)
MAPPING = {
    "1069:2009.pdf": "1069_2009.pdf",
    "CE_142:2011.pdf": "CE_142_2011.pdf",
    "Règlement UE 2019-1009 au 3 octobre 2022.pdf": "Reglement_UE_2019_1009.pdf",
}


def sync_json():
    if not os.path.exists(JSON_PATH):
        print("Fichier JSON introuvable.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for item in data:
        old_source = item.get("source_document")
        if old_source in MAPPING:
            item["source_document"] = MAPPING[old_source]
            count += 1

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Terminé ! {count} références mises à jour dans le dataset.")


if __name__ == "__main__":
    sync_json()
