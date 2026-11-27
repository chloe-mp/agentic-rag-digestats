import json
import os
import sys
from hashlib import md5
from pathlib import Path

# sys.path AVANT les imports du projet, sinon ils échouent quand on lance
# le script depuis la racine (python evaluation/rag_eval.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from tqdm import tqdm

from assemble_graph import graph

load_dotenv()

from models import text_of  # noqa: E402

BASE_DIR = PROJECT_ROOT
DATASET_PATH = BASE_DIR / "evaluation" / os.environ.get("EVAL_DATASET", "question_answers.json")
# Dossier de sortie : un par collection évaluée (eval_results.json est un cache
# de reprise — mélanger deux collections fausserait la comparaison).
OUT_DIR = Path(os.environ.get("EVAL_OUT_DIR", BASE_DIR / "evaluation"))
RESULTS_PATH = OUT_DIR / "eval_results.json"
LIMIT = int(os.environ["EVAL_LIMIT"]) if os.environ.get("EVAL_LIMIT") else None


def collect_responses():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:LIMIT]

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    processed_questions = {r["question"] for r in results}

    for entry in tqdm(dataset, desc="Collecte RAG Agentique"):
        question = entry["question"]
        if question in processed_questions:
            continue

        config = {
            "configurable": {
                "thread_id": f"eval_{md5(question.encode()).hexdigest()[:8]}"
            }
        }
        inputs = {"messages": [("user", question)]}

        answer = ""
        contexts = []

        try:
            for chunk in graph.stream(inputs, config=config):
                for node, update in chunk.items():
                    # Tous les ToolMessages du nœud (le planner peut lancer
                    # plusieurs recherches en parallèle), pas seulement le dernier.
                    if node == "retrieve" and "messages" in update:
                        for msg in update["messages"]:
                            for doc in getattr(msg, "artifact", None) or []:
                                if doc.page_content not in contexts:
                                    contexts.append(doc.page_content)

                    if node == "search_web" and "messages" in update:
                        content = text_of(update["messages"][-1].content)
                        if content not in contexts:
                            contexts.append(content)

                    # Gemini 3 : contenu en liste de blocs → texte
                    if node == "generate_answer" and "messages" in update:
                        answer = text_of(update["messages"][-1].content)

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": entry.get("ground_truth", ""),
                }
            )

            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"\n✗ Erreur sur : {question[:30]}... -> {e}")

    return results


def run_ragas(results):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from models import gemini

    valid_results = [
        r for r in results if r.get("answer") and len(r["answer"].strip()) > 0
    ]
    print(f"\n--- RAGAS : {len(valid_results)} cas valides ---")

    llm = LangchainLLMWrapper(gemini)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True}
        )
    )

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] if r["contexts"] else ["N/A"],
            reference=r["ground_truth"],
        )
        for r in valid_results
    ]

    dataset = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    ragas_results = evaluate(dataset=dataset, metrics=metrics)
    print("\nRésultats RAGAS :")
    print(ragas_results)

    ragas_path = RESULTS_PATH.parent / "ragas_results.json"
    ragas_results.to_pandas().to_json(
        ragas_path, orient="records", force_ascii=False, indent=4
    )
    print(f"Résultats sauvegardés dans {ragas_path}")


if __name__ == "__main__":
    eval_data = collect_responses()
    if eval_data:
        run_ragas(eval_data)
