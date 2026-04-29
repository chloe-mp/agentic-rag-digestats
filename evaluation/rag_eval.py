from assemble_graph import graph
import json
import sys
from hashlib import md5
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "evaluation/question_answers.json"
RESULTS_PATH = BASE_DIR / "evaluation/eval_results.json"


def collect_responses():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

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
                    if node == "retrieve":
                        if "messages" in update:
                            last_msg = update["messages"][-1]
                            if hasattr(last_msg, "artifact") and last_msg.artifact:
                                for doc in last_msg.artifact:
                                    if doc.page_content not in contexts:
                                        contexts.append(doc.page_content)

                    if node == "search_web":
                        if "messages" in update:
                            content = update["messages"][-1].content
                            if content not in contexts:
                                contexts.append(content)

                    if node == "generate_answer":
                        if "messages" in update:
                            answer = update["messages"][-1].content

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
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample
    from langchain_huggingface import HuggingFaceEmbeddings
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


def run_deepeval(results):
    valid_results = [
        r for r in results if r.get("answer") and len(r["answer"].strip()) > 0
    ]
    print(f"\n--- DeepEval : {len(valid_results)} cas valides ---")

    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate
    from models import judge

    test_cases = [
        LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            retrieval_context=r["contexts"] if r["contexts"] else ["N/A"],
            expected_output=r["ground_truth"],
        )
        for r in valid_results
    ]

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge),
    ]

    evaluate(test_cases, metrics)


if __name__ == "__main__":
    eval_data = collect_responses()
    if eval_data:
        run_ragas(eval_data)
        run_deepeval(eval_data)
