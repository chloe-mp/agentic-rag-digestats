import os
import re
import sys
import glob
import torch
from hashlib import md5
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
import warnings

os.environ["USER_AGENT"] = "regulatory-rag-bot/1.0"
load_dotenv()

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", "."))
PDF_DIR = BASE_DIR / "docs"


WEB_URLS = []

print(f"[INIT] Chargement des modeles sur {DEVICE}...")

_bge_m3_model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=True,
    device=DEVICE
)

def bge_m3_embed(texts: list[str]):
    if isinstance(texts, str):
        texts = [texts]
    output = _bge_m3_model.encode(
        texts,
        batch_size=32,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    return {
        "dense": output["dense_vecs"],
        "sparse": output["lexical_weights"]
    }

_QDRANT_URL = os.environ.get("QDRANT_URL")
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if _QDRANT_URL and _QDRANT_API_KEY:
    _qdrant_client = QdrantClient(
        url=_QDRANT_URL,
        api_key=_QDRANT_API_KEY,
        timeout=30,
    )
else:
    warnings.warn(
        "QDRANT_URL et QDRANT_API_KEY non définis. "
        "Le client Qdrant n'est pas initialisé — l'app ne pourra pas faire de retrieval. "
        "Ce comportement est attendu en environnement de test. "
        "En prod : configure les variables d'environnement.",
        RuntimeWarning,
    )
    _qdrant_client = None

if _qdrant_client is not None:
    if not any(c.name == "reglementation_digestats" for c in _qdrant_client.get_collections().collections):
        _qdrant_client.create_collection(
            collection_name="reglementation_digestats",
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            }
        )
        print("[INIT] Collection 'reglementation_digestats' créée.")

_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)

_qdrant_points = []
if _qdrant_client is not None:
    try:
        _qdrant_points, _ = _qdrant_client.scroll(
            collection_name="reglementation_digestats",
            limit=100_000,
            with_payload=True,
        )
    except Exception:
        _qdrant_points = []

print(f"[INIT] Pret. Device: {DEVICE}, Chunks: {len(_qdrant_points)}")

_bm25_retriever = None
if _qdrant_points:
    _all_docs_for_bm25 = [
        Document(
            page_content=p.payload.get("page_content", ""),
            metadata=p.payload.get("metadata", {}),
        )
        for p in _qdrant_points
    ]
    _bm25_retriever = BM25Retriever.from_documents(_all_docs_for_bm25)
    _bm25_retriever.k = 30


def clean_document(text):
    lines = text.split("\n")
    lines = [line.strip() for line in lines if len(line.strip()) > 20]
    text = "\n".join(lines)
    text = re.sub(
        r"Naviguer dans le sommaire.*|Mentions legales.*", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def load_pdf_with_tables(pdf_path):
    import pdfplumber

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join([" | ".join(filter(None, row)) for row in table])
                if table_text.strip():
                    text += f"\n\nTABLEAU:\n{table_text}"
            if len(text.strip()) > 50:
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"source": os.path.basename(pdf_path), "page": i},
                    )
                )
    return docs


def index_documents():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if any(c.name == "reglementation_digestats" for c in _qdrant_client.get_collections().collections):
        _qdrant_client.delete_collection("reglementation_digestats")
    _qdrant_client.create_collection(
        collection_name="reglementation_digestats",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print("[INDEX] Collection réinitialisée.")

    all_docs = []
    pdf_paths = glob.glob(f"{PDF_DIR}/*.pdf")
    print(f"[INDEX] {len(pdf_paths)} PDFs trouves dans {PDF_DIR}")
    for path in pdf_paths:
        all_docs.extend(load_pdf_with_tables(path))

    if WEB_URLS:
        loader = WebBaseLoader(WEB_URLS)
        web_docs = loader.load()
        for d in web_docs:
            d.page_content = clean_document(d.page_content)
            if len(d.page_content) > 50:
                all_docs.append(d)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_splits = text_splitter.split_documents(all_docs)
    print(f"[INDEX] {len(all_splits)} chunks à indexer...")

    points = []
    batch_size = 64

    for i in range(0, len(all_splits), batch_size):
        batch_docs = all_splits[i:i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        embeddings = bge_m3_embed(texts)
        dense_vecs = embeddings["dense"]
        sparse_vecs = embeddings["sparse"]

        for j, doc in enumerate(batch_docs):
            points.append(models.PointStruct(
                id=i + j,
                vector={
                    "dense": dense_vecs[j].tolist(),
                    "sparse": models.SparseVector(
                        indices=[int(k) for k in sparse_vecs[j].keys()],
                        values=list(sparse_vecs[j].values())
                    )
                },
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            ))

    _qdrant_client.upsert(
        collection_name="reglementation_digestats",
        points=points,
        wait=True
    )

    global _bm25_retriever
    points_scroll, _ = _qdrant_client.scroll(
        collection_name="reglementation_digestats",
        limit=100_000,
        with_payload=True,
    )
    if points_scroll:
        bm25_docs = [
            Document(page_content=p.payload.get("page_content", ""),
                     metadata=p.payload.get("metadata", {}))
            for p in points_scroll
        ]
        _bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        _bm25_retriever.k = 30

    print(f"[INDEX] Terminé. {len(all_splits)} chunks indexés en hybrid dense+sparse.")


def rrf_fusion(results_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion sur plusieurs listes de Documents."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = md5(doc.page_content.encode()).hexdigest()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Recherche hybride dense + sparse BGE-M3 + BM25 + RRF + reranking."""
    query_emb = bge_m3_embed(query)
    query_dense = query_emb["dense"][0].tolist()
    query_sparse = models.SparseVector(
        indices=[int(k) for k in query_emb["sparse"][0].keys()],
        values=list(query_emb["sparse"][0].values())
    )

    qdrant_results = _qdrant_client.query_points(
        collection_name="reglementation_digestats",
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=30),
            models.Prefetch(query=query_sparse, using="sparse", limit=30),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=60,
        with_payload=True,
    ).points

    qdrant_docs = [
        Document(page_content=r.payload["page_content"], metadata=r.payload["metadata"])
        for r in qdrant_results
    ]

    bm25_docs = _bm25_retriever.invoke(query) if _bm25_retriever else []
    candidates = rrf_fusion([qdrant_docs, bm25_docs])

    if not candidates:
        return "Aucune source trouvée.", []
    
    candidates = candidates[:10]

    pairs = [[query, doc.page_content] for doc in candidates]
    rerank_scores = _reranker.predict(pairs, batch_size=8)

    for i, doc in enumerate(candidates):
        doc.metadata["rerank_score"] = float(rerank_scores[i])

    sorted_candidates = sorted(candidates, key=lambda x: x.metadata.get("rerank_score", -100), reverse=True)
    final_docs = sorted_candidates[:5]

    return (
        "\n\n".join(f"SOURCE: {d.metadata.get('source')}\nCONTENU: {d.page_content}" for d in final_docs),
        final_docs,
    )


def evaluate_rag():
    import json
    from tqdm import tqdm
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI

    dataset_path = BASE_DIR / "evaluation/question_answers.json"
    results_path = BASE_DIR / "evaluation/ragas_results.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True}
        )
    )

    samples = []
    for entry in tqdm(dataset, desc="RAGAS eval"):
        question = entry["question"]
        ground_truth = entry.get("ground_truth", "")

        content, docs = retrieve_context.func(question)
        contexts = [d.page_content for d in docs] if docs else ["N/A"]

        prompt = f"Question: {question}\n\nContextes:\n{content}\n\nRéponds en français de façon concise."
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        answer = _llm.invoke(prompt).content

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    results = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
    print("\nRésultats RAGAS :")
    print(results)

    df = results.to_pandas()
    df.to_json(results_path, orient="records", force_ascii=False, indent=4)
    print(f"Sauvegardé dans {results_path}")


if __name__ == "__main__":
    if "--reindex" in sys.argv:
        index_documents()
    elif "--evaluate" in sys.argv:
        evaluate_rag()
    elif "--query" in sys.argv:
        idx = sys.argv.index("--query")
        question = " ".join(sys.argv[idx + 1:])
        content, docs = retrieve_context.func(question)
        print(content)
    else:
        print(f"--- DEVICE: {DEVICE} ---")