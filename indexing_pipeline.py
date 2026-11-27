import json
import os
import re
import sys
import time
import warnings
from hashlib import md5
from pathlib import Path

import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams
from sentence_transformers import CrossEncoder

os.environ["USER_AGENT"] = "regulatory-rag-bot/1.0"
load_dotenv()

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Collection Qdrant interrogée : permet d'indexer une nouvelle version du
# corpus à côté de la production, de l'évaluer, puis de basculer par variable
# d'environnement (sans réindexer ni redéployer le code).
COLLECTION = os.environ.get("QDRANT_COLLECTION", "reglementation_digestats")

def _cgroup_cpus() -> str:
    """Quota CPU du conteneur (cgroup v2), à comparer aux threads torch."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return "illimité" if quota == "max" else f"{int(quota) / int(period):g}"
    except Exception:
        return "?"


print(f"[INIT] Chargement des modeles sur {DEVICE}... "
      f"(torch threads={torch.get_num_threads()}, os.cpu_count={os.cpu_count()}, "
      f"quota cgroup={_cgroup_cpus()})")

_bge_m3_model = BGEM3FlagModel(
    'BAAI/bge-m3',
    # fp16 uniquement sur GPU. Avec device="cpu" passé explicitement,
    # FlagEmbedding ne désactive PAS le fp16 lui-même (bge_m3.py) : le modèle
    # tournait en half() sur le CPU de Cloud Run, très lent (70-100 s/recherche).
    use_fp16=DEVICE != "cpu",
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

if _qdrant_client is not None and not _qdrant_client.collection_exists(COLLECTION):
    # Pas de création silencieuse : une collection vide ferait répondre l'app
    # « sans source » au lieu de signaler une erreur de configuration.
    warnings.warn(
        f"Collection Qdrant '{COLLECTION}' introuvable : lancer index_chunks.py.",
        RuntimeWarning,
    )

_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)

_qdrant_points = []
if _qdrant_client is not None:
    try:
        _qdrant_points, _ = _qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=100_000,
            with_payload=True,
        )
    except Exception as e:
        print(f"[WARN] Impossible de charger les points Qdrant pour BM25 : {e}")
        _qdrant_points = []

print(f"[INIT] Pret. Device: {DEVICE}, Collection: {COLLECTION}, Chunks: {len(_qdrant_points)}")

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


def index_chunks(chunks_path: Path, collection: str, batch_size: int = 64) -> int:
    """Indexe chunks.jsonl (produit par build_chunks.py) dans une collection NEUVE.

    Refuse d'écraser une collection existante : on indexe à côté de la
    production, on évalue, puis on bascule via QDRANT_COLLECTION.
    """
    if _qdrant_client.collection_exists(collection):
        raise SystemExit(f"La collection '{collection}' existe déjà : choisir un autre nom.")
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    _qdrant_client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        emb = bge_m3_embed([c["page_content"] for c in batch])
        _qdrant_client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=start + j,
                    vector={
                        "dense": emb["dense"][j].tolist(),
                        "sparse": models.SparseVector(
                            indices=[int(k) for k in emb["sparse"][j]],
                            values=list(emb["sparse"][j].values()),
                        ),
                    },
                    payload={"page_content": c["page_content"], "metadata": c["metadata"]},
                )
                for j, c in enumerate(batch)
            ],
            wait=True,
        )
        print(f"[INDEX] {min(start + batch_size, len(chunks))}/{len(chunks)}", flush=True)
    return len(chunks)


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


def _dedup_key(doc: Document) -> str:
    """Empreinte du contenu, sans le titre de document ajouté en tête de chunk."""
    body = doc.page_content.split("\n", 1)[-1]
    return re.sub(r"\W+", "", body.lower())[:300]


def select_top(docs: list[Document], n: int = 5) -> list[Document]:
    """Les n meilleurs documents, sans doublon de contenu (même passage cité deux fois)."""
    seen, out = set(), []
    for doc in docs:
        key = _dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) == n:
            break
    return out


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Recherche hybride dense + sparse BGE-M3 + BM25 + RRF + reranking."""
    # Chronométrage par sous-étape (une recherche prenait ~100 s sur Cloud Run).
    t0 = time.perf_counter()
    timings = {}

    def lap(step):
        nonlocal t0
        now = time.perf_counter()
        timings[step] = round(now - t0, 2)
        t0 = now

    query_emb = bge_m3_embed(query)
    lap("embed")
    query_dense = query_emb["dense"][0].tolist()
    query_sparse = models.SparseVector(
        indices=[int(k) for k in query_emb["sparse"][0]],
        values=list(query_emb["sparse"][0].values())
    )

    qdrant_results = _qdrant_client.query_points(
        collection_name=COLLECTION,
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
    lap("qdrant")

    bm25_docs = _bm25_retriever.invoke(query) if _bm25_retriever else []
    lap("bm25")
    candidates = rrf_fusion([qdrant_docs, bm25_docs])

    if not candidates:
        return "Aucune source trouvée.", []

    candidates = candidates[:10]

    pairs = [[query, doc.page_content] for doc in candidates]
    rerank_scores = _reranker.predict(pairs, batch_size=8)
    lap("rerank")
    print(f"[TIMING] retrieve_context {timings} total={sum(timings.values()):.1f}s "
          f"(candidats={len(pairs)}, car. max={max(len(p[1]) for p in pairs)})", flush=True)

    for i, doc in enumerate(candidates):
        doc.metadata["rerank_score"] = float(rerank_scores[i])

    sorted_candidates = sorted(candidates, key=lambda x: x.metadata.get("rerank_score", -100), reverse=True)
    final_docs = select_top(sorted_candidates, 5)

    return (
        "\n\n".join(f"SOURCE: {d.metadata.get('source')}\nCONTENU: {d.page_content}" for d in final_docs),
        final_docs,
    )


if __name__ == "__main__":
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        question = " ".join(sys.argv[idx + 1:])
        content, docs = retrieve_context.func(question)
        print(content)
    else:
        print("Usage : python indexing_pipeline.py --query <question>")
        print("Indexation : python build_chunks.py puis index_chunks.py")
