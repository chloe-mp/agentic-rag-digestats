#!/usr/bin/env bash
# hf_job.sh — indexation et éval sur GPU loué (HF Jobs), pas sur le Mac.
#
#   ./hf_job.sh index reglementation_digestats_v2   # embeddings BGE-M3 → nouvelle collection
#   ./hf_job.sh eval  reglementation_digestats      # éval RAGAS (50 questions) d'une collection
#   EVAL_LIMIT=3 ./hf_job.sh eval <collection>      # essai rapide
#   EVAL_DATASET=question_answers_nfu.json ./hf_job.sh eval <collection>  # questions normes NF U
#   RERANKER_MODEL=<repo HF> ./hf_job.sh eval <collection>                # autre reranker
#
# Préalable pour index : python build_chunks.py (chunks.jsonl, local, sans modèle).
# Les deux évals comparées doivent tourner sur le MÊME matériel (FLAVOR fixe).
# Seuls le code, le jeu d'éval et chunks.jsonl partent sur le Hub, dans le
# bucket PRIVÉ (chunks.jsonl contient le texte des normes : jamais en public).
# Les résultats d'éval sont aussi imprimés dans les logs (entre marqueurs).

set -euo pipefail
cd "$(dirname "$0")"

BUCKET="Chloemp/agentic-rag"
# 1x L4 24 Go, 0,80 $/h. FLAVOR=a10g-small si aucun L4 n'est libre (file d'attente
# HF) : sans effet sur les scores (embeddings/reranker identiques, notes par Gemini).
FLAVOR="${FLAVOR:-l4x1}"
IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm"

MODE="${1:?usage : $0 index|eval <collection>}"
COLLECTION="${2:?usage : $0 index|eval <collection>}"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo -dirty)"
RUN="${MODE}-${COLLECTION}${EVAL_DATASET:+-${EVAL_DATASET%.json}}${RERANKER_MODEL:+-${RERANKER_MODEL##*/}}-$(date +%Y%m%d-%H%M)"

[ "$MODE" = index ] && [ ! -s chunks.jsonl ] && {
  echo "chunks.jsonl absent : lancer d'abord python build_chunks.py" >&2; exit 1; }

# --- code (+ corpus préparé) → bucket privé ---
hf buckets create "$BUCKET" --private --exist-ok >/dev/null
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp ./*.py requirements.txt "$STAGE/"
mkdir -p "$STAGE/evaluation"
cp evaluation/rag_eval.py evaluation/*.json "$STAGE/evaluation/"
[ "$MODE" = index ] && cp chunks.jsonl "$STAGE/"
hf buckets sync "$STAGE" "hf://buckets/$BUCKET/code" --delete >/dev/null

# Torch CUDA de PyPI à la place de torch+cpu (lock de l'image Cloud Run).
SETUP="cp -r /bucket/code /work && cd /work && \
grep -v -e '^--extra-index-url' -e '^torch==' requirements.txt > req-gpu.txt && \
echo 'torch==2.11.0' >> req-gpu.txt && \
uv pip install --system -q -r req-gpu.txt && nvidia-smi -L"

if [ "$MODE" = index ]; then
  KEYS="QDRANT_URL|QDRANT_API_KEY"
  CMD="python index_chunks.py --collection $COLLECTION"
else
  KEYS="QDRANT_URL|QDRANT_API_KEY|GOOGLE_API_KEY|TAVILY_API_KEY"
  OUT="/work/out"
  CMD="QDRANT_COLLECTION=$COLLECTION EVAL_OUT_DIR=$OUT python evaluation/rag_eval.py; \
echo '=====RAGAS_JSON====='; cat $OUT/ragas_results.json; echo '=====END====='; \
mkdir -p /bucket/results/$RUN && cp $OUT/*.json /bucket/results/$RUN/ || true"
fi

# Secrets : uniquement les clés nécessaires, lues dans .env, passées par stdin.
grep -E "^($KEYS)=" .env |
hf jobs run --detach \
  --secrets-file - \
  --name "rag-$RUN" \
  --flavor "$FLAVOR" \
  --timeout 3h \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e GIT_COMMIT="$COMMIT" \
  ${EVAL_LIMIT:+-e EVAL_LIMIT=$EVAL_LIMIT} \
  ${EVAL_DATASET:+-e EVAL_DATASET=$EVAL_DATASET} \
  ${RERANKER_MODEL:+-e RERANKER_MODEL=$RERANKER_MODEL} \
  -v "hf://buckets/$BUCKET:/bucket" \
  "$IMAGE" \
  bash -c "$SETUP && $CMD"
