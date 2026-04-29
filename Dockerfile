FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from FlagEmbedding import BGEM3FlagModel; \
from sentence_transformers import CrossEncoder; \
BGEM3FlagModel('BAAI/bge-m3', use_fp16=True); \
CrossEncoder('BAAI/bge-reranker-v2-m3')"

ENV HF_HUB_OFFLINE=1

COPY . .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"]
