**Déploiement Cloud Run — RAG Digestats**

*Architecture : Hybrid RAG — BGE-M3 dense + sparse, BM25, RRF fusion, CrossEncoder reranking, LangGraph*

---

## Prérequis

- Compte GCP avec facturation activée
- `gcloud` CLI installé et authentifié
- Docker installé localement
- Tes clés API dans un fichier `.env`

---

## Structure du projet pour le déploiement

```
Agentic_RAG/
├── app.py                  ← FastAPI + chatbot
├── assemble_graph.py       ← Graphe LangGraph
├── indexing_pipeline.py    ← Pipeline hybrid RAG (BGE-M3 + BM25 + Qdrant)
├── document_grader.py
├── generate_answer.py
├── query_pipeline.py
├── rewrite_question.py
├── web_tool.py
├── models.py
├── docs/                   ← Les PDFs réglementaires
│   ├── arrete_9_avril_2018.pdf
│   └── ...
├── qdrant/                 ← Index Qdrant (généré localement avant le build)
├── Dockerfile
├── requirements.txt
└── .dockerignore
```

---

## Variables d'environnement requises

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Clé API Gemini (LLM + évaluation) |
| `TAVILY_API_KEY` | Clé API Tavily (recherche web) |
| `APP_PASSWORD` | Mot de passe d'accès au chatbot |
| `LANGFUSE_PUBLIC_KEY` | *(optionnel)* Tracing Langfuse |
| `LANGFUSE_SECRET_KEY` | *(optionnel)* Tracing Langfuse |
| `LANGFUSE_HOST` | *(optionnel)* URL Langfuse (défaut : cloud.langfuse.com) |

---

## Étapes de déploiement

### 0. Construire l'index Qdrant en local (avant le build Docker)

L'index vectoriel doit être généré localement et copié dans l'image. Il n'est **pas** reconstruit automatiquement au démarrage.

```bash
# Depuis la racine du projet, avec ton .env chargé
python indexing_pipeline.py --reindex
```

Cela crée (ou met à jour) le dossier `qdrant/` avec l'index BGE-M3 dense + sparse.
Vérifier que `qdrant/` est bien présent avant de passer à l'étape suivante.

---

### 1. Créer le projet GCP

```bash
gcloud projects create rag-digestats --name="RAG Digestats"
gcloud config set project rag-digestats
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### 2. Créer le dépôt Artifact Registry

```bash
gcloud artifacts repositories create rag-repo \
    --repository-format=docker \
    --location=europe-west1 \
    --description="Images Docker RAG Digestats"
```

### 3. Build et push l'image

```bash
# Configurer Docker pour GCP
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build (depuis la racine du projet — inclut le dossier qdrant/)
docker build -t europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest .

# Push
docker push europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest
```

### 4. Déployer sur Cloud Run

```bash
gcloud run deploy rag-digestats \
    --image europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest \
    --region europe-west1 \
    --platform managed \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 2 \
    --timeout 300 \
    --set-env-vars GOOGLE_API_KEY=xxx,TAVILY_API_KEY=xxx,APP_PASSWORD=xxx \
    --allow-unauthenticated
```

> **Notes :**
> - `--min-instances 0` = scale to zero (pas de coût au repos)
> - `--memory 4Gi` est nécessaire pour charger BGE-M3 + CrossEncoder en mémoire
> - `--timeout 300` donne 5 min par requête (le reranking peut être lent sur CPU)

### 5. Accéder au chatbot

Cloud Run donne une URL type :
```
https://rag-digestats-xxxx-ew.a.run.app
```
Le chatbot est directement accessible à la racine (protégé par `APP_PASSWORD`).
L'API REST est sur `/api/chat` et `/api/health`.

---

## Mise à jour des documents

Quand ton client a de nouveaux PDFs réglementaires :

1. Ajoute-les dans le dossier `docs/`
2. Réindexe localement :
```bash
python indexing_pipeline.py --reindex
```
3. Rebuild et redéploie :
```bash
docker build -t europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest .
docker push europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest
gcloud run deploy rag-digestats \
    --image europe-west1-docker.pkg.dev/rag-digestats/rag-repo/rag-app:latest \
    --region europe-west1
```

---

## Sécurité

Le chatbot est protégé par un mot de passe (`APP_PASSWORD`) directement dans l'interface. Pour restreindre l'accès au niveau réseau en plus :

```bash
# Supprimer l'accès public
gcloud run services remove-iam-policy-binding rag-digestats \
    --region europe-west1 \
    --member="allUsers" \
    --role="roles/run.invoker"

# Ajouter ton client par email
gcloud run services add-iam-policy-binding rag-digestats \
    --region europe-west1 \
    --member="user:client@email.com" \
    --role="roles/run.invoker"
```

---

## Coûts estimés

| Service | Coût mensuel estimé |
|---|---|
| Cloud Run (scale to zero, 1-3 users) | 2-5 € |
| Gemini Flash API | 1-5 € |
| Artifact Registry | < 1 € |
| **Total** | **~5-15 €/mois** |