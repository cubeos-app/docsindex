# CubeOS Document Indexer

Syncs documentation from a Git repository, indexes it into ChromaDB for RAG search, and serves a docs API for the CubeOS dashboard.

## Features

- **Git Sync**: Clones/pulls documentation from configurable Git repo
- **Chunking**: Splits documents into optimal chunks for embedding
- **Embeddings**: Uses Ollama with nomic-embed-text model
- **ChromaDB Storage**: Stores embeddings in ChromaDB v2 API
- **Docs API**: Serves `/api/v1/docs/tree`, `/api/v1/docs/{path}`, `/api/v1/docs/search`
- **Filesystem Fallback**: Serves docs from filesystem even when ChromaDB/Ollama are unavailable
- **Scheduled Sync**: Runs periodically to keep docs up-to-date
- **Offline Ready**: Works after initial sync even without internet

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/docs/tree` | Document tree structure |
| GET | `/api/v1/docs/{path}` | Fetch document content |
| GET | `/api/v1/docs/search?q=query` | Search documents (ChromaDB + fallback) |
| GET | `/api/v1/index/status` | Indexing status |
| POST | `/api/v1/index/trigger` | Trigger re-index |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | HTTP listen address |
| `DOCS_REPO_URL` | `https://github.com/cubeos-app/docs.git` | Git repository URL |
| `DOCS_LOCAL_PATH` | `/cubeos/docs` | Local path to store docs |
| `OLLAMA_HOST` | `10.42.24.1` | Ollama server host |
| `OLLAMA_PORT` | `6030` | Ollama server port |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `CHROMADB_HOST` | `10.42.24.1` | ChromaDB server host |
| `CHROMADB_PORT` | `6031` | ChromaDB server port |
| `COLLECTION_NAME` | `cubeos_docs` | ChromaDB collection name |
| `SYNC_INTERVAL_HOURS` | `6` | Sync interval (0 = disable periodic) |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

## Deployment

### As a Swarm stack (production)

```bash
docker stack deploy -c /cubeos/coreapps/cubeos-docsindex/appconfig/docker-compose.yml \
  --resolve-image never cubeos-docsindex
```

### NPM Proxy Route (required)

Add a proxy host in Nginx Proxy Manager to route docs API requests from the dashboard:

- **Source**: `cubeos.cube` path `/api/v1/docs`
- **Target**: `http://10.42.24.1:6032`
- **Type**: Proxy pass (preserve path)

Or add a `location` block to the dashboard's nginx.conf:

```nginx
location /api/v1/docs/ {
    proxy_pass http://10.42.24.1:6032;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
}
```

## Building

```bash
cd src/
docker build -t cubeos-docsindex .
docker tag cubeos-docsindex ghcr.io/cubeos-app/cubeos-docsindex:latest
```

## License

Apache 2.0
