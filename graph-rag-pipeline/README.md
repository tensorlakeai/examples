# GraphRAG on Tensorlake

A serverless GraphRAG implementation that scales automatically. Based on [Nir Diamant's GraphRAG notebook](https://github.com/NirDiamant/RAG_Techniques).

## Architecture

```
Document → Parse → Chunk → [Embed, Extract Concepts] → Build Edges → Neo4j
                              (parallel)                 (parallel)

Query → Embed → Vector Search → Graph Traversal → Generate Answer
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set up Neo4j

Create a Neo4j Aura instance or run locally:

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 3. Configure secrets

```bash
tensorlake secrets set OPENAI_API_KEY <your-key>
tensorlake secrets set NEO4J_URI neo4j+s://xxx.databases.neo4j.io
tensorlake secrets set NEO4J_PASSWORD <password>
```

### 4. Deploy

```bash
tensorlake deploy app.py
```

## Usage

### Ingest a document

```bash
curl -X POST https://api.tensorlake.ai/applications/ingest_document \
  -F "file=@document.txt" \
  -F 'input={"document_id": "doc_001"}'
```

### Query the knowledge graph

```bash
curl -X POST https://api.tensorlake.ai/applications/query_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main causes of climate change?",
    "document_ids": ["doc_001"],
    "max_hops": 3,
    "top_k": 3
  }'
```

## Project Structure

```
graph_rag/
├── app.py           # Main Tensorlake application
├── models.py        # Pydantic data models
├── config.py        # Configuration and image setup
├── neo4j_ops.py     # Neo4j database operations
├── requirements.txt
└── README.md
```

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE` - Size of text chunks (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `SIMILARITY_THRESHOLD` - Minimum similarity for edges (default: 0.7)
- `TOP_K_EDGES` - Max edges per chunk (default: 20)

## How It Works

1. **Ingestion**: Documents are chunked, embedded, and concept-extracted in parallel using `.map()`. Edges are built based on embedding similarity and shared concepts, then stored in Neo4j with a vector index.

2. **Query**: The query is embedded, similar chunks are found via vector search, the graph is traversed to gather related context, and an answer is generated using GPT-4.

3. **Scaling**: Tensorlake automatically scales workers based on load. No documents? Scales to zero. Thousands of documents? Spins up parallel workers.

## References

- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Nir Diamant's RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)
- [Tensorlake Documentation](https://docs.tensorlake.ai)
