# GraphRAG with Tensorlake

Production-ready GraphRAG pipeline built on top of Tensorlake, showing how to go from a local graph-based RAG workflow to a deployed, observable API endpoint.

For the full walkthrough and architecture deep dive, see the blog post:  
**[Building a Production-Ready GraphRAG Pipeline with TensorLake](https://www.tensorlake.ai/blog/building-a-production-ready-graphrag-pipeline-with-tensorlake)**.

## What this example shows

- End-to-end GraphRAG pipeline (ingestion → graph construction → query)
- Using Tensorlake Document AI for PDF parsing and chunking
- Building and traversing a knowledge graph on top of vector search
- Packaging the pipeline as a Tensorlake application and deploying it as an API
- Running the same code both locally and on Tensorlake Cloud

## Setup

1. **Create and activate a virtualenv**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Set environment variables (local run)**

```bash
export OPENAI_API_KEY=your_openai_api_key
export TENSORLAKE_API_KEY=your_tensorlake_api_key
```

On Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_openai_api_key"
setx TENSORLAKE_API_KEY "your_tensorlake_api_key"
```

## Running locally

The example includes a `__main__` block that runs a test query against a sample PDF:

```bash
python tensorlake_graphrag.py
```

This will:

- Download and parse the sample PDF
- Build the vector index and knowledge graph
- Execute a GraphRAG query over the graph
- Print the final answer to stdout

## Deploying to Tensorlake

Once you have the Tensorlake CLI configured and your secrets set:

```bash
tensorlake secrets set TENSORLAKE_API_KEY=<your_tensorlake_key>
tensorlake secrets set OPENAI_API_KEY=<your_openai_key>

tensorlake deploy tensorlake_graphrag.py
```

After deployment, you can call the application via the Tensorlake API using the input schema shown in the blog post and in `tensorlake_graphrag.py`.

For detailed deployment, observability, and architecture notes, refer to the blog:  
**[Building a Production-Ready GraphRAG Pipeline with TensorLake](https://www.tensorlake.ai/blog/building-a-production-ready-graphrag-pipeline-with-tensorlake)**.
