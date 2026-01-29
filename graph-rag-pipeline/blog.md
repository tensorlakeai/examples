# The End of Database-Backed Workflow Engines: Building GraphRAG on Object Storage

GraphRAG sounds elegant in theory: build a knowledge graph from your documents, traverse it intelligently, get better answers than vanilla RAG.

Then you look at the compute requirements.

To build a GraphRAG system, you need to: parse documents, chunk text, generate embeddings for every chunk, extract concepts from every chunk, compute pairwise similarities, build graph edges, and store everything in a queryable format. For a single 100-page PDF, that's thousands of API calls, millions of similarity computations, and hours of processing.

Now imagine doing this for 10,000 documents. Or 100,000.

---

## What GraphRAG Actually Needs from Infrastructure

The algorithm is straightforward: chunk, embed, extract concepts, build edges, traverse. The infrastructure requirements are not.

**Parallel execution.** Documents are independent—processing them sequentially wastes time. You need a system that can spin up workers on demand and distribute work across them.

**Heterogeneous compute.** PDF parsing needs 4GB of memory. Embedding generation is I/O-bound waiting on API calls. Concept extraction needs CPU for NLP models. Running all of these on the same machine means over-provisioning for the hungriest step.

**Durable execution.** A 10-hour ingestion job will fail somewhere. Network timeout at hour 6. Rate limit at hour 8. OOM at hour 9. When step 3 fails and retries, it needs to read step 2's output from somewhere—that "somewhere" needs to survive container restarts. Without checkpointing to durable storage, you start over from zero.

**Job orchestration.** You need a cluster scheduler that spins up containers when work arrives and tears them down when it's done. Something has to track task dependencies—step 3 can't start until step 2 finishes. When you fan out to 1000 parallel tasks and 847 succeed, something has to retry the failures, collect partial results, and decide whether to proceed or abort.

---

## The DIY Stack

Building this yourself means assembling:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SELF-MANAGED GRAPHRAG STACK                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Kubernetes  │  │   Celery    │  │    Redis    │                 │
│  │   Cluster   │  │   Workers   │  │   Broker    │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │    Spark    │  │  Postgres   │  │     S3      │                 │
│  │   Cluster   │  │  Metadata   │  │   Buckets   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Your Glue Code                            │   │
│  │  • Fan-out logic       • Checkpointing    • Retry policies   │   │
│  │  • Result aggregation  • Idempotency      • Error handling   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Kubernetes** for container orchestration. But Kubernetes doesn't know anything about your jobs—it manages containers, not computations. It won't schedule your tasks, track dependencies, or handle fan-out.

**Celery + Redis** for task queuing. Note: queuing, not parallel execution. Celery distributes tasks to workers, but it's fundamentally a message broker with worker processes attached. It doesn't understand data locality, can't optimize task placement, and treats every task as independent. When you need real parallelism—map-reduce over 10,000 chunks, aggregating partial results, handling stragglers—Celery's primitives get you partway there. For the rest, you're writing glue code or reaching for Spark.

**Spark** for actual parallel compute. Now you're running a third system. Spark knows how to partition data, schedule parallel tasks, and aggregate results. But Spark wants to own the whole pipeline. Mixing Spark jobs with Celery tasks means shuffling data between systems, managing two job lifecycles, and debugging failures that span both.

**Postgres** for job metadata and durability. This is the state that workflow engines like Airflow and Temporal manage—but now you're building it yourself.

**The glue code.** You have a container orchestrator that doesn't understand jobs, a task queue that doesn't understand parallelism, and a compute engine that doesn't integrate with either. You're writing hundreds of lines to bridge these systems—and every bridge is a place where failures hide.

| Component | Setup Time | Ongoing Maintenance |
|-----------|------------|---------------------|
| Kubernetes cluster | 2-3 weeks | 10+ hrs/month |
| Celery workers + Redis | 1-2 weeks | 5 hrs/month |
| Spark (for real parallelism) | 2-3 weeks | 8 hrs/month |
| Postgres job tracking | 1 week | 2 hrs/month |
| Glue code bridging all systems | 3-4 weeks | 8 hrs/month |
| **Total** | **12-16 weeks** | **33+ hrs/month** |

And this assumes you get it right the first time. You won't.

Kubernetes was built for orchestrating long-running microservices, not bursty batch jobs. The Cluster Autoscaler checks for unschedulable pods every 10 seconds, then provisions nodes that take 30-60 seconds to come online. For a GraphRAG pipeline that needs to fan out to 500 workers immediately, that's minutes of latency before work even starts. The autoscaler [prioritizes stability over speed](https://scaleops.com/blog/kubernetes-cluster-autoscaler-best-practices-limitations-alternatives/)—a reasonable tradeoff for web services, but painful for batch processing.

This is why most GraphRAG implementations stay as notebooks. The infrastructure tax is too high.

---

## A Different Approach: Object Store Native Compute

For the past two years, we've been quietly building a new serverless compute stack for AI workloads at [Tensorlake](https://tensorlake.ai).

It powers our Document Ingestion API, which processes millions of documents every month across a heterogeneous fleet of machines—fully distributed, fully managed. Document processing was our testbed: OCR, layout detection, table extraction, entity recognition. Every document touches multiple models, multiple machines, multiple failure modes. If the infrastructure couldn't handle that, it couldn't handle anything.

But the compute stack itself is general purpose. It replaces the entire Kubernetes + Celery + Spark + Postgres stack with a single abstraction:

**Write your workflow as if it runs on a single machine. In production, it gets transparently distributed across CPUs and GPUs, and scales to whatever the workload demands.**

No queues to configure. No job schedulers to manage. No Spark clusters to provision. No glue code bridging systems that weren't designed to work together.

The key insight: use S3 as the backbone for durable execution instead of databases. AI workloads deal in unstructured data—documents, images, embeddings, model outputs. This data already lives in object storage. By building the execution engine around S3 rather than Postgres or Cassandra, we eliminated an entire class of serialization problems and made checkpointing nearly free.

---

## GraphRAG on Tensorlake

Each step runs as an isolated function with its own compute resources:

```python
@function(cpu=2, memory=4)
def parse_document(file: File) -> list[str]:
    return extract_pages(file)

@function(cpu=1, memory=1)
def generate_embedding(text: str) -> list[float]:
    return openai.embeddings.create(input=text, model="text-embedding-3-small")

@function(cpu=2, memory=4)
def extract_concepts(text: str) -> list[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]
```

The magic is in `.map()`. Fan out to thousands of workers with one line:

```python
@application()
@function()
def ingest_documents(files: list[File]) -> GraphStats:
    # Fan-out: parse all documents in parallel
    all_chunks: list[list[Chunk]] = parse_document.map(files)
    chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

    # Fan-out: embed and extract concepts in parallel
    embeddings = generate_embedding.map([c.text for c in chunks])
    concepts = extract_concepts.map([c.text for c in chunks])

    # Fan-in: build graph
    return build_and_store_graph(chunks, embeddings, concepts)
```

```
         ┌──────────────────────────────────────────────────────────┐
         │                    TENSORLAKE                            │
         │                                                          │
files ──►│  ┌─────────┐   ┌─────────┐   ┌─────────┐                │
         │  │ parse_1 │   │ parse_2 │   │ parse_3 │  ... (N files) │
         │  └────┬────┘   └────┬────┘   └────┬────┘                │
         │       └─────────────┴─────────────┘                      │
         │                     │                                    │
         │                     ▼                                    │
         │  ┌─────────┐   ┌─────────┐   ┌─────────┐                │
         │  │ embed_1 │   │ embed_2 │   │ embed_3 │  ... (M chunks)│
         │  └────┬────┘   └────┬────┘   └────┬────┘                │
         │       └─────────────┴─────────────┘                      │
         │                     │                                    │
         │                     ▼                                    │
         │            ┌───────────────┐                             │
         │            │  build_graph  │  (fan-in)                   │
         │            └───────┬───────┘                             │
         └────────────────────┼─────────────────────────────────────┘
                              ▼
                           Neo4j
```

When a function fails, Tensorlake doesn't re-execute successful steps—it reads the checkpointed output from S3 and continues. If the pipeline dies at chunk 847, the retry resumes from the last checkpoint, not from zero.

This isn't a batch job you run manually—it's a live HTTP endpoint. Deploy once, and it's available on-demand whenever someone wants to add a document to the knowledge graph:

```bash
curl -X POST https://api.tensorlake.ai/applications/ingest_documents \
  -F "files=@quarterly_report.pdf"
```

No documents in the queue? The system scales to zero. A thousand PDFs arrive at once? Tensorlake spins up workers to handle them in parallel. You're not paying for idle clusters or babysitting Spark jobs. The infrastructure responds to the workload, not the other way around.

---

## The Results

| Metric | Notebook Approach | Tensorlake Pipeline |
|--------|-------------------|---------------------|
| **1,000 docs ingestion** | 4.5 hours (sequential) | 23 minutes (parallel) |
| **Failure recovery** | Start over | Resume from checkpoint |
| **Memory required** | 32GB (all in memory) | 2GB per worker |
| **Concurrent ingestions** | 1 | Unlimited |

| Scenario | What Happens |
|----------|--------------|
| **10 documents** | Single container handles everything |
| **1,000 documents** | Tensorlake spawns parallel workers for each stage |
| **100,000 documents** | Same code, more workers, S3 handles intermediate data |
| **Failure at step 50,000** | Retry resumes from last checkpoint |

---

## Try It

```bash
git clone https://github.com/tensorlake/graph-rag-pipeline
cd graph-rag-pipeline

tensorlake secrets set OPENAI_API_KEY <your-key>
tensorlake secrets set NEO4J_URI neo4j+s://xxx.databases.neo4j.io
tensorlake secrets set NEO4J_PASSWORD <password>

tensorlake deploy app.py
```

For a proof-of-concept with 10 documents, a notebook is fine. For production with growing data, you need a pipeline that scales without the infrastructure tax.

---

*Built with [Tensorlake](https://tensorlake.ai) and [Neo4j](https://neo4j.com). See the [GraphRAG paper](https://arxiv.org/abs/2404.16130) for the original algorithm.*
