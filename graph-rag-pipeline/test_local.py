"""
Local testing script for GraphRAG pipeline.

Run this to test the pipeline locally before deploying to Tensorlake.

Prerequisites:
1. Set environment variables (or use .env file)
2. Run Neo4j locally: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
3. Install dependencies: pip install -r requirements.txt
4. Download spaCy model: python -m spacy download en_core_web_sm
5. Download NLTK data: python -c "import nltk; nltk.download('wordnet')"
"""

import os
from pathlib import Path

# Set environment variables for local testing
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "password")

from tensorlake.applications import run_local_application, File

from app import ingest_document, query_knowledge_graph
from models import IngestInput, QueryInput


def test_ingestion():
    """Test document ingestion locally."""
    print("=" * 60)
    print("Testing Document Ingestion")
    print("=" * 60)

    # Read sample document
    sample_path = Path(__file__).parent / "samples" / "climate_change.txt"

    if not sample_path.exists():
        print(f"Sample file not found: {sample_path}")
        return None

    with open(sample_path, "rb") as f:
        file_content = f.read()

    print(f"Ingesting document: {sample_path.name}")
    print(f"Document size: {len(file_content)} bytes")

    # Run ingestion
    result = run_local_application(
        ingest_document,
        file=File(content=file_content, filename="climate_change.txt"),
        input=IngestInput(document_id="climate_001")
    )

    print(f"\nIngestion Result:")
    print(f"  Document ID: {result.document_id}")
    print(f"  Chunks created: {result.chunk_count}")
    print(f"  Edges created: {result.edge_count}")

    return result


def test_queries():
    """Test querying the knowledge graph."""
    print("\n" + "=" * 60)
    print("Testing Queries")
    print("=" * 60)

    questions = [
        "What is the main cause of climate change?",
        "How do greenhouse gases affect the atmosphere?",
        "What are the effects of climate change on sea levels?",
        "What solutions exist for addressing climate change?"
    ]

    for question in questions:
        print(f"\n{'─' * 60}")
        print(f"Question: {question}")
        print("─" * 60)

        result = run_local_application(
            query_knowledge_graph,
            input=QueryInput(
                question=question,
                document_ids=["climate_001"],
                max_hops=3,
                top_k=3
            )
        )

        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources: {len(result.sources)} chunks used")
        print(f"Traversal path: {len(result.traversal_path)} nodes visited")

        if result.traversal_path:
            print(f"Path: {' → '.join(result.traversal_path[:5])}", end="")
            if len(result.traversal_path) > 5:
                print(f" ... ({len(result.traversal_path) - 5} more)")
            else:
                print()


def main():
    print("\n" + "=" * 60)
    print("GraphRAG Local Testing")
    print("=" * 60)
    print("\nMake sure:")
    print("1. Neo4j is running (docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest)")
    print("2. OPENAI_API_KEY is set")
    print("3. Dependencies are installed (pip install -r requirements.txt)")
    print()

    # Check environment
    if os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return

    # Test ingestion
    result = test_ingestion()

    if result and result.chunk_count > 0:
        # Test queries
        test_queries()
    else:
        print("\nSkipping query tests - no chunks were ingested")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
