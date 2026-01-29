"""Neo4j operations for GraphRAG pipeline."""

import os
from neo4j import GraphDatabase

from models import Chunk, Edge
from config import EMBEDDING_DIMENSIONS


def get_driver():
    """Get Neo4j driver from environment variables."""
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
    )


def ensure_vector_index():
    """Create vector index if it doesn't exist."""
    driver = get_driver()
    with driver.session() as session:
        # Create vector index for similarity search
        session.run(f"""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIMENSIONS},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

        # Create index on chunk id for fast lookups
        session.run("""
            CREATE INDEX chunk_id_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.id)
        """)

        # Create index on document_id for filtering
        session.run("""
            CREATE INDEX chunk_document_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.document_id)
        """)
    driver.close()


def store_chunks(chunks: list[Chunk]):
    """Store chunks as nodes in Neo4j."""
    driver = get_driver()
    with driver.session() as session:
        # Batch insert for performance
        session.run("""
            UNWIND $chunks AS chunk
            MERGE (c:Chunk {id: chunk.id})
            SET c.text = chunk.text,
                c.document_id = chunk.document_id,
                c.chunk_index = chunk.chunk_index,
                c.embedding = chunk.embedding,
                c.concepts = chunk.concepts
        """, {
            "chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "embedding": chunk.embedding,
                    "concepts": chunk.concepts or []
                }
                for chunk in chunks
            ]
        })
    driver.close()


def store_edges(edges: list[Edge]):
    """Store edges as relationships in Neo4j."""
    driver = get_driver()
    with driver.session() as session:
        # Batch insert for performance
        session.run("""
            UNWIND $edges AS edge
            MATCH (a:Chunk {id: edge.source_id})
            MATCH (b:Chunk {id: edge.target_id})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.weight = edge.weight,
                r.shared_concepts = edge.shared_concepts
        """, {
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                    "shared_concepts": edge.shared_concepts
                }
                for edge in edges
            ]
        })

        # Create reverse edges (undirected graph)
        session.run("""
            UNWIND $edges AS edge
            MATCH (a:Chunk {id: edge.source_id})
            MATCH (b:Chunk {id: edge.target_id})
            MERGE (b)-[r:SIMILAR_TO]->(a)
            SET r.weight = edge.weight,
                r.shared_concepts = edge.shared_concepts
        """, {
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                    "shared_concepts": edge.shared_concepts
                }
                for edge in edges
            ]
        })
    driver.close()


def vector_search(
    query_embedding: list[float],
    document_ids: list[str] | None,
    top_k: int
) -> list[tuple[str, float]]:
    """
    Find chunks similar to query embedding using vector search.

    Returns list of (chunk_id, similarity_score) tuples.
    """
    driver = get_driver()

    with driver.session() as session:
        if document_ids:
            # Filter by document IDs
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k * 2, $embedding)
                YIELD node, score
                WHERE node.document_id IN $document_ids
                RETURN node.id AS id, score
                ORDER BY score DESC
                LIMIT $top_k
            """, {
                "embedding": query_embedding,
                "top_k": top_k,
                "document_ids": document_ids
            })
        else:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
                YIELD node, score
                RETURN node.id AS id, score
                ORDER BY score DESC
            """, {
                "embedding": query_embedding,
                "top_k": top_k
            })

        matches = [(record["id"], record["score"]) for record in result]

    driver.close()
    return matches


def get_chunk_by_id(chunk_id: str) -> Chunk | None:
    """Get a single chunk by its ID."""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk {id: $id})
            RETURN c.id AS id,
                   c.text AS text,
                   c.document_id AS document_id,
                   c.chunk_index AS chunk_index,
                   c.concepts AS concepts
        """, {"id": chunk_id})

        record = result.single()
        if record is None:
            driver.close()
            return None

        chunk = Chunk(
            id=record["id"],
            text=record["text"],
            document_id=record["document_id"],
            chunk_index=record["chunk_index"],
            concepts=record["concepts"]
        )

    driver.close()
    return chunk


def get_chunks_by_ids(chunk_ids: list[str]) -> list[Chunk]:
    """Get multiple chunks by their IDs."""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $ids
            RETURN c.id AS id,
                   c.text AS text,
                   c.document_id AS document_id,
                   c.chunk_index AS chunk_index,
                   c.concepts AS concepts
        """, {"ids": chunk_ids})

        chunks = [
            Chunk(
                id=record["id"],
                text=record["text"],
                document_id=record["document_id"],
                chunk_index=record["chunk_index"],
                concepts=record["concepts"]
            )
            for record in result
        ]

    driver.close()
    return chunks


def get_neighbors_with_weights(chunk_id: str) -> list[tuple[str, float]]:
    """
    Get neighboring chunks and edge weights for graph traversal.

    Returns list of (neighbor_id, edge_weight) tuples.
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk {id: $id})-[r:SIMILAR_TO]-(neighbor:Chunk)
            RETURN neighbor.id AS neighbor_id,
                   r.weight AS weight
            ORDER BY r.weight DESC
        """, {"id": chunk_id})

        neighbors = [
            (record["neighbor_id"], record["weight"])
            for record in result
        ]

    driver.close()
    return neighbors


def get_all_embeddings(document_id: str) -> dict[str, list[float]]:
    """Get all embeddings for a document."""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk {document_id: $document_id})
            RETURN c.id AS id, c.embedding AS embedding
        """, {"document_id": document_id})

        embeddings = {r["id"]: r["embedding"] for r in result}

    driver.close()
    return embeddings


def delete_document(document_id: str):
    """Delete all chunks and edges for a document."""
    driver = get_driver()

    with driver.session() as session:
        # Delete all relationships first
        session.run("""
            MATCH (c:Chunk {document_id: $document_id})-[r]-()
            DELETE r
        """, {"document_id": document_id})

        # Delete all nodes
        session.run("""
            MATCH (c:Chunk {document_id: $document_id})
            DELETE c
        """, {"document_id": document_id})

    driver.close()
