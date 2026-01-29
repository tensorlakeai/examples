"""Pydantic models for GraphRAG pipeline."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A text chunk from a document."""
    id: str
    text: str
    document_id: str
    chunk_index: int
    embedding: list[float] | None = None
    concepts: list[str] | None = None


class Edge(BaseModel):
    """An edge between two chunks in the knowledge graph."""
    source_id: str
    target_id: str
    weight: float
    shared_concepts: list[str] = []


class IngestInput(BaseModel):
    """Input for document ingestion."""
    document_id: str


class IngestResult(BaseModel):
    """Result of document ingestion."""
    document_id: str
    chunk_count: int
    edge_count: int


class QueryInput(BaseModel):
    """Input for querying the knowledge graph."""
    question: str
    document_ids: list[str] | None = None
    max_hops: int = 3
    top_k: int = 3


class QueryResult(BaseModel):
    """Result of a query."""
    answer: str
    sources: list[str]
    traversal_path: list[str]
