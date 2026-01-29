"""
GraphRAG Pipeline - Tensorlake Application

A serverless GraphRAG implementation based on Nir Diamant's notebook.
https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/graph_rag.ipynb
"""

import hashlib
import heapq
import os

import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from tensorlake.applications import application, function, File

from models import Chunk, Edge, IngestInput, IngestResult, QueryInput, QueryResult
from config import (
    graph_rag_image,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    EDGE_WEIGHT_ALPHA,
    EDGE_WEIGHT_BETA,
)
from neo4j_ops import (
    ensure_vector_index,
    store_chunks,
    store_edges,
    vector_search,
    get_chunk_by_id,
    get_neighbors_with_weights,
    get_chunks_by_ids,
)


# ---------------------------------------------------------------------------
# Document Processing Functions
# ---------------------------------------------------------------------------

@function(image=graph_rag_image, cpu=2, memory=4)
def parse_document(file: File) -> str:
    """Extract text content from a document."""
    content = file.content.decode("utf-8")
    return content


@function(image=graph_rag_image, cpu=1, memory=2)
def chunk_text(text: str, document_id: str) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Uses RecursiveCharacterTextSplitter logic:
    - chunk_size: 1000 characters
    - chunk_overlap: 200 characters
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence end within last 100 chars
            for sep in ['. ', '.\n', '\n\n', '\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > CHUNK_SIZE - 200:  # Found good break point
                    end = start + last_sep + len(sep)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text:  # Only add non-empty chunks
            chunk_id = hashlib.sha256(
                f"{document_id}:{chunk_index}".encode()
            ).hexdigest()[:16]

            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index
            ))
            chunk_index += 1

        start = end - CHUNK_OVERLAP
        if start >= len(text) - CHUNK_OVERLAP:
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding Functions
# ---------------------------------------------------------------------------

@function(image=graph_rag_image, cpu=1, memory=1)
def generate_embedding(text: str) -> list[float]:
    """Generate embedding for a text using OpenAI."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Concept Extraction Functions (spaCy + LLM)
# ---------------------------------------------------------------------------

@function(image=graph_rag_image, cpu=2, memory=4)
def extract_concepts(text: str) -> list[str]:
    """
    Extract named entities and key concepts from text.

    Combines:
    1. spaCy NER for named entities (PERSON, ORG, GPE, WORK_OF_ART)
    2. LLM-based concept extraction for general concepts
    """
    nlp = spacy.load("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()

    doc = nlp(text)

    # Extract named entities
    named_entities = [
        ent.text.lower() for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT", "LAW"]
    ]

    # Extract key noun phrases as concepts
    noun_phrases = [
        chunk.text.lower() for chunk in doc.noun_chunks
        if len(chunk.text.split()) >= 2  # Multi-word phrases only
    ]

    # Use LLM for additional concept extraction
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract 3-5 key concepts from the text. Return only the concepts, one per line."
            },
            {
                "role": "user",
                "content": f"Extract key concepts from:\n\n{text[:1500]}"
            }
        ],
        temperature=0,
        max_tokens=100
    )

    llm_concepts = [
        line.strip().lower()
        for line in response.choices[0].message.content.split('\n')
        if line.strip()
    ]

    # Combine and deduplicate, lemmatizing for consistency
    all_concepts = set()
    for concept in named_entities + noun_phrases + llm_concepts:
        lemmatized = ' '.join([
            lemmatizer.lemmatize(word) for word in concept.split()
        ])
        all_concepts.add(lemmatized)

    return list(all_concepts)[:20]  # Limit to top 20 concepts


# ---------------------------------------------------------------------------
# Graph Construction Functions
# ---------------------------------------------------------------------------

@function(image=graph_rag_image, cpu=2, memory=8)
def build_edges(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    concepts_list: list[list[str]]
) -> list[Edge]:
    """
    Build edges between chunks based on semantic similarity and shared concepts.

    Edge weight formula (from Nir Diamant's implementation):
        weight = alpha * similarity_score + beta * normalized_shared_concepts

    Where:
        - alpha = 0.7 (weight for semantic similarity)
        - beta = 0.3 (weight for concept overlap)
        - normalized_shared_concepts = len(shared) / min(len(concepts1), len(concepts2))
    """
    edges = []
    n = len(chunks)

    # Compute similarity matrix
    embeddings_np = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_np)

    # Build edges for pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            similarity_score = similarity_matrix[i][j]

            if similarity_score < SIMILARITY_THRESHOLD:
                continue

            # Calculate shared concepts
            concepts_i = set(concepts_list[i]) if concepts_list[i] else set()
            concepts_j = set(concepts_list[j]) if concepts_list[j] else set()
            shared_concepts = list(concepts_i & concepts_j)

            # Calculate normalized shared concepts score
            max_possible_shared = min(len(concepts_i), len(concepts_j))
            normalized_shared = (
                len(shared_concepts) / max_possible_shared
                if max_possible_shared > 0 else 0
            )

            # Calculate edge weight
            edge_weight = (
                EDGE_WEIGHT_ALPHA * similarity_score +
                EDGE_WEIGHT_BETA * normalized_shared
            )

            edges.append(Edge(
                source_id=chunks[i].id,
                target_id=chunks[j].id,
                weight=float(edge_weight),
                shared_concepts=shared_concepts
            ))

    return edges


# ---------------------------------------------------------------------------
# Query Engine Functions (Dijkstra-like traversal)
# ---------------------------------------------------------------------------

@function(image=graph_rag_image, cpu=2, memory=4)
def traverse_and_answer(
    query: str,
    query_embedding: list[float],
    start_chunk_ids: list[str],
    document_ids: list[str] | None,
    max_context_length: int = 4000
) -> tuple[str, list[str], list[str]]:
    """
    Dijkstra-like graph traversal to gather context and generate answer.

    Algorithm (from Nir Diamant's QueryEngine):
    1. Start from most similar chunks (via vector search)
    2. Use priority queue with distances = 1/similarity
    3. Expand to neighbors based on edge weights
    4. At each step, check if we have enough context to answer
    5. Stop when answer is complete or context limit reached
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    lemmatizer = WordNetLemmatizer()

    # Priority queue: (distance, chunk_id)
    # Distance = 1/similarity (lower is better)
    priority_queue = []
    distances = {}

    # Initialize with start chunks
    for chunk_id in start_chunk_ids:
        priority = 1.0  # Start nodes have priority 1
        heapq.heappush(priority_queue, (priority, chunk_id))
        distances[chunk_id] = priority

    # Traversal state
    traversal_path = []
    visited_concepts = set()
    context_chunks = []
    accumulated_context = ""

    while priority_queue and len(accumulated_context) < max_context_length:
        current_priority, current_chunk_id = heapq.heappop(priority_queue)

        # Skip if we've found a better path
        if current_priority > distances.get(current_chunk_id, float('inf')):
            continue

        if current_chunk_id in traversal_path:
            continue

        # Add current chunk to traversal
        traversal_path.append(current_chunk_id)
        chunk = get_chunk_by_id(current_chunk_id)

        if chunk is None:
            continue

        context_chunks.append(chunk)
        accumulated_context += "\n\n" + chunk.text if accumulated_context else chunk.text

        # Check if we have enough context to answer
        if len(accumulated_context) > 1000:  # Minimum context before checking
            is_complete, answer = _check_answer_completeness(
                client, query, accumulated_context
            )
            if is_complete and answer:
                return answer, [c.id for c in context_chunks], traversal_path

        # Get chunk concepts and lemmatize
        chunk_concepts = set()
        if chunk.concepts:
            for concept in chunk.concepts:
                lemmatized = ' '.join([
                    lemmatizer.lemmatize(word) for word in concept.split()
                ])
                chunk_concepts.add(lemmatized)

        # Only expand if we have new concepts
        if not chunk_concepts.issubset(visited_concepts):
            visited_concepts.update(chunk_concepts)

            # Get neighbors and add to priority queue
            neighbors = get_neighbors_with_weights(current_chunk_id)

            for neighbor_id, edge_weight in neighbors:
                # Distance through current node
                new_distance = current_priority + (1.0 / edge_weight if edge_weight > 0 else 10.0)

                if new_distance < distances.get(neighbor_id, float('inf')):
                    distances[neighbor_id] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor_id))

    # Generate final answer if we didn't find complete answer during traversal
    answer = _generate_final_answer(client, query, accumulated_context)

    return answer, [c.id for c in context_chunks], traversal_path


def _check_answer_completeness(client: OpenAI, query: str, context: str) -> tuple[bool, str]:
    """Check if context provides complete answer to query."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Determine if the context provides a complete answer to the query.
                Respond in this exact format:
                COMPLETE: Yes or No
                ANSWER: [Your answer if complete, or "Incomplete" if not]"""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext: {context[:3000]}"
            }
        ],
        temperature=0,
        max_tokens=500
    )

    response_text = response.choices[0].message.content

    is_complete = "COMPLETE: Yes" in response_text
    answer = ""
    if is_complete and "ANSWER:" in response_text:
        answer = response_text.split("ANSWER:")[1].strip()

    return is_complete, answer


def _generate_final_answer(client: OpenAI, query: str, context: str) -> str:
    """Generate final answer from accumulated context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Answer the query based only on the provided context.
                Be concise and accurate. If the context doesn't contain enough
                information, say so."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context[:4000]}\n\nQuery: {query}\n\nAnswer:"
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main Applications
# ---------------------------------------------------------------------------

@application(
    image=graph_rag_image,
    secrets=["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD"]
)
@function(cpu=1, memory=2)
def ingest_document(file: File, input: IngestInput) -> IngestResult:
    """
    Ingest a document into the GraphRAG knowledge graph.

    Pipeline:
    1. Parse document → extract text
    2. Chunk text → overlapping segments (1000 chars, 200 overlap)
    3. Generate embeddings → parallel via .map()
    4. Extract concepts → parallel via .map() (spaCy + LLM)
    5. Build edges → similarity + shared concepts
    6. Store in Neo4j → nodes with vector index + edges
    """
    # Ensure vector index exists
    ensure_vector_index()

    # Step 1: Parse document
    text = parse_document(file)

    # Step 2: Chunk text
    chunks = chunk_text(text, input.document_id)

    if not chunks:
        return IngestResult(
            document_id=input.document_id,
            chunk_count=0,
            edge_count=0
        )

    # Step 3: Generate embeddings in parallel
    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embedding.map(texts)

    # Step 4: Extract concepts in parallel
    concepts_list = extract_concepts.map(texts)

    # Attach embeddings and concepts to chunks
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
        chunk.concepts = concepts_list[i]

    # Step 5: Store chunks in Neo4j (with embeddings for vector search)
    store_chunks(chunks)

    # Step 6: Build edges based on similarity and shared concepts
    edges = build_edges(chunks, embeddings, concepts_list)

    # Step 7: Store edges in Neo4j
    store_edges(edges)

    return IngestResult(
        document_id=input.document_id,
        chunk_count=len(chunks),
        edge_count=len(edges)
    )


@application(
    image=graph_rag_image,
    secrets=["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD"]
)
@function(cpu=1, memory=2)
def query_knowledge_graph(input: QueryInput) -> QueryResult:
    """
    Query the GraphRAG knowledge graph using Dijkstra-like traversal.

    Pipeline:
    1. Embed query → OpenAI embeddings
    2. Vector search → find top-k similar chunks
    3. Graph traversal → Dijkstra with edge weights
    4. Answer check → stop early if complete
    5. Generate answer → from accumulated context
    """
    # Step 1: Embed the query
    query_embedding = generate_embedding(input.question)

    # Step 2: Vector search for starting nodes
    matches = vector_search(query_embedding, input.document_ids, input.top_k)
    start_ids = [m[0] for m in matches]

    if not start_ids:
        return QueryResult(
            answer="No relevant documents found for your query.",
            sources=[],
            traversal_path=[]
        )

    # Step 3-5: Traverse graph and generate answer
    answer, sources, traversal_path = traverse_and_answer(
        query=input.question,
        query_embedding=query_embedding,
        start_chunk_ids=start_ids,
        document_ids=input.document_ids,
        max_context_length=4000
    )

    return QueryResult(
        answer=answer,
        sources=sources,
        traversal_path=traversal_path
    )
