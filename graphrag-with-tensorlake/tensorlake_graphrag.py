# graphrag_app.py
# =============================================================================
# TensorLake-powered GraphRAG tutorial
# =============================================================================

import os
import logging
import hashlib
from typing import List, Dict, Tuple, Any
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from pydantic import BaseModel, Field

# Only light-weight / safe imports at top level

# Quiet logging by default – only show real errors
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("tensorlake").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Disable Chroma anonymized telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# =============================================================================
# Custom container image – added system deps for spacy / numpy
# =============================================================================
from tensorlake.applications import Image

image = (
    Image(base_image="python:3.12-slim")
    .run("apt-get update && apt-get install -y --no-install-recommends "
         "gcc g++ libstdc++6 libblas-dev liblapack-dev "
         "ca-certificates && rm -rf /var/lib/apt/lists/*")
    .run("pip install --no-cache-dir tensorlake chromadb langchain langchain-openai langchain-community openai "
         "networkx spacy tqdm numpy scikit-learn nltk pydantic python-dotenv")
    .run("python -m spacy download en_core_web_sm")
)

# =============================================================================
# Pydantic models
# =============================================================================
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the context provides a complete answer")
    answer: str = Field(description="The answer if complete, else empty")


class GraphRAGInput(BaseModel):
    pdf_url: str
    query: str

# =============================================================================
# KnowledgeGraph – spaCy loaded once per container
# =============================================================================
class KnowledgeGraph:
    def __init__(self):
        import networkx as nx
        from nltk.stem import WordNetLemmatizer
        import spacy

        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache: Dict[str, List[str]] = {}
        # Load spaCy eagerly so the model is reused across invocations
        self.nlp = spacy.load("en_core_web_sm")
        self.edges_threshold = 0.78  # slightly lowered – 0.8 was too strict

    def reset(self) -> None:
        """Clear graph-specific state while keeping heavy resources (spaCy) loaded."""
        self.graph.clear()
        self.concept_cache.clear()

    def build_graph(self, splits, llm, embedding_model):
        self._add_nodes(splits)
        embeddings = embedding_model.embed_documents([s.page_content for s in splits])
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _extract_concepts(self, splits, llm):
        # Limit concurrency to avoid hammering LLM / OpenAI rate limits
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                for i, split in enumerate(splits)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(splits),
                desc="Extracting concepts",
                disable=True,
            ):
                node = futures[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts

    def _extract_concepts_and_entities(self, content, llm):
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        from langchain_core.prompts import PromptTemplate
        chain = PromptTemplate.from_template(
            "Extract 5–12 most important nouns, noun phrases and key domain-specific terms "
            "from this text. Be specific, avoid very generic words. "
            "Return only the comma-separated list.\n\nText:\n{text}\n\nConcepts:"
        ) | llm.with_structured_output(Concepts)

        general_concepts = chain.invoke({"text": content}).concepts_list

        all_concepts = list(set(named_entities + general_concepts))
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _add_edges(self, embeddings):
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges", disable=True):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    concept_overlap = len(shared_concepts) / max(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']), 1)
                    edge_weight = 0.65 * similarity_score + 0.35 * concept_overlap
                    self.graph.add_edge(node1, node2, weight=edge_weight, similarity=similarity_score, shared_concepts=list(shared_concepts))


# Reusable KnowledgeGraph instance so heavy resources (like spaCy) are loaded once
_GLOBAL_KNOWLEDGE_GRAPH = KnowledgeGraph()

# =============================================================================
# QueryEngine – traversal logic still has known issues
# =============================================================================
class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Query:\n{query}\n\n"
            "Context:\n{context}\n\n"
            "Instructions:\n"
            "- Decide if the context is sufficient to answer the query.\n"
            "- If YES, write a NEW, concise, abstracted answer in your own words.\n"
            "- Do NOT copy sentences verbatim from the context.\n"
            "- Limit the answer to 2–4 sentences.\n\n"
            "Is the answer complete? (Yes/No)\n"
            "Final Answer (only if Yes):"
        )
        return prompt | self.llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> bool:
        """Return True if the context is sufficient to answer the query."""
        response = self.answer_check_chain.invoke({"query": query, "context": context})
        return response.is_complete

    def _expand_context(
        self, query: str, relevant_docs
    ) -> Dict[str, Any]:
        # ────────────────────────────────────────────────────────────────
        # IMPORTANT: traversal logic still has duplication & early marking bug
        # Recommended minimal fix: remove the inner if-block that adds neighbors prematurely
        # ────────────────────────────────────────────────────────────────
        MAX_CONTEXT_CHARS = 12000
        MAX_NODES = 15

        filtered_content: Dict[int, str] = {}
        expanded_context = ""
        traversal_path: List[Dict[str, Any]] = []
        final_answer = None

        relevant_nodes = [int(doc.metadata.get('chunk_id')) for doc in relevant_docs
                          if doc.metadata.get('chunk_id') is not None]

        if not relevant_nodes:
            return "", [], {}, "No relevant starting chunks found."

        priority_queue = [(0, node) for node in relevant_nodes]
        heapq.heapify(priority_queue)
        distances: Dict[int, float] = {node: 0 for node in relevant_nodes}
        visited = set()
        parents: Dict[int, Tuple[int, float]] = {}

        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            current_content = self.knowledge_graph.graph.nodes[current_node]['content']
            filtered_content[current_node] = current_content

            if len(expanded_context) < MAX_CONTEXT_CHARS:
                expanded_context = (
                    expanded_context + "\n\n" + current_content
                    if expanded_context
                    else current_content
                )

            parent_info = parents.get(current_node)
            traversal_path.append(
                {
                    "node_id": current_node,
                    "from_node": parent_info[0] if parent_info else None,
                    "edge_weight": parent_info[1] if parent_info else None,
                    "distance": distances[current_node],
                    "concepts": self.knowledge_graph.graph.nodes[current_node].get("concepts", []),
                }
            )

            if len(traversal_path) >= MAX_NODES:
                break

            is_complete = self._check_answer(query, expanded_context)
            if is_complete:
                break

            for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                edge_weight = self.knowledge_graph.graph[current_node][neighbor].get('weight', 0.5)
                new_distance = current_priority + (1 / max(edge_weight, 0.1))  # avoid div by zero

                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    parents[neighbor] = (current_node, edge_weight)
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        if not final_answer:
            from langchain_core.prompts import PromptTemplate
            response_prompt = PromptTemplate.from_template(
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer the question concisely in 2–4 sentences. "
                "Do not copy text verbatim from the context."
            )
            chain = response_prompt | self.llm
            final_answer = chain.invoke(
                {"context": expanded_context, "query": query}
            ).content.strip()

        return {
            "expanded_context": expanded_context,
            "traversal_path": traversal_path,
            "filtered_content": filtered_content,
            "answer": final_answer,
        }

    def query(self, query: str) -> Dict[str, Any]:
        relevant_docs = self._retrieve_relevant_documents(query)
        return self._expand_context(query, relevant_docs)

    def _retrieve_relevant_documents(self, query: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever.invoke(query)

# =============================================================================
# GraphRAG class
# =============================================================================
class GraphRAG:
    def __init__(self):
        self._initialized = False
        self._ingested_pdf = None
        self.doc_ai = None
        self.llm = None
        self.embeddings = None
        # Reuse a single KnowledgeGraph instance so heavy resources (spaCy) are reused
        self.knowledge_graph = _GLOBAL_KNOWLEDGE_GRAPH
        self.vector_store = None
        self.query_engine = None

    def _init_resources(self):
        if self._initialized:
            return

        from tensorlake.documentai import DocumentAI
        self.doc_ai = DocumentAI()

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

        self._initialized = True

    def ingest(self, pdf_url: str):
        self._init_resources()
        if self._ingested_pdf == pdf_url:
            return


        from tensorlake.documentai import ParsingOptions, ChunkingStrategy
        from langchain_core.documents import Document

        parse_id = self.doc_ai.read(
            file_url=pdf_url,
            parsing_options=ParsingOptions(
                chunking_strategy=ChunkingStrategy.PAGE,
                include_tables=True,
                include_figures=True,
                generate_summaries=True,
            )
        )
        result = self.doc_ai.wait_for_completion(parse_id)

        if not result or not result.chunks:
            raise RuntimeError("DocumentAI parsing failed")

        documents = []
        for i, chunk in enumerate(result.chunks):
            documents.append(Document(
                page_content=chunk.content,
                metadata={
                    "page": chunk.page_number,
                    "chunk_id": i,
                    "source": pdf_url,
                }
            ))


        from langchain_community.vectorstores import Chroma

        # Optional local persistence for Chroma (useful for dev; cloud persistence requires a volume/object store)
        persist_dir = os.getenv("CHROMA_PERSIST_DIR")  # e.g. "chroma_db"
        collection_name = "graphrag_" + hashlib.sha256(pdf_url.encode("utf-8")).hexdigest()[:16]

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        # Reset graph-specific state but keep spaCy model loaded
        self.knowledge_graph.reset()
        self.knowledge_graph.build_graph(documents, self.llm, self.embeddings)
        self.query_engine = QueryEngine(self.vector_store, self.knowledge_graph, self.llm)

        self._ingested_pdf = pdf_url

    def query(self, query: str) -> Dict:
        if not self._initialized:
            raise RuntimeError("Call ingest() first")
        return self.query_engine.query(query)

# =============================================================================
# Main application entrypoint
# =============================================================================
from tensorlake.applications import application, function, run_local_application, RequestContext

@application()
@function(image=image, secrets=["OPENAI_API_KEY", "TENSORLAKE_API_KEY"])
def graphrag_agent(input: GraphRAGInput) -> Dict:
    """
    TensorLake application entrypoint.

    Uses a single Pydantic input object so it works cleanly
    over the HTTP JSON API.
    """
    rag = GraphRAG()
    rag.ingest(input.pdf_url)
    result = rag.query(input.query)

    # Return only the final abstractive answer
    return {"answer": result["answer"]}

# =============================================================================
# Local testing
# =============================================================================
if __name__ == "__main__":
    test_pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    test_query = "What is the main contribution of the paper 'Attention Is All You Need'?"

    request = run_local_application(
        graphrag_agent,
        GraphRAGInput(pdf_url=test_pdf_url, query=test_query),
    )
    print("\nLocal test result:")
    print(request.output())