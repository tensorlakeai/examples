"""Configuration for GraphRAG pipeline."""

from tensorlake.applications import Image

# Base image with all dependencies
graph_rag_image = Image().name("graph-rag").with_python_version("3.11").run(
    "pip install openai neo4j spacy numpy scikit-learn nltk"
).run(
    "python -m spacy download en_core_web_sm"
).run(
    "python -c \"import nltk; nltk.download('wordnet')\""
)

# Chunking configuration (from Nir Diamant's notebook)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Graph configuration (from Nir Diamant's notebook)
SIMILARITY_THRESHOLD = 0.8  # edges_threshold in original
EDGE_WEIGHT_ALPHA = 0.7     # Weight for semantic similarity
EDGE_WEIGHT_BETA = 0.3      # Weight for concept overlap

# Query configuration
DEFAULT_MAX_HOPS = 3
DEFAULT_TOP_K = 5
MAX_CONTEXT_LENGTH = 4000
