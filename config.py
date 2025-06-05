from dataclasses import dataclass

@dataclass
class RagConfig:
    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
    # Supported embedding-models:
    # BAAI/bge-base-en
    # BAAI/bge-base-en-v1.5
    # BAAI/bge-large-en-v1.5
    # BAAI/bge-small-en
    # BAAI/bge-small-en-v1.5
    # BAAI/bge-small-zh-v1.5
    # mixedbread-ai/mxbai-embed-large-v1
    # snowflake/snowflake-arctic-embed-xs
    # snowflake/snowflake-arctic-embed-s
    # snowflake/snowflake-arctic-embed-m
    # snowflake/snowflake-arctic-embed-m-long
    # snowflake/snowflake-arctic-embed-l
    # jinaai/jina-clip-v1
    # Qdrant/clip-ViT-B-32-text
    # sentence-transformers/all-MiniLM-L6-v2
    # jinaai/jina-embeddings-v2-base-en
    # jinaai/jina-embeddings-v2-small-en
    # jinaai/jina-embeddings-v2-base-de
    # jinaai/jina-embeddings-v2-base-code
    # jinaai/jina-embeddings-v2-base-zh
    # jinaai/jina-embeddings-v2-base-es
    # thenlper/gte-base
    # thenlper/gte-large
    # nomic-ai/nomic-embed-text-v1.5
    # nomic-ai/nomic-embed-text-v1.5-Q
    # nomic-ai/nomic-embed-text-v1
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # intfloat/multilingual-e5-large
    # jinaai/jina-embeddings-v3
    COLLECTION_NAME = "test-collection"
    SOURCE_URLS_FILE = "./source_urls.txt"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100