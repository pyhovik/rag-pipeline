from dataclasses import dataclass


@dataclass
class RagConfig:
    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
    COLLECTION_NAME = "test-collection"
    SOURCE_URLS_FILE = "./source_urls.txt"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
