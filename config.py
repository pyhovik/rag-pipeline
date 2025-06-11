from dataclasses import dataclass


@dataclass
class RagConfig:
    CHAT_MODEL = "gemma3:1b"
    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
    COLLECTION_NAME = "test-collection"
    SOURCE_URLS_FILE = "./source_urls.txt"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
