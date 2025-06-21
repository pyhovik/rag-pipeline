from dataclasses import dataclass


@dataclass
class RagConfig:
    CHAT_MODEL = "gemma3:1b"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_DIM = 384
    COLLECTION_NAME = "test-collection"
    SOURCE_URLS_FILE = "./source_urls.txt"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
