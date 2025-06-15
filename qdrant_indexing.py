from qdrant_client import QdrantClient
from qdrant_client.models import Document, VectorParams, Distance
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RagConfig


print("Qdrant client init...")
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qdrant.db"
)

print("Setup embedding model...")
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
client.set_model(RagConfig.EMBEDDING_MODEL)

COLLECTION_NAME = "demo-collection"
MODEL_NAME = "BAAI/bge-small-en"

try:
    client.delete_collection(COLLECTION_NAME)
except:
    print(f"{COLLECTION_NAME} does not exist")
finally:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=client.get_embedding_size(MODEL_NAME),
            distance=Distance.COSINE
        ),  # size and distance are model dependent
    )

print("Documents splitter init...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=RagConfig.CHUNK_SIZE,
    chunk_overlap=RagConfig.CHUNK_OVERLAP,
     add_start_index=True,  # track index in original document   
)


def load_doc_to_vectorstore(url: str) -> None:
    # загружаем список урлов для парсинга
    # парсим урлы - получаем список объектов Documents
    loader = WebBaseLoader(
        web_path=url,
        # bs_kwargs={"parse_only": bs4.SoupStrainer('div',{'id': 'main-content'})}    # выборочный парсинг страниц intdocs
    )
    doc = loader.load()
    doc[0].metadata["title"] = doc[0].metadata["title"].split(" - Eltex Documentation")[0]

    doc_splits = splitter.split_documents(doc)
    for doc in doc_splits:
        client.add(
            collection_name=COLLECTION_NAME,
            document
        )
        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=[Document(text=doc.page_content, model=MODEL_NAME)],
            payload=[{"content": doc.page_content}],
        )

    # client.upload_collection(
    #     collection_name=COLLECTION_NAME,
    #     vectors=
    # )    
    print(f"Document '{doc[0].metadata['title']}' was splitted ({len(doc_splits)}) and was loaded to vectore store")


if __name__ == "__main__":
    with open(RagConfig.SOURCE_URLS_FILE) as f:
        for url in f:
            load_doc_to_vectorstore(url.strip())
        
