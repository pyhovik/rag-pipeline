from qdrant_client import QdrantClient, models
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RagConfig


print("Qdrant client init...")
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qdrant.db"
)

COLLECTION_NAME = "demo-collection"
MODEL_NAME = "BAAI/bge-small-en"

try:
    client.delete_collection(COLLECTION_NAME)
except:
    print(f"{COLLECTION_NAME} does not exist")
finally:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(MODEL_NAME),
            distance=models.Distance.COSINE
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
    print(models.Document)

    # client.upload_collection(
    #     collection_name=COLLECTION_NAME,
    #     vectors=
    # )    
    print(f"Document '{doc[0].metadata['title']}' was splitted ({len(doc_splits)}) and was loaded to vectore store")


if __name__ == "__main__":
    with open(RagConfig.SOURCE_URLS_FILE) as f:
        for url in f:
            load_doc_to_vectorstore(url.strip())
