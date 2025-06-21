
import os
os.environ["USER_AGENT"] = "indexing-pipeline"

import bs4
import html2text
import requests
import uuid
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from config import RagConfig

print("Embeddings model init...")
embeddings = HuggingFaceEmbeddings(model_name=RagConfig.EMBEDDING_MODEL)

print("Qdrant client init...")
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qdrant.db"
)

try:
    client.delete_collection(RagConfig.COLLECTION_NAME)
except:
    print(f"{RagConfig.COLLECTION_NAME} does not exist")
finally:
    client.create_collection(
        collection_name=RagConfig.COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=RagConfig.EMBEDDING_MODEL_DIM, # EMBEDDING_MODEL dimensions - размерность эмбеддинг-модели
            distance=models.Distance.COSINE
        ),  # size and distance are model dependent
    )

print("VectoreStore init...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=RagConfig.COLLECTION_NAME,
    embedding=embeddings
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
    html = requests.get(url).text
    soup = bs4.BeautifulSoup(html, "html.parser")
    metadata = {
        "source": url,
        "title": soup.title.string.split(" - Eltex Documentation")[0],
        "id": uuid.uuid4()
    }
    page_content = html2text.html2text(html)
    document = Document(
        page_content=page_content,
        metadata=metadata
    )
    doc_splits = splitter.split_documents([document])
    vector_store.add_documents(doc_splits)
    print(f"Document '{metadata['title']}' was splitted ({len(doc_splits)}) and was loaded to vectore store")


if __name__ == "__main__":
    with open(RagConfig.SOURCE_URLS_FILE) as f:
        for url in f:
            load_doc_to_vectorstore(url.strip()) 