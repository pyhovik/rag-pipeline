
import os
os.environ["USER_AGENT"] = "indexing-pipeline"

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from config import RagConfig

print("Documents load via WebBaseLoader...")
# загружаем список урлов для парсинга
with open(RagConfig.SOURCE_URLS_FILE) as f:
    urls = [line.strip() for line in f]
# парсим урлы - получаем список объектов Documents
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs={"parse_only": bs4.SoupStrainer('div',{'id': 'main-content'})}    # выборочный парсинг страниц intdocs
)
docs = loader.load()

print("Documents splitter init...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=RagConfig.CHUNK_SIZE,
    chunk_overlap=RagConfig.CHUNK_OVERLAP,
     add_start_index=True,  # track index in original document   
)
doc_splits = splitter.split_documents(docs)

print("Save web-pages content into /tmp ...")
# запись контента в файлы чтобы понимать, с какими данными вообще работаем
for doc in docs:
    page_id = doc.metadata["source"].split("=")[1]
    with open(f"/tmp/{page_id}.txt", "w") as f:
        f.write(doc.page_content)

print("Embeddings model init...")
embeddings = HuggingFaceEmbeddings(model_name=RagConfig.EMBEDDING_MODEL)

print("Qdrant client init...")
client = QdrantClient(
    #location = ":memory:",
    path="/tmp/qdrant.db"
)
if not client.collection_exists(RagConfig.COLLECTION_NAME):
    client.create_collection(
        collection_name=RagConfig.COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024, # EMBEDDING_MODEL dimensions - размерность эмбеддинг-модели
            distance=models.Distance.COSINE
        ),  # size and distance are model dependent
    )

print("VectoreStore init...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=RagConfig.COLLECTION_NAME,
    embedding=embeddings
)
vector_store.add_documents(doc_splits)
