
import os
os.environ["USER_AGENT"] = "dima-test"

from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from config import RagConfig

COLLECTION_NAME = RagConfig.COLLECTION_NAME
EMBEDDING_MODEL = RagConfig.EMBEDDING_MODEL
SOURCE_URLS_FILE = RagConfig.SOURCE_URLS_FILE
CHUNK_SIZE = RagConfig.CHUNK_SIZE
CHUNK_OVERLAP = RagConfig.CHUNK_OVERLAP

print("Documents load...")
# загружаем список урлов для парсинга
with open(RagConfig.SOURCE_URLS_FILE) as f:
    urls = [line.strip() for line in f]
# парсим урлы - получаем список объектов Documents
loader = AsyncHtmlLoader(web_path=urls, verify_ssl=False)
loaded_docs = loader.load()

print("Documents transform...")
# преобразуем html в текст
html2text = Html2TextTransformer()
docs = html2text.transform_documents(loaded_docs)

print("Documents splitter init...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
doc_splits = splitter.split_documents(docs)
print(len(doc_splits))
print(doc_splits[0])

print("Сохранение веб-страницы в директорию /tmp ...")
# запись контента в файлы чтобы понимать, с какими данными вообще работаем
for doc in docs:
    with open(f"/tmp/{doc.metadata['title'].replace(' ', '_')}.txt", "w") as f:
        f.write(doc.page_content)

print("Embeddings model init...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Qdrant client init...")
client = QdrantClient(":memory:")   # or QdrantClient(path="/tmp/qrant.db")ß
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024, # EMBEDDING_MODEL dimensions - размерность эмбеддинг-модели
            distance=models.Distance.COSINE
        ),  # size and distance are model dependent
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# client.upload_collection(
#     collection_name=collection_name,
#     vectors=[models.Document(text=doc.page_content, model=model_name) for doc in docs],
#     payload=[doc.metadata for doc in loaded_docs],
# )
# search_result = client.query_points(
#     collection_name=collection_name,
#     query=models.Document(
#         text="What is ECCM?", 
#         model=model_name
#     )
# )
# for point in search_result.points:
#     print(point.payload["title"], point.score)
