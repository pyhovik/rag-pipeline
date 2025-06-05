import os

os.environ["USER_AGENT"] = "dima-test"

from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents.base import Document
from qdrant_client import QdrantClient, models


def load_pages(file_path: str = "./source_urls.txt") -> list[Document]:
    """
    Функция предназначена для парсинга списка веб-страниц и преобразования его в 
    пригодные для векторизации объекты.
    Функция принимает как аргумент текстовый файл, содержащий ссылки на веб-страницы.
    """
    # загружаем список урлов для парсинга
    with open(file_path) as f:
        urls = []
        for line in f:
            urls.append(line.strip())

    ## Класс WebBaseLoader работает плохо на страницах конфлюенса - парсит совсем не то, что нужно
    # loader = WebBaseLoader(web_path=urls, verify_ssl=False, default_parser="html5lib")
    # docs = loader.load()

    # парсим урлы - получаем список объектов Documents
    loader = AsyncHtmlLoader(web_path=urls, verify_ssl=False)
    loaded_docs = loader.load()
    # преобразуем html в текст
    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(loaded_docs)

    # запись контента в файлы чтобы понимать, с какими данными вообще работаем
    for doc in docs:
        with open(f"/tmp/{doc.metadata['title'].replace(' ', '_')}.txt", "w") as f:
            f.write(doc.page_content)
    
    return docs

def init_qdrant_client() -> QdrantClient:

    # Initialize the client
    client = QdrantClient(":memory:")   # Create in-memory Qdrant instance, for testing, CI/CD
    model_name = "BAAI/bge-small-en"

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
    
    
    collection_name="demo_collection"
    # client = QdrantClient(path="/tmp/qrant.db")   # Persists changes to disk
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=client.get_embedding_size(model_name), 
                distance=models.Distance.COSINE
            ),  # size and distance are model dependent
        )
    # Prepare your documents, metadata, and IDs
    docs = [
        "Qdrant has a LangChain integration for chatbots.",
        "Qdrant has a LlamaIndex integration for agents.",
    ]
    metadata = [
        {"source": "Langchain-docs"},
        {"source": "Llama-index-docs"},
    ]
    ids = [42, 2]

    metadata_with_docs = [
        {"document": doc, "source": meta["source"]} for doc, meta in zip(docs, metadata)
    ]
    client.upload_collection(
        collection_name=collection_name,
        vectors=[models.Document(text=doc, model=model_name) for doc in docs],
        payload=metadata_with_docs,
        ids=ids,
    )
    return client



    # If you want to change the model:
    # client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    # List of supported models: https://qdrant.github.io/fastembed/examples/Supported_Models

    # Use the new add() instead of upsert()
    # This internally calls embed() of the configured embedding model
    # client.add(
    #     collection_name="demo_collection",
    #     documents=docs,
    #     metadata=metadata,
    #     ids=ids
    # )




if __name__ == "__main__":
    # docs = load_pages()
    # print(docs[0].metadata)
    # print(docs[0].id)
    client = init_qdrant_client()
    collection_name="demo_collection"
    model_name = "BAAI/bge-small-en"

    search_result = client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text="Which integration is best for agents?", 
            model=model_name
        )
    ).points
    print(search_result)


    # search_result = client.query(
    #     collection_name=collection_name,
    #     query_text="This is a query document"
    # )
    # print(search_result)



    from qdrant_client import QdrantClient

    # Initialize the client
    client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

    # Prepare your documents, metadata, and IDs
    docs = [
        "Qdrant has a LangChain integration for chatbots.",
        "Qdrant has a LlamaIndex integration for agents.",
    ]
    metadata = [
        {"source": "Langchain-docs"},
        {"source": "Llama-index-docs"},
    ]
    ids = [42, 2]

    # Use the new add method
    client.add(
        collection_name="demo_collection",
        documents=docs,
        metadata=metadata,
        ids=ids
    )

    search_result = client.query(
        collection_name="demo_collection",
        query_text="Which integration is best for agents?"
    )
    print(search_result)