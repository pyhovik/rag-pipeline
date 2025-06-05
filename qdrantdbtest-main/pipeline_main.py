import os
import requests
import argparse
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext

def parse_page(url):
    """Парсинг страницы с улучшенной обработкой текста"""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Ошибка загрузки {url}: {str(e)}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Извлечение текста с сохранением структуры
    text_parts = []
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'article']:
        for element in soup.find_all(tag):
            text = element.get_text(" ", strip=True)
            if text and len(text) > 20:
                text_parts.append(text)

    if not text_parts:
        print(f"⚠ Не удалось извлечь текст из {url}")
        return None

    full_text = "\n".join(text_parts)
    images = [img['src'] for img in soup.find_all('img', src=True) if img.get('src')]

    return Document(
        text=full_text,
        metadata={
            "url": url,
            "images": images,
            "source": "eltex-docs"
        }
    )

def main(args):
    # Конфигурация
    QDRANT_HOST = args.qdrant_host
    QDRANT_PORT = 6333
    COLLECTION_NAME = args.collection_name
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    URLS_FILE = "urls.txt"

    # 1. Загрузка документов
    print("🔍 Загрузка документов...")
    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]

    documents = [doc for url in urls if (doc := parse_page(url)) is not None]

    if not documents:
        print("🛑 Нет документов для обработки")
        return

    # 2. Настройка модели эмбеддингов
    print("\n🧠 Инициализация модели...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu"
    )

    # 3. Подключение к Qdrant
    print("\n🔌 Подключение к Qdrant...")
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=30,
        prefer_grpc=False  # Важно отключить для совместимости
    )

    # Удаление старой коллекции (для теста)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"♻ Удалена старая коллекция {COLLECTION_NAME}")
    except Exception as e:
        print(f"ℹ️ Не удалось удалить коллекцию: {str(e)}")

    # Создание новой коллекции (только dense-векторы)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=384,  # Размерность для all-MiniLM-L6-v2
            distance=qdrant_models.Distance.COSINE
        )
    )
    print(f"✅ Создана коллекция {COLLECTION_NAME}")

    # 4. Настройка хранилища (без гибридного поиска)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=False  # Ключевое исправление!
    )

    # 5. Настройка разбиения на чанки
    node_parser = SentenceSplitter(
        chunk_size=2048,
        chunk_overlap=50
    )

    # 6. Индексация документов
    print("\n📦 Начало индексации...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[node_parser],
        metadata_mode="exclude",
        show_progress=True
    )

    # 7. Проверка результатов
    print("\n🔍 Проверка результатов:")
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Всего точек: {collection_info.points_count}")

    if collection_info.points_count == 0:
        print("\n❌ Векторы не добавлены! Проверьте:")
        print("1. Логи Qdrant на сервере")
        print("2. Доступность порта 6333")
        print("3. Размер чанков (попробуйте уменьшить до 256)")
    else:
        print("\n🎉 Индексация успешно завершена!")
        print(f"Добавлено векторов: {collection_info.points_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Индексация документов в Qdrant.")
    parser.add_argument('--collection-name', type=str, default='test-collection',
                        help='Название коллекции в Qdrant')
    parser.add_argument('--qdrant-host', type=str, default='127.0.0.1',
                        help='IP-адрес или хост Qdrant сервера')

    args = parser.parse_args()

    main(args)
