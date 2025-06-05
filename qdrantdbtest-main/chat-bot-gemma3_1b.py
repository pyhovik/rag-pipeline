from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import json
import argparse

# 1. Подключение к модели Gemma через Ollama (запущенный локально)
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(context, question, model_name="gemma3:1b"):
    prompt = f"""
Ты — помощник, который отвечает на вопросы, основываясь на предоставленной информации.
Информация: {context}
Вопрос: {question}
Ответ:
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("response", "Нет ответа").strip()
    else:
        print("Ошибка при генерации:", response.text)
        return "Не удалось сгенерировать ответ."

# 2. Загрузка модели эмбеддингов
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 3. Функция поиска релевантных документов
def search_similar_documents(query, collection_name, top_k=5):
    query_embedding = embedding_model.encode([query])[0].tolist()

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k
        )
        return results.points
    except Exception as e:
        print("Ошибка при поиске в Qdrant:", str(e))
        return []

# 4. Основной цикл работы
def ask_question(collection_name, model_name="gemma3:1b"):
    while True:
        question = input("\nЗадайте ваш вопрос (или 'exit' для выхода): ")
        if question.lower() == "exit":
            break

        print("Ищу информацию...\n")
        documents = search_similar_documents(question, collection_name, top_k=5)

        if not documents:
            print("Ничего не найдено.")
            continue

        # Извлечение и декодирование текста
        contexts = []
        for doc in documents:
            payload = doc.payload
            node_content = json.loads(payload["_node_content"])
            encoded_text = node_content["text"]
            decoded_text = encoded_text.encode('utf-8').decode('utf-8')
            contexts.append(decoded_text)

        full_context = "\n\n".join(contexts[:3])  # используем первые 3 самых релевантных

        print("Генерирую ответ...")
        answer = generate_answer(full_context, question, model_name=model_name)

        print("\nОтвет:")
        print(answer)

# 5. Точка входа
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск и ответы на основе данных из Qdrant")
    parser.add_argument("--collection", type=str, default="my_collection",
                        help="Имя коллекции в Qdrant (по умолчанию: web_knowledge)")
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                        help="Адрес сервера Qdrant (по умолчанию: localhost)")
    parser.add_argument("--model", type=str, default="gemma3:1b",
                        help="Модель LLM для генерации ответа (по умолчанию: gemma3:1b)")
    args = parser.parse_args()

    # Подключение к Qdrant с возможностью указать хост
    client = QdrantClient(host=args.qdrant_host, port=6333, timeout=10)

    # Проверяем, существует ли коллекция
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if args.collection not in collection_names:
            print(f"❌ Коллекция '{args.collection}' не найдена в списке: {collection_names}")
        else:
            ask_question(args.collection, model_name=args.model)
    except Exception as e:
        print("Ошибка при проверке коллекций:", str(e))
