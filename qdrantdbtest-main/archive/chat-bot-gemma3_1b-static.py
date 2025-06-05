from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import json

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

# 3. Подключение к Qdrant
client = QdrantClient(host="localhost", port=6333)

# 4. Функция поиска релевантных документов
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

# 5. Основной цикл работы
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

# 6. Точка входа
if __name__ == "__main__":
    COLLECTION_NAME = "web_knowledge"  # ← замени на имя своей коллекции
    MODEL_NAME = "gemma3:1b"  # можно попробовать "gemma:20b" если доступна

    # Проверяем, существует ли коллекция
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if COLLECTION_NAME not in collection_names:
            print(f"❌ Коллекция '{COLLECTION_NAME}' не найдена в списке: {collection_names}")
        else:
            ask_question(COLLECTION_NAME, model_name=MODEL_NAME)
    except Exception as e:
        print("Ошибка при проверке коллекций:", str(e))
