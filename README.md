# rag-pipeline
Проект-НИР по настройке RAG-пайпланов

# Общий workflow

1. Подготовка набора данных и нарезание на чанки. Преобразуем статьи, документы и проч. в формат, который потом можно будеть преобразовать в эмбеддинг.
2. Подготовка набора эмбеддингов. С помощью embedding-модели преобразовать набор данных в эмбеддинги (преврать в вектор)
3. Сохранить эмбеддинги в векторную БД (Qrant, PGvector и тд)
4. Преобразовать запрос от пользователя (request) в эмбеддинг (embed-request).
5. Получить из векторной БД документы (context), похожие на embed-request.
6. Подать на вход LLM request и context, для получения ответа (response).

# Как запустить модель из HuggingFace на Ollama

[Документация Ollama](https://github.com/ollama/ollama/blob/main/docs/import.md)

1) Установка зависимостей: 

```
pip install --upgrade huggingface_hub
```

2. Скачивание нужной модели:

```
$ mkdir model_dir && cd model_dir/
$ huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir .
```

3. Создание Modelfile:

* для Safetensors- модели: `$ echo "FROM ." > Modelfile`

* для GGUF-модели: `$ echo "FROM {model.name}.gguf" > Modelfile`

4. Создание модели и зупуск в ollama:

```
$ ollama create deepSeek-r1-distill-qwen-1.5B
$ ollama run deepSeek-r1-distill-qwen-1.5B:latest
```

