from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader

# загружаем список урлов для парсинга
with open("./source_urls.txt") as f:
    urls = []
    for line in f:
        urls.append(line.strip())
    print(urls)

## Класс WebBaseLoader работает плохо на страницах конфлюенса - парсит совсем не то, что нужно
# loader = WebBaseLoader(web_path=urls, verify_ssl=False, default_parser="html5lib")
# docs = loader.load()

# парсим урлы - получаем список объектов Documents
loader = AsyncHtmlLoader(web_path=urls, verify_ssl=False)
loaded_docs = loader.load()
# преобразуем html в текст
from langchain_community.document_transformers import Html2TextTransformer
html2text = Html2TextTransformer()
docs = html2text.transform_documents(loaded_docs)

# запись контента в файлы чтобы понимать, с какими данными вообще работаем
for doc in docs:
    with open(f"/tmp/{doc.metadata['title'].replace(' ', '_')}.txt", "w") as f:
        f.write(doc.page_content)
