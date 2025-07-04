import bs4
from langchain_community.document_loaders import WebBaseLoader
import requests

# Only keep post title, headers, and content from the full HTML.
# bs4_strainer = 
loader = WebBaseLoader(
    web_paths=("https://docs.eltex-co.ru/pages/viewpage.action?pageId=612047179",),
    bs_kwargs={"parse_only": bs4.SoupStrainer('div',{'id': 'main-content'})},
)
docs = loader.load()
print(f"Total characters: {len(docs[0].page_content)}")


loader = WebBaseLoader(
    web_paths=("https://docs.eltex-co.ru/pages/viewpage.action?pageId=612047179",),
    # bs_kwargs={"parse_only": bs4.SoupStrainer('div',{'id': 'main-content'})},
)
docs = loader.load()
print(f"Total characters: {len(docs[0].page_content)}")


html = requests.get("https://docs.eltex-co.ru/pages/viewpage.action?pageId=612047179").text

soup = bs4.BeautifulSoup(html, "html.parser")
# print(soup.get_text(strip=True))
import html2text
text = html2text.html2text(html)
print(text)
# print(f"Total characters: {len(soup.get_text())}")