import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import signal
import sys

# === Настройки ===
DOMAIN = "http://xgu.ru/wiki/%D0%97%D0%B0%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D0%B0%D1%8F_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B0"  # Замените на ваш домен
USER_AGENT = "KnowledgeCrawler/1.0"
DELAY = 2  # Пауза между запросами (в секундах)
OUTPUT_FILE = "urls.txt"

# === Парсер robots.txt ===
rp = RobotFileParser()
robots_url = f"{urlparse(DOMAIN).scheme}://{urlparse(DOMAIN).netloc}/robots.txt"

try:
    rp.set_url(robots_url)
    rp.read()
    print(f"[+] robots.txt загружен: {robots_url}")
except Exception as e:
    print(f"[!] Не удалось прочитать robots.txt: {e}")

def can_fetch(url):
    return rp.can_fetch(USER_AGENT, url)

# === Сбор ссылок ===
visited = set()

def save_urls():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for url in sorted(visited):
            f.write(url + "\n")
    print(f"\n[+] Сохранено {len(visited)} ссылок в '{OUTPUT_FILE}'")

def signal_handler(sig, frame):
    print("\n[!] Обнаружено прерывание (Ctrl+C). Сохраняю собранные ссылки...")
    save_urls()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def crawl(start_url):
    queue = [start_url]

    while queue:
        url = queue.pop(0)

        if url in visited or not can_fetch(url):
            continue

        try:
            print(f"[+] Обрабатываю: {url}")
            time.sleep(DELAY)
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)

            if response.status_code != 200:
                print(f"[-] Получен код {response.status_code}: {url}")
                visited.add(url)
                continue

            visited.add(url)
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                next_url = urljoin(url, href)

                # Проверяем, что это внутренняя ссылка
                if urlparse(next_url).netloc == urlparse(DOMAIN).netloc:
                    if next_url not in visited and next_url not in queue:
                        queue.append(next_url)

        except Exception as e:
            print(f"[!] Ошибка при обработке {url}: {e}")
            visited.add(url)

# === Запуск сбора ===
try:
    crawl(DOMAIN)
    print("\n[+] Сканирование завершено.")
finally:
    save_urls()