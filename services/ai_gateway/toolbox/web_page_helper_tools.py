import logging

import requests
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel

# Constants
MAX_RESULTS = 10
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")}


class SearchResults(BaseModel):
    id: int
    url: str
    title: str
    result_snippet: str


def duckduckgo_search(query: str) -> list[SearchResults]:
    # Clean up the query string
    if query[0] == '"':
        query = query[1:-1]
    if query.startswith("NEWS:"):
        query = query[len("NEWS:") :] + "&iar=news"
    if query[0] == " ":
        query = query[1:]
    query = query.replace(" ", "+")

    # Construct the search URL
    url = f"https://html.duckduckgo.com/html/?q={query}"

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SearchResults] = []

    for i, result in enumerate(soup.find_all("div", class_="result"), start=1):
        if i > MAX_RESULTS:
            break

        title_tag = result.find("a", class_="result__a")
        if not title_tag:
            continue

        link = str(title_tag["href"])  # Extract link
        title = title_tag.text  # Extract title
        snippet_tag = result.find("a", class_="result__snippet")
        snippet = str(snippet_tag.text.strip()) if snippet_tag else "No description available"  # Extract short text under link

        results.append(SearchResults(id=i, url=link, title=title, result_snippet=snippet))

    return results


def page_scrap(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url=url)
        content = trafilatura.extract(downloaded, include_formatting=True, include_links=True)
        if content:
            return content
        else:
            return "Nothing found"
    except Exception as e:
        print(f"Failed to extract content from website. Error: {e}")
        return ""
