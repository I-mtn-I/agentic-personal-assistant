import logging

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel

from ai_gateway.domain import AgentFactory

# Constants
MAX_RESULTS = 10
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    )
}


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
        snippet = (
            str(snippet_tag.text.strip()) if snippet_tag else "No description available"
        )  # Extract short text under link

        results.append(SearchResults(id=i, url=link, title=title, result_snippet=snippet))

    return results


@tool(description="Useful to gather content of a web page")
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


# create agents
best_result_chooser = AgentFactory.build_agent(
    name="best_result_picker",
    prompt="""
    You are an AI Agent trained to select the best result from a list of ten search results.
    All user messages you receive in this conversation will have the format of:
    SEARCH_RESULTS: [{}, {}, {}]
    SEARCH_QUERY: 'search query ran to get the above 10 results'

    Each element of SEARCH_RESULTS is a dictionary with the following keys:
    - id: The unique identifier of the search result
    - link: The URL of the search result
    - title: The title of the search result
    - snippet: A brief description of the search result

    You'll follow exactly below steps to accomplish your task:
    - understand the SEARCH_QUERY and its need
    - check the SEARCH_RESULTS and choose the best one based on the SEARCH_QUERY
    - return the index of the best result.

    You'll only respond with an index number. Since SEARCH_QUERY has 10 elements,
    your responses to this conversation should always be a single integer between 0-9.
    """,
)

summary_assistant = AgentFactory.build_agent(
    name="summary_assistant",
    prompt="""
    You are a precise extractor for web page content.
    Your job is to prepare detailed and structured content from a web page.

    CRITICAL RULES:
    1. Preserve ALL factual details, numbers, dates, names, quotes and lists exactly
    2. Do NOT summarize, shorten, or paraphrase - extract comprehensively
    3. Use original wording as much as possible
    4. Structure output to mirror source in markdown format
    5. Extract ALL key entities (people, orgs, locations, products, stats)
    6. Include ALL relevant sections - omit only ads, footers, navigation

    You'll follow below steps exactly to accomplish your task:
    - You'll recieve a URL and you'll send this to page_scrap tool
    - once you receive the content, you'll prepare a summary according to above rules
    - you'll respond summary in markdown format.

    OUTPUT FORMAT: markdown summary
    """,
    tools=[page_scrap],
)


async def search_web(query: str) -> str:
    try:
        search_results = duckduckgo_search(query)  # Perform search
        results_str = "\n".join(str(result) for result in search_results)  # Prepare results string
        best_result = await best_result_chooser.ask(results_str)  # Get best result
        choosen_link = search_results[int(best_result)].url  # Extract chosen URL
        summary = await summary_assistant.ask(choosen_link)  # Get summary
        return summary
    except Exception as e:
        logging.error(f"An error occurred during web search: {e}")  # Log the error
        return "An error occurred while processing your request."
