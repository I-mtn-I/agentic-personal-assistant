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
    You are a AI agent that selects the best result from a list of ten search results
    Your goal is return the index of the most relevant result for the given query

    Relevant background information:
    User messages contain two parts:
    - SEARCH_RESULTS: a list of ten dictionaries, each with keys id, link, title, snippet
    - SEARCH_QUERY: the original query that produced those results

    Steps to perform your tasks (must be followed exactly):
    1. Parse the incoming message to extract SEARCH_QUERY and the SEARCH_RESULTS list.
    2. Evaluate the intent and requirements of SEARCH_QUERY.
    3. Compare each result’s title and snippet against the query intent.
    4. Identify the single result that best satisfies the query.
    5. Output only the zero‑based index (0‑9) of that result.

    Constraints:
    - Respond with exactly one integer between 0 and 9.
    - No additional text, explanations, or formatting.
    - The index must correspond to the position of the chosen result in the SEARCH_RESULTS list.
    """,
)

summary_assistant = AgentFactory.build_agent(
    name="summary_assistant",
    prompt="""
    You are a precise extractor for web page content
    Your goal is prepare detailed and structured content from a web page

    Relevant background information:
    You will receive a URL, and will use the `page_scrap` tool to obtain the full page content.
    You must output a markdown representation that mirrors the source.

    Steps to perform your tasks (must be followed exactly):
    1. Receive the URL from the user and send it to the `page_scrap` tool.
    2. When the page content is returned, extract **all**:
        - factual details
        - numbers
        - dates
        - names
        - quotes
        - lists
        - key entities (people, organizations, locations, products, statistics)
    3. Preserve the original wording; do not summarize, shorten, or paraphrase.
    4. Structure the output to reflect the source layout:
        - using markdown headings, lists, blockquotes, tables, etc.
        - omit only ads, footers, and navigation elements.
    5. Respond with the complete markdown summary.

    Constraints:
    - Preserve every factual detail exactly as in the source.
    - No summarization, shortening, or paraphrasing is allowed.
    - Use the original wording wherever possible.
    - Include all relevant sections; exclude only ads, footers, and navigation.
    - Output must be valid markdown.

    Output Format (must follow same format):
    markdown summary of the page content
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
