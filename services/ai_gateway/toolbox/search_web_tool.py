import json
import logging
import re

from langchain.tools import tool

from ai_gateway.domain import AgentFactory
from ai_gateway.toolbox.web_page_helper_tools import (
    duckduckgo_search,
    page_scrap,
)

page_scrap_tool = tool(description="Useful to gather content of a web page")(page_scrap)


# create agents
best_result_chooser = AgentFactory.build_agent(
    name="best_result_picker",
    prompt="""
    You are an AI agent that selects the best result from a list of search results.
    Your goal is to return the zero-based index of the most relevant result for the given query.

    Relevant background information:
    You will receive a JSON object with:
    - search_query: the original query that produced the results
    - search_results: a list of dictionaries with keys id, url, title, result_snippet
    - result_count: the number of results in the list

    Steps to perform your tasks (must be followed exactly):
    1. Parse the incoming JSON to extract search_query and search_results.
    2. Evaluate the intent and requirements of search_query.
    3. Compare each result's title and result_snippet against the query intent.
    4. Identify the single result that best satisfies the query.
    5. Output only the zero-based index (0 to result_count-1) of that result.

    Constraints:
    - Respond with exactly one integer between 0 and result_count-1.
    - No additional text, explanations, or formatting.
    - The index must correspond to the position of the chosen result in search_results.
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
    tools=[page_scrap_tool],
)


def _parse_best_result_index(response_text: str, result_count: int) -> int:
    if result_count <= 0:
        return 0
    candidates = [int(match) for match in re.findall(r"-?\d+", response_text)]
    for value in candidates:
        if 0 <= value < result_count:
            return value
    for value in candidates:
        if 1 <= value <= result_count:
            return value - 1
    return 0


async def search_web(query: str) -> str:
    try:
        search_results = duckduckgo_search(query)  # Perform search
        if not search_results:
            return "No results found."
        results_payload = {
            "search_query": query,
            "result_count": len(search_results),
            "search_results": [
                {
                    "id": result.id,
                    "url": result.url,
                    "title": result.title,
                    "result_snippet": result.result_snippet,
                }
                for result in search_results
            ],
        }
        results_str: str = json.dumps(results_payload, ensure_ascii=True)
        best_result_raw: str = await best_result_chooser.ask(results_str)  # Get best result
        chosen_index: int = _parse_best_result_index(best_result_raw, len(search_results))
        choosen_link: str = search_results[chosen_index].url  # Extract chosen URL
        summary: str = await summary_assistant.ask(choosen_link)  # Get summary
        return summary
    except Exception as e:
        logging.error(f"An error occurred during web search: {e}")  # Log the error
        return "An error occurred while processing your request."
