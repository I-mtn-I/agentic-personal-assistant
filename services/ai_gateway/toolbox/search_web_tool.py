from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec


def search_web(query):
    tool_spec = DuckDuckGoSearchToolSpec()
    result = tool_spec.duckduckgo_full_search(query, max_results=50)  # pyright: ignore
    return result
