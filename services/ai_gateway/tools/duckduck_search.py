from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from ai_gateway.domain import Tool

tool_spec = DuckDuckGoSearchToolSpec()

duckduck_search_tool = Tool(
    target=tool_spec.to_tool_list()[0],
    description="DuckDuckGo search tool for web queries",
).create_tool()
