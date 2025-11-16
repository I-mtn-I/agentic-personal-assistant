from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from domain.tool_factory import create_lc_tool

tool_spec = DuckDuckGoSearchToolSpec()

duckduck_search_tool = create_lc_tool(
    callable=tool_spec.to_tool_list()[0],
    description="DuckDuckGo search tool for web queries",
)
