from config.settings import APP_CONFIG
from llama_index.core import SQLDatabase
from domain.tool_factory import create_lc_tool
from llama_index.core.query_engine import NLSQLTableQueryEngine


pg_db = SQLDatabase.from_uri(
    f"postgresql://{APP_CONFIG.PG_USER}:{APP_CONFIG.PG_PASSWORD}@{APP_CONFIG.PG_HOST}:{APP_CONFIG.PG_PORT}/{APP_CONFIG.PG_DB}"
)
engine = NLSQLTableQueryEngine(
    sql_database=pg_db,
    tables=["sales"],
    llm=APP_CONFIG.LLM_MODEL,
    embed_model=APP_CONFIG.LLM_EMBED_MODEL,
    markdown_response=True,
    verbose=True,
)


async def get_data(query_str: str):
    return await engine.aquery(query_str)


pg_query_tool = create_lc_tool(
    callable=get_data,
    description="Query the sales database for analytics questions",
)
