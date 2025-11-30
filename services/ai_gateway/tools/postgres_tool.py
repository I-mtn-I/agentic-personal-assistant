from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from ai_gateway.config import APP_CONFIG
from ai_gateway.domain import Tool

pg_db = SQLDatabase.from_uri(  # pyright: ignore
    (
        f"postgresql://{APP_CONFIG.POSTGRES_USER}:{APP_CONFIG.POSTGRES_PASSWORD}"
        f"@{APP_CONFIG.POSTGRES_HOST}:{APP_CONFIG.POSTGRES_PORT}/{APP_CONFIG.POSTGRES_DB}"
    )
)

llm = Ollama(model=APP_CONFIG.LLM_MODEL, base_url=APP_CONFIG.LLM_HOST)
embed_model = OllamaEmbedding(
    model_name=APP_CONFIG.LLM_EMBED_MODEL,
    base_url=APP_CONFIG.LLM_HOST,
)

engine = NLSQLTableQueryEngine(
    sql_database=pg_db,
    tables=["sales"],
    llm=llm,
    embed_model=embed_model,
    markdown_response=True,
    verbose=True,
)


async def get_data(query_str: str):
    return await engine.aquery(query_str)


pg_query_tool = Tool(
    target=get_data,
    description="Query the sales database for analytics questions",
).create_tool()
