from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine

from ai_gateway.config import APP_CONFIG
from ai_gateway.utils.llm_provider import build_llama_index_embed_model, build_llama_index_llm

pg_db = SQLDatabase.from_uri((f"postgresql://{APP_CONFIG.POSTGRES_USER}:{APP_CONFIG.POSTGRES_PASSWORD}@{APP_CONFIG.POSTGRES_HOST}:{APP_CONFIG.POSTGRES_PORT}/{APP_CONFIG.POSTGRES_DB}"))

llm = build_llama_index_llm(APP_CONFIG)
embed_model = build_llama_index_embed_model(APP_CONFIG)

engine = NLSQLTableQueryEngine(
    sql_database=pg_db,
    tables=["sales"],
    llm=llm,
    embed_model=embed_model,
    markdown_response=True,
    verbose=True,
)


async def query_pgdb(query_str: str):
    return await engine.aquery(query_str)
