from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from ai_gateway.config import APP_CONFIG

pg_db = SQLDatabase.from_uri(
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


async def query_pgdb(query_str: str):
    return await engine.aquery(query_str)
