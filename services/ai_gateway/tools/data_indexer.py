import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ai_gateway.domain import Tool

# Load documents from folder
# TODO: hardcoded for now. We can get attachments from the UI in the future
documents = SimpleDirectoryReader("./data/").load_data()

# Initialize Qdrant client and vector store
# TODO: add env variable for these to read from APP_CONFIG
client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="my_documents")

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the index and store vectors in Qdrant
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

# Create a query engine from the index - llamaindex FTW!
query_engine = index.as_query_engine()  # pyright: ignore


async def query_vector_db(query: str):
    return await query_engine.aquery(query)


vector_query_tool = Tool(
    target=query_vector_db,
    description="RAG tool for querying vector DB. Pass the question as an argument to this tool.",
).create_tool()
