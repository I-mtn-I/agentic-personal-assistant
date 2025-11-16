import qdrant_client
from config.settings import APP_CONFIG
from domain.tool_factory import create_lc_tool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

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

# Create a query engine from the index - llamindex FTW!
query_engine = index.as_query_engine()

# Query the index
response = query_engine.query("What did the author do growing up?")


async def query_vector_db(query: str):
    result = await query_engine.query(query)
    return result


vector_query_tool = create_lc_tool(
    callable=query_vector_db,
    description="RAG tool for querying vector DB. Pass the question as an argument to this tool.",
)
