import os

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()

from config.llm import EMBEDDING

VECTOR_DB = QdrantVectorStore.from_existing_collection(
    embedding=EMBEDDING,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name=os.environ["QDRANT_COLLECTION_NAME"],
)
