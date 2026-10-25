import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_qdrant import QdrantVectorStore

load_dotenv()

from src.config.llm import EMBEDDING

VECTOR_DB = QdrantVectorStore.from_existing_collection(
    embedding=EMBEDDING,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name=os.environ["QDRANT_COLLECTION_NAME"],
)

SQL_DB = SQLDatabase.from_uri(f"sqlite:///{os.environ['DB_SQLITE_PATH']}")
