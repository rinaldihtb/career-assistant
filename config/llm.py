import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING = OpenAIEmbeddings(model=os.environ["LLM_EMBEDDING_MODEL"])
LLM_MODEL = init_chat_model(model=os.environ["LLM_INFERENCE_MODEL"])
