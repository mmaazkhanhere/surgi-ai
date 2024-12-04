import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

@tool
def query_database(surgeon_query: str):
  """Query the pinecone database for the surgeon query"""
  print("Query database")
  pinecone_api_key = os.getenv('PINECONE_API_KEY')
  pc = Pinecone(api_key=pinecone_api_key)
  index_name = "surgical-assistant"
  index = pc.Index(index_name)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
  vector_store = PineconeVectorStore(index=index, embedding=embeddings)
  retriever = vector_store.as_retriever()
  retrieved_information = retriever.invoke(surgeon_query)
  return retrieved_information
