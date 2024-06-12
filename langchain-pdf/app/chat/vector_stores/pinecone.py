import os
from langchain_community.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings
 
vector_store = Pinecone.from_existing_index(
  os.environ.get("PINECONE_INDEX_NAME"), embeddings
)

def build_retriever(chat_args):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id }}
    return vector_store.as_retriever(search_kwargs=search_kwargs)