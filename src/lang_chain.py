from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from src.prepare_data import get_vector_db, get_content_chunks_documents


def init_vector_db():
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    content_chunks_documents = get_content_chunks_documents()

    vector_db = get_vector_db(content_chunks_documents, embeddings)

    return llm, vector_db
