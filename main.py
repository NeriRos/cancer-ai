from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from src.prepare_data import get_state_of_union, get_vector_db, get_content_chunks_documents


def main():
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    text = get_state_of_union()
    content_chunks_documents = get_content_chunks_documents()

    vector_db = get_vector_db(content_chunks_documents, embeddings)

    # generate_insights(llm, vector_db, text)
    return llm, vector_db


if __name__ == '__main__':
    main()
