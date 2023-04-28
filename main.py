from sys import argv

from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from data_querying import generate_insights
from prepare_data import get_state_of_union, get_vector_db, get_content_chunks_documents

if __name__ == '__main__':
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    text = get_state_of_union()
    content_chunks_documents = get_content_chunks_documents()

    vector_db = get_vector_db(content_chunks_documents, embeddings)

    if argv[1] == "insights":
        question = argv[2]

        result = generate_insights(llm, vector_db, question)

        print(result)
