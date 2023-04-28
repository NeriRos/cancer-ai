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
    # question = "Number of patients that were included between August 2016 and April 2018"
    question = "The effects of fluphenazine on gbm"

    result = generate_insights(llm, vector_db, question)

    print(result)
