from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import AnalyzeDocumentChain, RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings

from prepare_data import get_state_of_union, prepare_docs, get_split_texts, get_docsearch, get_vector_db


def summarize_text(text_to_summarize):
    print("Summarizing text")

    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    summary_result = summarize_document_chain.run(text_to_summarize)

    print(summary_result)
    return summary_result


def qa_text(text_to_qa, question):
    print("Querying text")

    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    qa_result = qa_document_chain.run(input_document=text_to_qa, question=question)

    print(qa_result)
    return qa_result


def retrieve_qa(docsearch_to_retrieve_from, question):
    retriever = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff",
                                                            retriever=docsearch_to_retrieve_from.as_retriever())
    result = retriever({"question": question}, return_only_outputs=True)
    print(result)
    return result


def generate_insights(topic):
    prompt_template = """Use the context below to generate insights about the topic below:
        Context: {context}
        Topic: {topic}
        Insights:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "topic"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    docs = vector_db.similarity_search(topic, k=4)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    result = chain.apply(inputs)

    print(result)
    return result


if __name__ == '__main__':
    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    text = get_state_of_union()
    split_texts = get_split_texts(text)
    documents = prepare_docs(text)

    # docsearch = get_docsearch(split_texts, embeddings)
    vector_db = get_vector_db(documents, embeddings)

    # summarize_text(text)
    # qa_text(text, "What are the effects of fluphenazine on gbm?")
    # retrieve_qa(docsearch, "What are the effects of fluphenazine on gbm?")
    generate_insights("The effects of fluphenazine on gbm")
