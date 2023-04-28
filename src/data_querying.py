from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain, RetrievalQAWithSourcesChain


def test():
    return "Hello World"

def summarize_text(llm, text_to_summarize):
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    summary_result = summarize_document_chain.run(text_to_summarize)

    return summary_result


def qa_text(llm, text_to_qa, question):
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    qa_result = qa_document_chain.run(input_document=text_to_qa, question=question)

    return qa_result


def retrieve_qa(llm, docsearch_to_retrieve_from, question):
    retriever = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff",
                                                            retriever=docsearch_to_retrieve_from.as_retriever())
    result = retriever({"question": question}, return_only_outputs=True)
    return result


def generate_insights(llm, vector_db, topic):
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

    return result
