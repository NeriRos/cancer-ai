from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


def get_state_of_union():
    with open("./data.txt") as f:
        state_of_the_union = f.read()
    return state_of_the_union


def prepare_docs(text):
    texts = get_split_texts(text)

    return [Document(page_content=t, metadata={"source": f"{i}-pl"}) for i, t in enumerate(texts)]


def get_split_texts(text):
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

    return splitter.split_text(text)


def get_docsearch(texts, embeddings):
    return Chroma.from_texts(texts, embeddings)


def get_vector_db(docs, embeddings):
    return Chroma.from_documents(docs, embeddings)
