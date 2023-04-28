import pathlib

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


def get_contents_of_directory(path):
    directory_path = pathlib.Path(path)
    files = list(directory_path.glob("**/*.txt"))
    for file in files:
        with open(file, "r") as f:
            yield Document(page_content=f.read(), metadata={"source": f.name})


def get_content_chunks_documents():
    sources = get_contents_of_directory('./content')
    source_chunks = []

    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    return source_chunks


def get_state_of_union(file_path="content/data.txt"):
    with open(file_path) as f:
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
