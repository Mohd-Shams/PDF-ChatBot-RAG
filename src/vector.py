from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, embeddings):

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectordb