import streamlit as st
from langchain_mistralai import ChatMistralAI

def get_llm():
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=st.secrets["MISTRAL_API_KEY"]
    )

    return llm


def generate_answer(question, retriever):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful PDF assistant.

Answer only from the provided context.

Context:
{context}

Question:
{question}
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content
