import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def clean_text(text):
    """Removes extra spaces and newlines from text."""
    return text.replace("\n", " ").replace("  ", " ").strip()

def format_source_docs(source_documents):
    """Formats source documents for display."""
    formatted_docs = ""
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'Unknown Page')
        content = clean_text(doc.page_content)
        formatted_docs += f"**Source:** {source} (Page {page})\n**Content:** {content}\n\n"
    return formatted_docs

def main():
    st.title("Ask Medihelp!")

    # Initialize messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Answer the user's question **only** using the provided context.  
        - If the answer is not found in the context, **reply only with 'NA'**—no explanations, assumptions, or reasoning.  
        - Do **not** include citations, sources, or references in your response.  

        Context: {context}  
        Question: {question}  
        Provide a direct answer:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Prepare the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Run the query
            response = qa_chain.invoke({"query": prompt})
            result = response["result"].strip()

            # If the answer is "NA", show only "NA" with no citations
            if result == "NA":
                result_to_show = "NA"
            else:
                source_documents = response["source_documents"]
                result_to_show = clean_text(result)
                source_docs = format_source_docs(source_documents)
                result_to_show = f"{result_to_show}\n\n**Source Documents**:\n{source_docs}"

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
