import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is missing. Please set it in your .env file.")
    st.stop()

# FAISS database path
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads FAISS vector store."""
    if not os.path.exists(DB_FAISS_PATH):
        st.error("FAISS database not found. Run embedding storage script first.")
        st.stop()
    
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    """Strict prompt to prevent irrelevant answers."""
    return PromptTemplate(
        template="""
        Use only the provided context to answer the user's question.
        If the answer is not found within the context, respond with 'NA' and nothing else.  
        Do not provide unrelated information.

        Context: {context}  
        Question: {question}  

        Answer:
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    """Loads HuggingFace LLM."""
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        token=HF_TOKEN,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )

def format_source_docs(source_documents):
    """Formats source documents for display."""
    formatted_docs = ""
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'Unknown Page')
        content = doc.page_content.replace("\n", " ").strip()
        formatted_docs += f"**Source:** {source} (Page {page})\n**Content:** {content}\n\n"
    return formatted_docs.strip()

def main():
    st.title("Ask Medihelp!")

    # Session state for chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your question")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Load vector store and LLM
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            # Get response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"].strip()
            source_docs = format_source_docs(response["source_documents"])

            # Handle strict responses
            if result == "NA":
                result_to_show = "I don't have an answer based on the provided context."
            else:
                result_to_show = f"{result}\n\n**Source Documents:**\n{source_docs}"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
