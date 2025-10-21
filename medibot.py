import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Get HuggingFace token from environment or Streamlit secrets
try:
    # Try to get from Streamlit secrets first (for deployment)
    HF_TOKEN = st.secrets["HUGGINGFACE_HUB_TOKEN"]
except:
    # Fallback to environment variable (for local development)
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not HF_TOKEN:
        st.error("HuggingFace token not found. Please set HUGGINGFACE_HUB_TOKEN in secrets or environment variables.")
        st.stop()

os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'token': HF_TOKEN}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=False)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

@st.cache_resource
def get_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)

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

        # Build conversation history for context
        conversation_history = ""
        if len(st.session_state.messages) > 1:
            recent_messages = st.session_state.messages[-6:-1]  # Last 3 Q&A pairs
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_history += f"{role}: {msg['content'][:200]}\n"

        CUSTOM_PROMPT_TEMPLATE = f"""
Use only the provided context to answer the user's question.
If the answer is not found within the context, respond with 'NA' and nothing else.
Be concise and direct.

Previous conversation:
{conversation_history}

Context: {{context}}

Question: {{question}}

Answer:"""

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Prepare the QA chain with improved retrieval
            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "fetch_k": 10}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Run the query with streaming
            with st.chat_message("assistant"):
                with st.spinner("Searching medical encyclopedia..."):
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

                st.markdown(result_to_show)

            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except FileNotFoundError as e:
            st.error(f"Vector store not found: {str(e)}")
        except ValueError as e:
            st.error(f"Configuration error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
