import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# Initialize the LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        token=HF_TOKEN,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )
    return llm
# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, revert back with JUST 'NA', nothing else. Don't try to make up an answer. 
Don't provide anything out of the given context. 

Context: {context}
Question: {question}
Start the answer directly. No small talk, please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt
# Load FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}. Please create it first.")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)
# Invoke the chain with a query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
