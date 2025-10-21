import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Get HuggingFace token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")

os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Initialize the LLM
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        do_sample=True
    )
    
    return HuggingFacePipeline(pipeline=pipe)

CUSTOM_PROMPT_TEMPLATE = """
Use only the provided context to answer the user's question.
If the answer is not found within the context, respond with 'NA' and nothing else.
Be concise and direct.

Context: {context}

Question: {question}

Answer:"""

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
    llm=load_llm(),
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
