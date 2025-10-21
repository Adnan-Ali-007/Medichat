#store emebedding in faiss
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#load raw pdf
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)   
    documents=loader.load()
    return documents #pdf loaded here processed returned as documents
documents=load_pdf_files(data=DATA_PATH)
print("lnght of doc pages:",len(documents))
#Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks=create_chunks(extracted_data=documents)
#Create vector emedding
def get_embedding_model():
   # Get HuggingFace token from environment variable
   HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
   if not HF_TOKEN:
       raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
   
   embedding_model=HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={'token': HF_TOKEN}
   )
   return embedding_model
embedding_model=get_embedding_model()
#store in faiss locally
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
