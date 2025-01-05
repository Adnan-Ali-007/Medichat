#store emebedding in faiss
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
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, #size of chunk 
                                                 chunk_overlap=50)#for context overlap
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks=create_chunks(extracted_data=documents)
#Create vector emedding
def get_embedding_model():
   embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")#chuncks to numeric emdeiing numbers
   return embedding_model
embedding_model=get_embedding_model()
#store in faiss locally
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
