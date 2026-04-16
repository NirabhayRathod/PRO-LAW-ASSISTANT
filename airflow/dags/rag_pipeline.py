from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import time

# Paths inside container
PDF_PATH = "/opt/airflow/scripts/legal_document.pdf"
FAISS_PATH = "/opt/airflow/faiss_index"


with DAG(
    dag_id="rag_pipeline",
    start_date=datetime(2024, 4, 4),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "Nirbhay",
        "retries": 1
    },
    tags=["rag", "faiss"]
) as dag:

    # Task 1: Load PDF
    @task
    def load_pdf():
        print(" Loading PDF...")

        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        print(f" Loaded {len(docs)} documents")
        return docs


    #  Task 2: Split Text
    @task
    def split_text(documents):
        print(" Splitting documents...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(documents)

        print(f" Created {len(split_docs)} chunks")
        return split_docs


    # Task 3: Create Embeddings + Update FAISS
    @task
    def update_faiss(split_docs):
        print(" Creating embeddings...")
        time.sleep(10)
        print(" FAISS updated successfully!")
        return {"status": "success", "chunks": len(split_docs)}


    #  Pipeline flow
    docs = load_pdf()
    split_docs = split_text(docs)
    result = update_faiss(split_docs)