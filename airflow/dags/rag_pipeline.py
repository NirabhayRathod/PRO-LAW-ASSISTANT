from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# MAIN FUNCTION (DOES EVERYTHING)
def build_faiss_pipeline():
    print(" Starting RAG pipeline...")

    PDF_PATH = "/opt/airflow/scripts/legal_document.pdf"
    FAISS_PATH = "/opt/airflow/faiss_index"

    # Step 1: Load PDF
    print(" Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Step 2: Split
    print(" Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    # Step 3: Embeddings
    print(" Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    # Step 4: FAISS (Incremental logic)
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS...")
        db = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Adding new documents...")
        db.add_documents(split_docs)

    else:
        print("Creating new FAISS index...")
        db = FAISS.from_documents(split_docs, embeddings)

    # Step 5: Save FAISS
    db.save_local(FAISS_PATH)

    print(" FAISS updated successfully!")


# DAG Definition
with DAG(
    dag_id="rag_pipeline",
    start_date=datetime(2026,4 ,4),
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=["rag", "faiss"]
) as dag:

    run_pipeline = PythonOperator(
        task_id="build_faiss_pipeline",
        python_callable=build_faiss_pipeline
    )

    run_pipeline