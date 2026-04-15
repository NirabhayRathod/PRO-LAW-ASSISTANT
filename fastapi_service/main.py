from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

FAISS_PATH = "../airflow/faiss_index"

#  Load embedding
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

#  Global DB
db = None


# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db

    print("Loading FAISS...")

    if not os.path.exists(FAISS_PATH):
        raise Exception(f"FAISS not found at {FAISS_PATH}")

    db = FAISS.load_local(
        FAISS_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    print("FAISS loaded!")

    yield  # app runs here

    print(" Shutting down...")


#  Create app with lifespan
app = FastAPI(lifespan=lifespan)


# Request schema
class Query(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "RAG API running "}


@app.post("/query")
def query(data: Query):
    global db

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(data.text)

    results = [doc.page_content for doc in docs]

    return {
        "query": data.text,
        "results": results
    }