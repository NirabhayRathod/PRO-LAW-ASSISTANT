from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#  Load env
load_dotenv()

#  Config
FAISS_PATH = os.getenv("FAISS_PATH", "../src/faiss_index")
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found")

#  Embedding
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

#  LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

#  Prompt
prompt = ChatPromptTemplate.from_template(
    """You are a knowledgeable legal assistant specializing in Indian laws.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{input}

Give a clear, professional answer:"""
)

#  DB
db = None


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

    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "RAG API running"}


@app.post("/query")
def query(data: Query):
    global db

    try:
        retriever = db.as_retriever(search_kwargs={"k": 3})

        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        response = retrieval_chain.invoke({"input": data.text})

        return {
            "query": data.text,
            "answer": response["answer"]
        }

    except Exception as e:
        return {"error": str(e)}