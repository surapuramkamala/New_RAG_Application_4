from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel

from rag_engine import EnterpriseRAG


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description=(
        "Production-oriented RAG system with "
        "FAISS, Hybrid Search, Reranking, "
        "Query Transformation and Gemini."
    ),
    version="1.0.0"
)


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

rag = EnterpriseRAG()


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(
    BaseModel
):

    question: str

    stage: str = "basic"

    file_type: Optional[str] = None

    source: Optional[str] = None


# ============================================================
# RESPONSE
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Enterprise Knowledge Assistant API is running.",

        "embedding_provider":
            "Ollama",

        "vector_database":
            "FAISS",

        "answer_model":
            "Gemini",

        "available_stages": [
            "basic",
            "hybrid",
            "reranking",
            "query_transformation"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "chunks_loaded":
            len(rag.chunks)
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    try:

        response = rag.answer_question(

            question=request.question,

            stage=request.stage,

            file_type=request.file_type,

            source=request.source
        )

        return response


    except Exception as error:

        error_message = str(
            error
        )

        print("\n" + "=" * 60)
        print("API ERROR")
        print("=" * 60)
        print(error_message)
        print("=" * 60 + "\n")


        # Ollama connection/model error
        if (
            "Ollama embedding failed"
            in error_message
            or
            "Ollama batch embedding failed"
            in error_message
        ):

            raise HTTPException(

                status_code=503,

                detail=(
                    "Ollama embedding service is "
                    "not available. Make sure Ollama "
                    "is running and nomic-embed-text "
                    "is installed."
                )
            )


        # Gemini generation quota error
        if (
            "429"
            in error_message
            or
            "RESOURCE_EXHAUSTED"
            in error_message
        ):

            raise HTTPException(

                status_code=429,

                detail=(
                    "Gemini generation API quota is "
                    "temporarily exhausted. "
                    "Please wait and try again."
                )
            )


        raise HTTPException(

            status_code=500,

            detail=error_message
        )


# ============================================================
# VECTOR STORE INFORMATION
# ============================================================

@app.get("/vector-store")
def vector_store_info():

    return {

        "total_chunks":
            len(rag.chunks),

        "vector_dimension":
            rag.index.d,

        "total_vectors":
            rag.index.ntotal
    }
    