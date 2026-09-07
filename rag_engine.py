import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from google import genai
from ollama import Client

from pypdf import PdfReader
from docx import Document


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"


# ============================================================
# GEMINI CONFIGURATION
# GEMINI IS USED ONLY FOR:
# 1. QUERY REWRITING
# 2. FINAL ANSWER GENERATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# OLLAMA CONFIGURATION
# OLLAMA IS USED ONLY FOR EMBEDDINGS
# ============================================================

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)


print("\n" + "=" * 60)
print("RAG CONFIGURATION")
print("=" * 60)
print("Embedding Provider: OLLAMA")
print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Embedding Model: {EMBEDDING_MODEL}")
print(f"Gemini Model: {GEMINI_MODEL}")
print("=" * 60 + "\n")


# ============================================================
# CLIENTS
# ============================================================

ollama_client = Client(
    host=OLLAMA_HOST
)


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing in the .env file."
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# DOCUMENT PARSING
# ============================================================

def parse_pdf(
    file_path: Path
) -> List[Dict]:

    documents = []

    reader = PdfReader(
        str(file_path)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "section": f"Page {page_number}",
                        "page": page_number
                    }
                }
            )

    return documents


def parse_txt(
    file_path: Path
) -> List[Dict]:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": file_path.suffix.lower(),
                "section": "Full Document"
            }
        }
    ]


def parse_docx(
    file_path: Path
) -> List[Dict]:

    document = Document(
        str(file_path)
    )

    parts = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            parts.append(text)

    # Read tables
    for table in document.tables:

        for row in table.rows:

            row_values = [
                cell.text.strip()
                for cell in row.cells
            ]

            parts.append(
                " | ".join(row_values)
            )

    text = "\n".join(parts)

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": file_path.suffix.lower(),
                "section": "Full Document"
            }
        }
    ]


def parse_csv(
    file_path: Path
) -> List[Dict]:

    dataframe = pd.read_csv(
        file_path
    )

    text = dataframe.to_csv(
        index=False
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "file_type": file_path.suffix.lower(),
                "section": "CSV Data"
            }
        }
    ]


def parse_excel(
    file_path: Path
) -> List[Dict]:

    excel_file = pd.ExcelFile(
        file_path
    )

    documents = []

    for sheet_name in excel_file.sheet_names:

        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

        text = dataframe.to_csv(
            index=False
        )

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "section": f"Sheet: {sheet_name}"
                }
            }
        )

    return documents


# ============================================================
# LOAD DOCUMENT
# ============================================================

def load_document(
    file_path: Path
) -> List[Dict]:

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        return parse_pdf(file_path)

    elif suffix == ".txt":

        return parse_txt(file_path)

    elif suffix == ".docx":

        return parse_docx(file_path)

    elif suffix == ".csv":

        return parse_csv(file_path)

    elif suffix in [".xlsx", ".xls"]:

        return parse_excel(file_path)

    else:

        print(
            f"Skipping unsupported file: "
            f"{file_path.name}"
        )

        return []


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk
            )

        start += (
            chunk_size - overlap
        )

    return chunks


def create_chunks(
    documents: List[Dict]
) -> List[Dict]:

    chunks = []

    chunk_id = 0

    for document in documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for chunk_number, text in enumerate(
            text_chunks,
            start=1
        ):

            metadata = document[
                "metadata"
            ].copy()

            metadata[
                "chunk_number"
            ] = chunk_number

            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata
                }
            )

            chunk_id += 1

    return chunks


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def create_embedding(
    text: str
) -> List[float]:
    """
    Create one embedding using Ollama.
    """

    try:

        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=text
        )

        embeddings = response[
            "embeddings"
        ]

        if not embeddings: raise RuntimeError( "Ollama returned empty embeddings." )

        return embeddings[0]

    except Exception as error:

        raise RuntimeError(
            "\nOllama embedding failed.\n"
            "Check that:\n"
            "1. Ollama is running\n"
            "2. nomic-embed-text is installed\n"
            f"3. OLLAMA_HOST is correct\n\n"
            f"Original error: {error}"
        )


def create_embeddings(
    chunks: List[Dict]
) -> np.ndarray:
    """
    Batch embedding generation using Ollama.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating Ollama embeddings for "
        f"{len(texts)} chunks..."
    )

    try:

        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=texts
        )

        embeddings = response[
            "embeddings"
        ]

        if not embeddings: raise RuntimeError( "Ollama returned empty embeddings." )

        embeddings_array = np.array(
            embeddings,
            dtype="float32"
        )

        print( f"\nSUCCESS: Created " f"{len(embeddings_array)} embeddings." )

        return embeddings_array

    except Exception as error:

        print( "\nOLLAMA EMBEDDING ERROR:" ) 
        print(error)
        raise RuntimeError( f"Ollama batch embedding failed: " f"{error}" )


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store():

    VECTOR_STORE_DIR.mkdir(
        exist_ok=True
    )

    all_documents = []

    supported_extensions = [
        ".pdf",
        ".txt",
        ".docx",
        ".csv",
        ".xlsx",
        ".xls"
    ]

    files = [

        file

        for file in DATA_DIR.iterdir()

        if file.is_file()

        and file.suffix.lower()
        in supported_extensions
    ]

    print(
        f"\nDocuments found: "
        f"{len(files)}"
    )

    for file_path in files:

        print(
            f"\nProcessing: "
            f"{file_path.name}"
        )

        documents = load_document(
            file_path
        )

        all_documents.extend(
            documents
        )

    if not all_documents:

        raise RuntimeError(
            "No documents were loaded."
        )

    chunks = create_chunks(
        all_documents
    )

    print(
        f"\nTotal chunks created: "
        f"{len(chunks)}"
    )

    embeddings = create_embeddings(
        chunks
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding Dimension: {dimension}")
    print(f"Index: {FAISS_INDEX_PATH}")
    print("=" * 60)

    return index, chunks


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    if (
        not FAISS_INDEX_PATH.exists()
        or
        not CHUNKS_PATH.exists()
    ):

        print(
            "Vector store not found."
        )

        return build_vector_store()

    print(
        "Loading existing FAISS vector store..."
    )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
        )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    return index, chunks


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int = 10
) -> List[Dict]:

    query_embedding = np.array(
        [
            create_embedding(
                query
            )
        ],
        dtype="float32"
    )

    # Normalize query embedding
    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        min(
            top_k,
            len(chunks)
        )
    )

    results = []

    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        if chunk_index == -1:

            continue

        chunk = chunks[
            int(chunk_index)
        ]

        results.append(
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "vector_score": float(score)
            }
        )

    return results


# ============================================================
# BM25-LIKE KEYWORD SEARCH
# SIMPLE LEXICAL RETRIEVAL
# ============================================================

def tokenize(
    text: str
) -> set:

    return set(
        re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            text.lower()
        )
    )


def keyword_search(
    query: str,
    chunks: List[Dict],
    top_k: int = 10
) -> List[Dict]:

    query_tokens = tokenize(
        query
    )

    scored_results = []

    for chunk in chunks:

        chunk_tokens = tokenize(
            chunk["text"]
        )

        common_tokens = (
            query_tokens
            &
            chunk_tokens
        )

        score = len(
            common_tokens
        )

        if score > 0:

            scored_results.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "keyword_score": float(score)
                }
            )

    scored_results.sort(
        key=lambda item:
            item["keyword_score"],
        reverse=True
    )

    return scored_results[
        :top_k
    ]


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int = 10
) -> List[Dict]:

    vector_results = vector_search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    keyword_results = keyword_search(
        query=query,
        chunks=chunks,
        top_k=top_k
    )

    combined = {}

    for result in vector_results:

        key = (
            result["metadata"]["source"],
            result["metadata"].get(
                "chunk_number",
                0
            )
        )

        combined[key] = result.copy()

        combined[key][
            "hybrid_score"
        ] = (
            result["vector_score"]
        )

    for result in keyword_results:

        key = (
            result["metadata"]["source"],
            result["metadata"].get(
                "chunk_number",
                0
            )
        )

        keyword_normalized = min(
            result["keyword_score"] / 10,
            1.0
        )

        if key not in combined:

            combined[key] = result.copy()

            combined[key][
                "hybrid_score"
            ] = (
                0.4
                *
                keyword_normalized
            )

        else:

            combined[key][
                "hybrid_score"
            ] += (
                0.4
                *
                keyword_normalized
            )

    results = list(
        combined.values()
    )

    results.sort(
        key=lambda item:
            item.get(
                "hybrid_score",
                0
            ),
        reverse=True
    )

    return results[
        :top_k
    ]


# ============================================================
# SIMPLE RERANKING
# ============================================================

def rerank_results(
    query: str,
    results: List[Dict],
    top_k: int = 5
) -> List[Dict]:

    query_tokens = tokenize(
        query
    )

    reranked = []

    for result in results:

        text_tokens = tokenize(
            result["text"]
        )

        overlap = len(
            query_tokens
            &
            text_tokens
        )

        base_score = result.get(
            "hybrid_score",
            result.get(
                "vector_score",
                0
            )
        )

        rerank_score = (
            0.7 * base_score
            +
            0.3 * (
                overlap
                /
                max(
                    len(query_tokens),
                    1
                )
            )
        )

        result_copy = result.copy()

        result_copy[
            "rerank_score"
        ] = float(
            rerank_score
        )

        reranked.append(
            result_copy
        )

    reranked.sort(
        key=lambda item:
            item["rerank_score"],
        reverse=True
    )

    return reranked[
        :top_k
    ]


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(
    query: str
) -> str:

    prompt = f"""
Rewrite the following user question into a short
search query for retrieving enterprise documents.

Keep important technical terms, roles, systems,
and policies.

Return only the rewritten search query.

User question:
{query}
"""

    try:

        response = (
            gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )

        rewritten = response.text.strip()

        return rewritten or query

    except Exception as error:

        print(
            f"Query rewriting failed: "
            f"{error}"
        )

        return query


# ============================================================
# METADATA FILTERING
# ============================================================

def filter_by_metadata(
    results: List[Dict],
    file_type: Optional[str] = None,
    source: Optional[str] = None
) -> List[Dict]:

    filtered = results

    if file_type:

        filtered = [

            result

            for result
            in filtered

            if result["metadata"]
            .get(
                "file_type",
                ""
            )
            .lower()
            ==
            file_type.lower()
        ]

    if source:

        filtered = [

            result

            for result
            in filtered

            if result["metadata"]
            .get(
                "source",
                ""
            )
            .lower()
            ==
            source.lower()
        ]

    return filtered


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    results: List[Dict]
) -> float:

    if not results:

        return 0.0

    best_result = results[0]

    score = best_result.get(
        "rerank_score",
        best_result.get(
            "hybrid_score",
            best_result.get(
                "vector_score",
                0
            )
        )
    )

    score = max(
        0,
        min(
            float(score),
            1
        )
    )

    return round(
        score,
        2
    )


# ============================================================
# ENTERPRISE RAG CLASS
# ============================================================

class EnterpriseRAG:

    def __init__(self):

        print(
            "\nInitializing Enterprise RAG..."
        )

        self.index, self.chunks = (
            load_vector_store()
        )

        self.history = []

        print(
            "Enterprise RAG ready."
        )


    # ========================================================
    # RETRIEVAL PIPELINE
    # ========================================================

    def retrieve(
        self,
        query: str,
        stage: str = "basic",
        file_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> Tuple[List[Dict], str]:

        search_query = query

        # Query Transformation
        if stage == "query_transformation":

            search_query = rewrite_query(
                query
            )

            results = hybrid_search(
                query=search_query,
                index=self.index,
                chunks=self.chunks,
                top_k=10
            )

            results = rerank_results(
                query=search_query,
                results=results,
                top_k=5
            )

        # Basic Vector Search
        elif stage == "basic":

            results = vector_search(
                query=search_query,
                index=self.index,
                chunks=self.chunks,
                top_k=5
            )

        # Hybrid Search
        elif stage == "hybrid":

            results = hybrid_search(
                query=search_query,
                index=self.index,
                chunks=self.chunks,
                top_k=5
            )

        # Reranking
        elif stage == "reranking":

            results = hybrid_search(
                query=search_query,
                index=self.index,
                chunks=self.chunks,
                top_k=10
            )

            results = rerank_results(
                query=search_query,
                results=results,
                top_k=5
            )

        else:

            raise ValueError(
                "Invalid stage. Use: "
                "basic, hybrid, reranking, "
                "or query_transformation."
            )

        # Metadata filtering
        filtered_results = filter_by_metadata(
            results=results,
            file_type=file_type,
            source=source
        )

        # Fallback to original results if filter
        # accidentally removes everything
        if (
            not filtered_results
            and results
            and (
                file_type
                or source
            )
        ):

            print(
                "Metadata filtering returned zero "
                "results. Using original results."
            )

            filtered_results = results

        results = filtered_results

        # Debug output
        print("\n" + "=" * 60)
        print("RETRIEVAL DEBUG")
        print("=" * 60)
        print(
            f"Original Query: {query}"
        )
        print(
            f"Search Query: {search_query}"
        )
        print(
            f"Stage: {stage}"
        )
        print(
            f"Results Found: {len(results)}"
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {number}"
            )

            print(
                "Document:",
                result["metadata"].get(
                    "source"
                )
            )

            print(
                "Section:",
                result["metadata"].get(
                    "section"
                )
            )

            print(
                "Score:",
                result.get(
                    "rerank_score",
                    result.get(
                        "hybrid_score",
                        result.get(
                            "vector_score",
                            0
                        )
                    )
                )
            )

            print(
                "Text:",
                result["text"][:250]
            )

        print("=" * 60 + "\n")

        return (
            results,
            search_query
        )


    # ========================================================
    # ANSWER QUESTION
    # ========================================================

    def answer_question(
        self,
        question: str,
        stage: str = "basic",
        file_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict:

        results, rewritten_query = (
            self.retrieve(
                query=question,
                stage=stage,
                file_type=file_type,
                source=source
            )
        )

        if not results:

            return {
                "answer": (
                    "I don't know. I could not "
                    "find relevant information "
                    "in the available documents."
                ),
                "confidence": 0,
                "sources": [],
                "rewritten_query":
                    rewritten_query
            }

        context_parts = []

        sources = []

        for number, result in enumerate(
            results,
            start=1
        ):

            metadata = result[
                "metadata"
            ]

            context_parts.append(
                f"""
SOURCE {number}
Document: {metadata.get("source")}
Section: {metadata.get("section")}

Content:
{result["text"]}
"""
            )

            sources.append(
                {
                    "document":
                        metadata.get(
                            "source"
                        ),
                    "section":
                        metadata.get(
                            "section"
                        ),
                    "page":
                        metadata.get(
                            "page"
                        ),
                    "score":
                        result.get(
                            "rerank_score",
                            result.get(
                                "hybrid_score",
                                result.get(
                                    "vector_score",
                                    0
                                )
                            )
                        )
                }
            )

        context = "\n\n".join(
            context_parts
        )

        history_text = ""

        for item in self.history[-4:]:

            history_text += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n"
            )

        prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   say exactly:
   "I don't know based on the available documents."
3. Provide a clear and concise answer.
4. Mention supporting document names.
5. Do not use external knowledge.

Conversation History:
{history_text}

User Question:
{question}

Retrieved Context:
{context}
"""

        try:

            response = (
                gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
            )

            answer = response.text.strip()

        except Exception as error:

            raise RuntimeError(
                f"Gemini answer generation failed: "
                f"{error}"
            )

        confidence = calculate_confidence(
            results
        )

        self.history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "rewritten_query":
                rewritten_query
        }


# ============================================================
# BUILD VECTOR STORE DIRECTLY
# ============================================================

if __name__ == "__main__":

    print(
        "Starting vector store creation..."
    )

    build_vector_store()
    


    