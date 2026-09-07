import json
from pathlib import Path

from rag_engine import EnterpriseRAG

from pathlib import Path


BASE_DIR = Path(
    __file__
).resolve().parent


EVALUATION_DIR = (
    BASE_DIR / "evaluation"
)


EVALUATION_DATASET = (
    EVALUATION_DIR
    / "evaluation_dataset.json"
)


RESULTS_PATH = (
    EVALUATION_DIR
    / "evaluation_results.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    with open(

        EVALUATION_DATASET,

        "r",

        encoding="utf-8"

    ) as file:

        return json.load(
            file
        )


# ============================================================
# SIMPLE RETRIEVAL RELEVANCE
# ============================================================

def calculate_retrieval_relevance(

    expected_sources,

    retrieved_sources

):

    if not expected_sources:
        return 0

    retrieved_documents = [

        source["document"]

        for source
        in retrieved_sources
    ]

    matches = sum(

        1

        for expected
        in expected_sources

        if expected
        in retrieved_documents
    )

    return round(

        matches
        /
        len(expected_sources),

        2
    )


# ============================================================
# CONTEXT PRECISION
# ============================================================

def calculate_context_precision(

    expected_sources,

    retrieved_sources

):

    if not retrieved_sources:
        return 0

    relevant = sum(

        1

        for source
        in retrieved_sources

        if source["document"]
        in expected_sources
    )

    return round(

        relevant
        /
        len(retrieved_sources),

        2
    )


# ============================================================
# CONTEXT RECALL
# ============================================================

def calculate_context_recall(

    expected_sources,

    retrieved_sources

):

    if not expected_sources:
        return 0

    retrieved_documents = {

        source["document"]

        for source
        in retrieved_sources
    }

    matched_sources = sum(

        1

        for source
        in expected_sources

        if source
        in retrieved_documents
    )

    return round(

        matched_sources
        /
        len(expected_sources),

        2
    )


# ============================================================
# ANSWER RELEVANCE
# ============================================================

def calculate_answer_relevance(

    answer,

    expected_keywords

):

    if not expected_keywords:
        return 0

    answer_lower = answer.lower()

    matches = sum(

        1

        for keyword
        in expected_keywords

        if keyword.lower()
        in answer_lower
    )

    return round(

        matches
        /
        len(expected_keywords),

        2
    )


# ============================================================
# FAITHFULNESS
# ============================================================

def calculate_faithfulness(

    answer,

    retrieved_sources

):

    if not retrieved_sources:
        return 0

    source_text = " ".join(

        source[
            "text_preview"
        ].lower()

        for source
        in retrieved_sources
    )

    answer_words = set(

        answer.lower().split()
    )

    source_words = set(

        source_text.split()
    )

    overlap = len(

        answer_words
        &
        source_words
    )

    return round(

        overlap
        /
        max(
            len(answer_words),
            1
        ),

        2
    )


# ============================================================
# CITATION CORRECTNESS
# ============================================================

def calculate_citation_correctness(

    answer,

    sources

):

    if not sources:
        return 0

    citation_count = sum(

        1

        for number
        in range(
            1,
            len(sources) + 1
        )

        if f"[Source {number}]"
        in answer
    )

    return round(

        citation_count
        /
        len(sources),

        2
    )


# ============================================================
# RUN EVALUATION
# ============================================================

def evaluate_stage(
    rag,
    dataset,
    stage
):

    results = []

    total_questions = len(dataset)

    for question_number, item in enumerate(
        dataset,
        start=1
    ):

        question = item["question"]

        print(
            f"\n[{stage}] "
            f"Question {question_number}/{total_questions}"
        )

        print(
            f"Question: {question}"
        )

        expected_sources = item.get(
            "expected_sources",
            []
        )

        expected_keywords = item.get(
            "expected_keywords",
            []
        )

        try:

            response = rag.answer_question(

                question=question,

                stage=stage
            )

            retrieval_relevance = (
                calculate_retrieval_relevance(

                    expected_sources,

                    response["sources"]
                )
            )

            context_precision = (
                calculate_context_precision(

                    expected_sources,

                    response["sources"]
                )
            )

            context_recall = (
                calculate_context_recall(

                    expected_sources,

                    response["sources"]
                )
            )

            answer_relevance = (
                calculate_answer_relevance(

                    response["answer"],

                    expected_keywords
                )
            )

            faithfulness = (
                calculate_faithfulness(

                    response["answer"],

                    response["sources"]
                )
            )

            citation_correctness = (
                calculate_citation_correctness(

                    response["answer"],

                    response["sources"]
                )
            )

            result = {

                "id": item.get("id"),

                "question": question,

                "stage": stage,

                "status": "success",

                "answer": response["answer"],

                "retrieval_relevance":
                    retrieval_relevance,

                "context_precision":
                    context_precision,

                "context_recall":
                    context_recall,

                "faithfulness":
                    faithfulness,

                "answer_relevance":
                    answer_relevance,

                "citation_correctness":
                    citation_correctness
            }

            print("Status: SUCCESS")

        except Exception as error:

            print(
                f"Status: FAILED"
            )

            print(
                f"Error: {str(error)}"
            )

            result = {

                "id": item.get("id"),

                "question": question,

                "stage": stage,

                "status": "failed",

                "error": str(error),

                "answer": None,

                "retrieval_relevance": 0,

                "context_precision": 0,

                "context_recall": 0,

                "faithfulness": 0,

                "answer_relevance": 0,

                "citation_correctness": 0
            }

        results.append(result)

    return results

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Create evaluation folder if missing
    RESULTS_PATH.parent.mkdir(
        exist_ok=True
    )

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} "
        f"evaluation questions."
    )

    rag = EnterpriseRAG()

    stages = [

        "basic",

        "hybrid",

        "reranking",

        "query_transformation"
    ]

    all_results = {}

    # Create file immediately
    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {},
            file,
            indent=2
        )

    print(
        f"Created results file: "
        f"{RESULTS_PATH}"
    )

    for stage in stages:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"STARTING EVALUATION: "
            f"{stage.upper()}"
        )

        print(
            "=" * 60
        )

        results = evaluate_stage(

            rag,

            dataset,

            stage
        )

        all_results[stage] = results

        # Save immediately after every stage
        with open(
            RESULTS_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                all_results,

                file,

                indent=2,

                ensure_ascii=False
            )

        print(
            f"\nStage completed: {stage}"
        )

        print(
            f"Results saved to: "
            f"{RESULTS_PATH}"
        )

    print(
        "\nEvaluation completed successfully."
    )