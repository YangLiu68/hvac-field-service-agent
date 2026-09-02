import argparse
import json
from pathlib import Path

from app.services.rag_service import hybrid_search


def evaluate(cases_path: Path, top_k: int) -> dict:
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    hits = 0
    reciprocal_rank_sum = 0.0
    details = []

    for case in cases:
        results = hybrid_search(case["query"], top_k=top_k)
        documents = [result["document"] for result in results]
        expected_documents = case.get("expected_documents", [case.get("expected_document")])
        rank = next(
            (position for position, document in enumerate(documents, start=1) if document in expected_documents),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank
        details.append(
            {
                "query": case["query"],
                "expected_documents": expected_documents,
                "rank": rank,
                "retrieved_documents": documents,
            }
        )

    total = len(cases)
    return {
        "cases": total,
        f"recall_at_{top_k}": hits / total if total else 0.0,
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evaluation/retrieval_cases.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases, args.top_k), indent=2))
