import argparse
import json
from pathlib import Path

from smartshop_rag.model.factory import create_chat_model, create_embedding_model
from smartshop_rag.rag.rag_service import RagSummarizeService
from smartshop_rag.rag.vector_store import VectorStoreService


def load_queries(query_set_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(query_set_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="运行混合检索离线对照实验")
    parser.add_argument("--query-set", default="data/query_sets/air_fryer_midea_query_set_v1.jsonl")
    parser.add_argument("--output", default="data/query_sets/air_fryer_midea_experiment_v1.jsonl")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=[
            "vector",
            "bm25",
            "hybrid",
            "hybrid_rerank",
            "weighted_rrf",
            "weighted_rrf_rerank",
            "weighted_rrf_v2",
            "weighted_rrf_v2_rerank",
        ],
    )
    args = parser.parse_args()

    embedding_model = create_embedding_model()
    vector_store_service = VectorStoreService(embedding_function=embedding_model)
    rag_service = RagSummarizeService(
        model=create_chat_model(role="rag_chat"),
        rewrite_model=create_chat_model(role="rewrite_chat"),
        rerank_model=create_chat_model(role="rerank_chat"),
        vector_store_service=vector_store_service,
    )

    rows = load_queries(args.query_set)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            for mode in args.modes:
                query = str(row.get("query", "")).strip()
                answer = rag_service.rag_summarize(query, mode=mode)
                payload = {
                    "id": row.get("id"),
                    "query": query,
                    "category": row.get("category"),
                    "target_models": row.get("target_models", []),
                    "difficulty": row.get("difficulty"),
                    "mode": mode,
                    "answer": answer,
                    "trace": rag_service.last_retrieval_trace,
                    "docs": [
                        {
                            "source": doc.metadata.get("source", ""),
                            "doc_type": doc.metadata.get("doc_type", ""),
                            "model": doc.metadata.get("model", ""),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                        }
                        for doc in rag_service.last_retrieved_docs
                    ],
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"实验结果已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
