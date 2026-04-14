from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from smartshop_rag.eval.common import current_date_tag, load_jsonl, write_jsonl
from smartshop_rag.model.factory import create_chat_model, create_embedding_model
from smartshop_rag.rag.rag_service import RagSummarizeService
from smartshop_rag.rag.vector_store import VectorStoreService

DEFAULT_QUERY_SET = "data/query_sets/air_fryer_midea_query_set_main_v2.jsonl"
DEFAULT_ANNOTATIONS = "data/eval/ragas/annotations/main_v2_reference_answers_v1.jsonl"


AnswerBuilder = Callable[[str], tuple[str, list[str], dict[str, Any], list[dict[str, Any]]]]


def load_annotation_map(path: str | Path) -> dict[str, str]:
    rows = load_jsonl(path)
    mapping: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        reference = str(row.get("reference", "")).strip()
        if sample_id and reference:
            mapping[sample_id] = reference
    if not mapping:
        raise ValueError("标注文件中没有可用的 reference")
    return mapping


def build_dataset_rows(
    query_rows: list[dict[str, Any]],
    annotations: dict[str, str],
    answer_builder: AnswerBuilder,
) -> list[dict[str, Any]]:
    rows_by_id = {str(row["id"]): row for row in query_rows}
    dataset_rows: list[dict[str, Any]] = []
    for sample_id, reference in annotations.items():
        query_row = rows_by_id.get(sample_id)
        if query_row is None:
            raise KeyError(f"标注样本 {sample_id} 未在 query set 中找到")
        query = str(query_row.get("query", "")).strip()
        response, retrieved_contexts, trace, docs = answer_builder(query)
        dataset_rows.append(
            {
                "id": sample_id,
                "category": query_row.get("category"),
                "difficulty": query_row.get("difficulty"),
                "target_models": query_row.get("target_models", []),
                "mode": trace.get("mode", ""),
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": reference,
                "trace": trace,
                "docs": docs,
            }
        )
    return dataset_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Ragas 可消费的数据集")
    parser.add_argument("--query-set", default=DEFAULT_QUERY_SET)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output")
    parser.add_argument("--mode", default="hybrid_rerank")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    query_rows = load_jsonl(args.query_set)
    annotations = load_annotation_map(args.annotations)
    if args.limit is not None:
        selected_ids = list(annotations.keys())[: args.limit]
        annotations = {sample_id: annotations[sample_id] for sample_id in selected_ids}

    embedding_model = create_embedding_model()
    vector_store_service = VectorStoreService(embedding_function=embedding_model)
    rag_service = RagSummarizeService(
        model=create_chat_model(role="rag_chat"),
        rewrite_model=create_chat_model(role="rewrite_chat"),
        rerank_model=create_chat_model(role="rerank_chat"),
        vector_store_service=vector_store_service,
    )

    def answer_builder(query: str) -> tuple[str, list[str], dict[str, Any], list[dict[str, Any]]]:
        response = rag_service.rag_summarize(query, mode=args.mode)
        contexts = [doc.page_content for doc in rag_service.last_retrieved_docs]
        docs = [
            {
                "source": doc.metadata.get("source", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "model": doc.metadata.get("model", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for doc in rag_service.last_retrieved_docs
        ]
        return response, contexts, rag_service.last_retrieval_trace, docs

    dataset_rows = build_dataset_rows(query_rows, annotations, answer_builder)
    output_path = Path(args.output) if args.output else Path(
        f"data/eval/ragas/datasets/main_v2_ragas_dataset_{current_date_tag()}.jsonl"
    )
    write_jsonl(output_path, dataset_rows)
    print(f"Ragas 数据集已写入: {output_path} | 样本数: {len(dataset_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
