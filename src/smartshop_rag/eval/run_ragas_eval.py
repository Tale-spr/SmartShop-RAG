from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from smartshop_rag.eval.common import current_date_tag, load_jsonl, write_csv, write_json, write_jsonl
from smartshop_rag.model.factory import create_chat_model, create_embedding_model

DEFAULT_DATASET = "data/eval/ragas/datasets/main_v2_ragas_dataset_latest.jsonl"
METADATA_KEYS = {
    "id",
    "category",
    "difficulty",
    "target_models",
    "mode",
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "trace",
    "docs",
}


def _derive_output_paths(
    dataset_path: Path,
    detail_jsonl: str | None,
    detail_csv: str | None,
    summary_json: str | None,
) -> tuple[Path, Path, Path]:
    if detail_jsonl and detail_csv and summary_json:
        return Path(detail_jsonl), Path(detail_csv), Path(summary_json)

    stem = dataset_path.stem.replace("_dataset_", "_scores_")
    if "_scores_" not in stem:
        stem = f"main_v2_ragas_scores_{current_date_tag()}"

    detail_jsonl_path = Path(detail_jsonl) if detail_jsonl else Path("data/eval/ragas/results") / f"{stem}.jsonl"
    detail_csv_path = Path(detail_csv) if detail_csv else Path("data/eval/ragas/results") / f"{stem}.csv"
    summary_json_path = Path(summary_json) if summary_json else Path("data/eval/ragas/results") / f"{stem}_summary.json"
    return detail_jsonl_path, detail_csv_path, summary_json_path


def _mean_of_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not math.isnan(float(row[key]))]
    return round(fmean(values), 4) if values else None


def _detect_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    detected: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in METADATA_KEYS or key in detected:
                continue
            if isinstance(value, (int, float)):
                detected.append(key)
    return detected


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Ragas 评测")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--detail-output-jsonl")
    parser.add_argument("--detail-output-csv")
    parser.add_argument("--summary-output-json")
    parser.add_argument("--with-reference-metrics", action="store_true")
    args = parser.parse_args()

    dataset_rows = load_jsonl(args.dataset_path)
    if not dataset_rows:
        raise ValueError("Ragas 数据集为空，无法评测")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    metrics = [ContextPrecision(), Faithfulness(), AnswerRelevancy()]
    if args.with_reference_metrics:
        from ragas.metrics import AnswerCorrectness, ContextRecall

        metrics.extend([ContextRecall(), AnswerCorrectness()])

    hf_dataset = Dataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "retrieved_contexts": row["retrieved_contexts"],
                "response": row["response"],
                "reference": row["reference"],
            }
            for row in dataset_rows
        ]
    )

    eval_llm = LangchainLLMWrapper(create_chat_model(role="eval_chat"))
    eval_embeddings = LangchainEmbeddingsWrapper(create_embedding_model())
    result = evaluate(dataset=hf_dataset, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings)

    if hasattr(result, "to_pandas"):
        score_rows = result.to_pandas().to_dict(orient="records")
    else:
        score_rows = list(result.scores)

    merged_rows: list[dict[str, Any]] = []
    for sample_row, score_row in zip(dataset_rows, score_rows):
        merged = dict(sample_row)
        merged.update(score_row)
        merged_rows.append(merged)

    metric_keys = _detect_metric_keys(merged_rows)
    summary = {
        "dataset_path": args.dataset_path,
        "sample_count": len(merged_rows),
        "metrics": metric_keys,
        "averages": {key: _mean_of_metric(merged_rows, key) for key in metric_keys},
    }

    detail_jsonl_path, detail_csv_path, summary_json_path = _derive_output_paths(
        Path(args.dataset_path),
        args.detail_output_jsonl,
        args.detail_output_csv,
        args.summary_output_json,
    )
    write_jsonl(detail_jsonl_path, merged_rows)
    write_csv(detail_csv_path, merged_rows)
    write_json(summary_json_path, summary)
    print(f"Ragas 明细已写入: {detail_jsonl_path}")
    print(f"Ragas CSV 已写入: {detail_csv_path}")
    print(f"Ragas 汇总已写入: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
