from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean
from typing import Any

from smartshop_rag.eval.common import load_jsonl

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


def detect_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    detected: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in METADATA_KEYS or key in detected:
                continue
            if isinstance(value, (int, float)):
                detected.append(key)
    return detected


def summarize_by_category(rows: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category", "unknown")), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for category, category_rows in grouped.items():
        payload: dict[str, Any] = {"category": category, "sample_count": len(category_rows)}
        for key in metric_keys:
            values = [float(item[key]) for item in category_rows if isinstance(item.get(key), (int, float))]
            payload[key] = round(fmean(values), 4) if values else None
        summary_rows.append(payload)
    summary_rows.sort(key=lambda item: item["category"])
    return summary_rows


def infer_issue_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    context_precision = row.get("context_precision")
    faithfulness = row.get("faithfulness")
    answer_relevancy = row.get("answer_relevancy")
    answer_correctness = row.get("answer_correctness")

    if isinstance(context_precision, (int, float)) and context_precision < 0.5:
        tags.append("retrieval_or_rerank")
    if isinstance(faithfulness, (int, float)) and faithfulness < 0.5:
        tags.append("grounding_or_generation")
    if isinstance(answer_relevancy, (int, float)) and answer_relevancy < 0.5:
        tags.append("response_focus")
    if (
        isinstance(answer_correctness, (int, float))
        and answer_correctness < 0.5
        and isinstance(faithfulness, (int, float))
        and faithfulness >= 0.7
    ):
        tags.append("reference_or_coverage_gap")
    return tags or ["needs_manual_review"]


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 Ragas 结果并输出 Markdown 报告")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--output-report")
    args = parser.parse_args()

    rows = load_jsonl(args.results_jsonl)
    if not rows:
        raise ValueError("结果文件为空，无法分析")

    metric_keys = detect_metric_keys(rows)
    overall: dict[str, float | None] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        overall[key] = round(fmean(values), 4) if values else None

    category_rows = summarize_by_category(rows, metric_keys)

    def score_row(row: dict[str, Any]) -> float:
        values = [float(row[key]) for key in metric_keys if isinstance(row.get(key), (int, float))]
        return round(fmean(values), 4) if values else 0.0

    ranked_low = sorted(rows, key=score_row)[:5]

    lines: list[str] = []
    lines.append("# Ragas 结果分析")
    lines.append("")
    lines.append(f"- 结果文件: `{args.results_jsonl}`")
    lines.append(f"- 样本数: `{len(rows)}`")
    lines.append(f"- 指标: `{', '.join(metric_keys)}`")
    lines.append("")
    lines.append("## 总体均值")
    lines.append("")
    for key, value in overall.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 按类别均值")
    lines.append("")
    for row in category_rows:
        metrics_text = "，".join(f"{key}={row[key]}" for key in metric_keys)
        lines.append(f"- `{row['category']}` ({row['sample_count']}条): {metrics_text}")
    lines.append("")
    lines.append("## 低分样本")
    lines.append("")
    for row in ranked_low:
        tags = ", ".join(infer_issue_tags(row))
        lines.append(f"### {row.get('id')} | {row.get('category')}")
        lines.append(f"- query: {row.get('user_input')}")
        lines.append(f"- 平均分: {score_row(row)}")
        lines.append(f"- 标签: {tags}")
        for key in metric_keys:
            lines.append(f"- {key}: {row.get(key)}")
        lines.append("")

    report_text = "\n".join(lines).strip() + "\n"
    output_path = (
        Path(args.output_report)
        if args.output_report
        else Path(str(args.results_jsonl).replace("results/", "reports/").replace("_scores_", "_analysis_").replace(".jsonl", ".md"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Ragas 分析报告已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
