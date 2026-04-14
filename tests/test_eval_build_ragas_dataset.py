import unittest

from smartshop_rag.eval.build_ragas_dataset import build_dataset_rows, load_annotation_map


class BuildRagasDatasetTestCase(unittest.TestCase):
    def test_load_annotation_map_keeps_valid_rows(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "annotations.jsonl"
            path.write_text(
                "{\"id\": \"af_001\", \"reference\": \"答案1\"}\n"
                "{\"id\": \"\", \"reference\": \"忽略\"}\n"
                "{\"id\": \"af_002\", \"reference\": \"答案2\"}\n",
                encoding="utf-8",
            )
            result = load_annotation_map(path)
        self.assertEqual(result, {"af_001": "答案1", "af_002": "答案2"})

    def test_build_dataset_rows_uses_annotations_and_builder(self):
        query_rows = [
            {
                "id": "af_001",
                "query": "MF-KZ30E201 适合宿舍吗",
                "category": "scenario",
                "difficulty": "easy",
                "target_models": ["MF-KZ30E201"],
            }
        ]
        annotations = {"af_001": "适合宿舍但要注意宿舍功率限制"}

        def answer_builder(query: str):
            self.assertEqual(query, "MF-KZ30E201 适合宿舍吗")
            return (
                "可以，但先看宿舍用电规则。",
                ["3L 小容量，适合小空间。"],
                {"mode": "hybrid_rerank", "final_docs": []},
                [{"source": "detail.md", "doc_type": "detail", "model": "MF-KZ30E201", "chunk_id": "c1"}],
            )

        rows = build_dataset_rows(query_rows, annotations, answer_builder)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], "af_001")
        self.assertEqual(row["reference"], "适合宿舍但要注意宿舍功率限制")
        self.assertEqual(row["response"], "可以，但先看宿舍用电规则。")
        self.assertEqual(row["retrieved_contexts"], ["3L 小容量，适合小空间。"])
        self.assertEqual(row["mode"], "hybrid_rerank")


if __name__ == "__main__":
    unittest.main()
