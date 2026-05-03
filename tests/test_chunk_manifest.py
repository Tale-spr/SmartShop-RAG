import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartshop_rag.rag.chunk_manifest import (
    ChunkManifest,
    build_chunked_documents_for_file,
    build_markdown_semantic_chunks,
    calculate_manifest_fingerprint,
    get_chunk_strategy,
    group_markdown_sections,
    parse_markdown_sections,
    read_chunk_manifest,
    write_chunk_manifest,
)


class ChunkManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"chunk_manifest_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "chunks.jsonl"

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_write_and_read_chunk_manifest_round_trips_documents(self):
        doc = Document(
            page_content="MF-KZC6054 是 5.5L 双热源空气炸锅",
            metadata={
                "chunk_id": "data/knowledge_base/air_fryer/midea/MF-KZC6054/cleaned/detail.md#chunk_0",
                "chunk_index": "0",
                "source_path": "data/knowledge_base/air_fryer/midea/MF-KZC6054/cleaned/detail.md",
                "source": "data/knowledge_base/air_fryer/midea/MF-KZC6054/cleaned/detail.md",
                "brand": "midea",
                "model": "MF-KZC6054",
                "doc_type": "detail",
                "source_md5": "abc",
                "heading_path": "商品详情 > 核心卖点",
                "section_index": "0",
            },
        )
        meta = {
            "record_type": "manifest_meta",
            "schema_version": 1,
            "chunk_strategy": {"chunk_strategy_version": 3, "chunk_size": 650, "chunk_overlap": 80},
            "source_files": [{"source_path": doc.metadata["source_path"], "md5": "abc"}],
            "source_file_count": 1,
            "chunk_count": 1,
            "manifest_fingerprint": "fp",
        }

        with patch("smartshop_rag.rag.chunk_manifest.get_abs_path", return_value=str(self.manifest_path)):
            write_chunk_manifest(ChunkManifest(meta=meta, documents=[doc]))
            loaded = read_chunk_manifest()

        self.assertEqual(len(loaded.documents), 1)
        loaded_doc = loaded.documents[0]
        self.assertEqual(loaded_doc.page_content, doc.page_content)
        self.assertEqual(loaded_doc.metadata["chunk_id"], doc.metadata["chunk_id"])
        self.assertEqual(loaded_doc.metadata["chunk_index"], "0")
        self.assertEqual(loaded_doc.metadata["source_path"], doc.metadata["source_path"])
        self.assertEqual(loaded_doc.metadata["model"], "MF-KZC6054")
        self.assertEqual(loaded_doc.metadata["doc_type"], "detail")
        self.assertEqual(loaded_doc.metadata["heading_path"], "商品详情 > 核心卖点")
        self.assertEqual(loaded_doc.metadata["section_index"], "0")

    def test_chunk_strategy_change_changes_fingerprint(self):
        base_meta = {
            "record_type": "manifest_meta",
            "schema_version": 1,
            "chunk_strategy": {"chunk_strategy_version": 3, "chunk_size": 650, "chunk_overlap": 80},
            "source_files": [{"source_path": "a.md", "md5": "abc"}],
            "source_file_count": 1,
            "chunk_count": 2,
        }
        changed_meta = json.loads(json.dumps(base_meta))
        changed_meta["chunk_strategy"]["chunk_size"] = 500

        self.assertNotEqual(calculate_manifest_fingerprint(base_meta), calculate_manifest_fingerprint(changed_meta))

    def test_chunk_strategy_uses_markdown_semantic_version(self):
        strategy = get_chunk_strategy()
        self.assertEqual(strategy["chunk_strategy_version"], 3)
        self.assertEqual(strategy["boundary_strategy"], "markdown_semantic_merged")
        self.assertTrue(strategy["merge_short_sections"])

    def test_parse_markdown_sections_preserves_heading_path(self):
        text = (
            "# 使用说明\n\n"
            "## 清洁保养\n"
            "- 冷却后清洁。\n\n"
            "## 常见问题\n"
            "### 冒出白烟\n"
            "- 油脂高属于正常现象。\n"
        )

        sections = parse_markdown_sections(text)

        self.assertEqual([section.heading_path for section in sections], ["使用说明 > 清洁保养", "使用说明 > 常见问题 > 冒出白烟"])
        self.assertEqual(sections[0].heading_level, 2)
        self.assertIn("## 清洁保养", sections[0].content)
        self.assertNotIn("## 常见问题", sections[0].content)

    def test_markdown_semantic_chunks_split_long_section_with_same_heading_path(self):
        long_body = "\n".join(f"- 清洁步骤 {index}：等待冷却后清洁炸篮和炸桶。" for index in range(30))
        doc = Document(page_content=f"# 使用说明\n\n## 清洁保养\n{long_body}", metadata={})
        splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20, separators=["\n", "。", " ", ""])

        chunks = build_markdown_semantic_chunks(
            [doc],
            base_metadata={"source_path": "manual.md", "source": "manual.md", "model": "MF-KZC6054", "doc_type": "manual"},
            splitter=splitter,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual({chunk.metadata["heading_path"] for chunk in chunks}, {"使用说明 > 清洁保养"})
        self.assertEqual([chunk.metadata["chunk_in_section_index"] for chunk in chunks], [str(index) for index in range(len(chunks))])
        self.assertEqual({chunk.metadata["merged_section_count"] for chunk in chunks}, {"1"})

    def test_markdown_short_sections_are_merged_with_range_metadata(self):
        text = (
            "# 商品详情\n\n"
            "## 核心定位\n"
            "适合家庭日常使用。\n\n"
            "## 核心卖点\n"
            "- 5L 容量\n"
            "- 双旋钮\n\n"
            "## 适用场景\n"
            "- 日常家用\n"
        )
        doc = Document(page_content=text, metadata={})
        splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=80, separators=["\n\n", "\n", " ", ""])

        chunks = build_markdown_semantic_chunks(
            [doc],
            base_metadata={"source_path": "detail.md", "source": "detail.md", "model": "MF-KZE5004", "doc_type": "detail"},
            splitter=splitter,
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("## 核心定位", chunks[0].page_content)
        self.assertIn("## 核心卖点", chunks[0].page_content)
        self.assertIn("## 适用场景", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["merged_section_count"], "3")
        self.assertEqual(chunks[0].metadata["section_index_start"], "0")
        self.assertEqual(chunks[0].metadata["section_index_end"], "2")
        self.assertEqual(
            json.loads(chunks[0].metadata["heading_path_list"]),
            ["商品详情 > 核心定位", "商品详情 > 核心卖点", "商品详情 > 适用场景"],
        )

    def test_markdown_query_hint_section_stays_separate(self):
        sections = parse_markdown_sections(
            "# 商品详情\n\n"
            "## 核心定位\n"
            "主流容量。\n\n"
            "## 核心卖点\n"
            "- 5L 容量\n\n"
            "## 典型问法映射\n"
            "- 这款适合谁？\n"
        )

        groups = group_markdown_sections(sections, max_chars=650)

        self.assertEqual([len(group.sections) for group in groups], [2, 1])
        self.assertIn("典型问法映射", groups[1].heading_path)

    def test_manual_sections_merge_operation_topics_and_keep_faq_separate(self):
        sections = parse_markdown_sections(
            "# 使用说明\n\n"
            "## 快速入门\n"
            "先清洁后放入食材。\n\n"
            "## 清洁保养\n"
            "冷却后清洁。\n\n"
            "## 常见问题\n"
            "### 产品未工作\n"
            "检查电源。\n\n"
            "### 冒出白烟\n"
            "清理残油。\n"
        )

        groups = group_markdown_sections(sections, max_chars=650)

        self.assertEqual([len(group.sections) for group in groups], [2, 2])
        self.assertEqual(groups[0].heading_path, "使用说明 > 快速入门 | 使用说明 > 清洁保养")
        self.assertEqual(groups[1].heading_path, "使用说明 > 常见问题 > 产品未工作 | 使用说明 > 常见问题 > 冒出白烟")

    def test_build_chunked_documents_for_markdown_keeps_heading_context_in_metadata_only(self):
        text = "# 使用说明\n\n## 清洁保养\n- 冷却后清洁。"
        fake_doc = Document(page_content=text, metadata={})
        path = r"E:\Python\SmartShop-RAG\data\knowledge_base\air_fryer\midea\MF-KZC6054\cleaned\manual.md"

        with patch("smartshop_rag.rag.chunk_manifest.load_file_documents", return_value=[fake_doc]):
            chunks = build_chunked_documents_for_file(path, source_md5="abc")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["heading_path"], "使用说明 > 清洁保养")
        self.assertEqual(chunks[0].metadata["section_index"], "0")
        self.assertEqual(chunks[0].metadata["chunk_in_section_index"], "0")
        self.assertNotIn("使用说明 > 清洁保养", chunks[0].page_content)
        self.assertIn("## 清洁保养", chunks[0].page_content)


if __name__ == "__main__":
    unittest.main()
