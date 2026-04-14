import unittest

from langchain_core.documents import Document

from smartshop_rag.rag.bm25_retriever import BM25Index, tokenize_for_bm25


class BM25RetrieverTestCase(unittest.TestCase):
    def test_tokenize_for_bm25_keeps_alnum_and_chinese(self):
        tokens = tokenize_for_bm25('MF-KZC6054 适合几个人用')
        self.assertIn('mf-kzc6054', tokens)
        self.assertIn('适', tokens)

    def test_bm25_search_returns_relevant_doc(self):
        docs = [
            Document(page_content='MF-KZC6054 是 5.5L 双热源空气炸锅', metadata={'chunk_id': '1'}),
            Document(page_content='七天无理由退货规则说明', metadata={'chunk_id': '2'}),
        ]
        index = BM25Index(docs)
        matches = index.search('6054是几升', top_k=2)
        self.assertTrue(matches)
        self.assertEqual(matches[0].document.metadata['chunk_id'], '1')


if __name__ == '__main__':
    unittest.main()
