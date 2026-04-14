import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from smartshop_rag.rag.vector_store import VectorStoreService, ensure_vector_store_ready, get_knowledge_source_files, get_vector_store_sqlite_path, vector_store_exists


class VectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path('tests') / '.tmp' / f'vector_store_{uuid.uuid4().hex}'
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_vector_store_exists_returns_false_when_sqlite_missing(self):
        self.assertFalse(vector_store_exists(str(self.base_dir)))

    def test_vector_store_exists_returns_true_when_sqlite_exists(self):
        sqlite_path = get_vector_store_sqlite_path(str(self.base_dir))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_path.write_text('', encoding='utf-8')
        self.assertTrue(vector_store_exists(str(self.base_dir)))

    def test_ensure_vector_store_ready_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            ensure_vector_store_ready(str(self.base_dir))

    def test_get_knowledge_source_files_raises_when_data_dir_missing(self):
        with self.assertRaises(FileNotFoundError):
            get_knowledge_source_files(str(self.base_dir / 'missing_data'))

    def test_vector_store_service_uses_injected_embedding_function(self):
        embedding = object()
        with patch('smartshop_rag.rag.vector_store.Chroma') as mock_chroma:
            with patch('smartshop_rag.rag.vector_store.create_embedding_model') as mock_factory:
                VectorStoreService(embedding_function=embedding)
        mock_factory.assert_not_called()
        self.assertEqual(mock_chroma.call_args.kwargs['embedding_function'], embedding)

    def test_get_chunked_documents_enriches_metadata(self):
        with patch('smartshop_rag.rag.vector_store.Chroma'):
            with patch('smartshop_rag.rag.vector_store.create_embedding_model'):
                service = VectorStoreService(embedding_function=Mock())
        fake_doc = Document(page_content='内容', metadata={})
        with patch('smartshop_rag.rag.vector_store._load_file_documents', return_value=[fake_doc]):
            docs = service.get_chunked_documents(r'E:\Python\SmartShop-RAG\data\knowledge_baseir_fryer\midea\MF-KZ30E201\cleaned\detail.md')
        self.assertTrue(docs)
        metadata = docs[0].metadata
        self.assertEqual(metadata['brand'], 'midea')
        self.assertEqual(metadata['model'], 'MF-KZ30E201')
        self.assertEqual(metadata['doc_type'], 'detail')
        self.assertIn('chunk_id', metadata)


if __name__ == '__main__':
    unittest.main()
