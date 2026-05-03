import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from smartshop_rag.rag.chunk_manifest import ChunkManifest
from smartshop_rag.rag.vector_store import VectorStoreService, ensure_chunk_manifest_ready, ensure_vector_store_ready, get_knowledge_source_files, get_vector_store_sqlite_path, vector_store_exists


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
                service = VectorStoreService(embedding_function=embedding)
                service._get_vector_store()
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

    def test_load_all_chunked_documents_reads_manifest(self):
        with patch('smartshop_rag.rag.vector_store.Chroma'):
            with patch('smartshop_rag.rag.vector_store.create_embedding_model'):
                service = VectorStoreService(embedding_function=Mock())
        doc = Document(page_content='manifest 内容', metadata={'chunk_id': 'c1'})
        manifest = ChunkManifest(meta={'record_type': 'manifest_meta'}, documents=[doc])
        with patch('smartshop_rag.rag.vector_store.read_chunk_manifest', return_value=manifest):
            with patch('smartshop_rag.rag.vector_store._load_file_documents') as mock_loader:
                docs = service.load_all_chunked_documents()
        mock_loader.assert_not_called()
        self.assertEqual(docs[0].metadata['chunk_id'], 'c1')

    def test_ensure_chunk_manifest_ready_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            ensure_chunk_manifest_ready(str(self.base_dir / 'missing.jsonl'))

    def test_load_document_skips_when_manifest_and_chroma_are_current(self):
        current = ChunkManifest(
            meta={'manifest_fingerprint': 'same', 'source_file_count': 2},
            documents=[Document(page_content='A', metadata={'chunk_id': 'a'})],
        )
        with patch('smartshop_rag.rag.vector_store.Chroma'):
            with patch('smartshop_rag.rag.vector_store.create_embedding_model'):
                service = VectorStoreService(embedding_function=Mock())
        with patch('smartshop_rag.rag.vector_store.build_current_chunk_manifest', return_value=current):
            with patch('smartshop_rag.rag.vector_store.read_manifest_fingerprint', return_value='same'):
                with patch('smartshop_rag.rag.vector_store.chunk_manifest_exists', return_value=True):
                    with patch('smartshop_rag.rag.vector_store.vector_store_exists', return_value=True):
                        stats = service.load_document()
        self.assertEqual(stats['skipped'], 2)

    def test_load_document_reset_rebuilds_manifest_and_chroma(self):
        docs = [Document(page_content='A', metadata={'chunk_id': 'a'})]
        current = ChunkManifest(meta={'manifest_fingerprint': 'new', 'source_file_count': 1}, documents=docs)
        with patch('smartshop_rag.rag.vector_store.Chroma') as mock_chroma:
            with patch('smartshop_rag.rag.vector_store.create_embedding_model'):
                service = VectorStoreService(embedding_function=Mock())
                with patch('smartshop_rag.rag.vector_store.build_current_chunk_manifest', return_value=current):
                    with patch('smartshop_rag.rag.vector_store.clear_generated_index_storage') as mock_clear:
                        with patch('smartshop_rag.rag.vector_store.write_chunk_manifest') as mock_write:
                            stats = service.load_document(reset=True)
        mock_clear.assert_called_once_with(remove_manifest=True)
        mock_write.assert_called_once_with(current)
        mock_chroma.return_value.add_documents.assert_called_once_with(docs, ids=['a'])
        self.assertEqual(stats['loaded'], 1)


if __name__ == '__main__':
    unittest.main()
