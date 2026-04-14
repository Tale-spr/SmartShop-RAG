import re
import unittest

from langchain_core.documents import Document

from smartshop_rag.rag.rag_service import RagSummarizeService


class RagServiceModelConfirmationTestCase(unittest.TestCase):
    def setUp(self):
        self.service = RagSummarizeService.__new__(RagSummarizeService)
        self.service._model_pattern = re.compile(r"\bMF-[A-Z0-9]+\b", re.IGNORECASE)

    def test_extract_query_models_keeps_unique_uppercase_models(self):
        models = self.service._extract_query_models('mf-kzc6054 和 MF-KZE7001 哪个更大？', 'MF-KZC6054 哪个更大')
        self.assertEqual(models, ['MF-KZC6054', 'MF-KZE7001'])

    def test_model_confirmation_status_is_unconfirmed_without_query_model(self):
        status = self.service._determine_model_confirmation_status(
            detected_query_models=[],
            retrieved_models=['MF-KZC6054'],
        )
        self.assertEqual(status, 'unconfirmed')

    def test_model_confirmation_status_is_confirmed_when_retrieved_models_match_query_models(self):
        status = self.service._determine_model_confirmation_status(
            detected_query_models=['MF-KZC6054', 'MF-KZE7001'],
            retrieved_models=['MF-KZC6054'],
        )
        self.assertEqual(status, 'confirmed')

    def test_model_confirmation_source_is_explicit_query_when_status_confirmed(self):
        source = self.service._determine_model_confirmation_source(
            detected_query_models=['MF-KZC6054'],
            retrieved_models=['MF-KZC6054'],
            model_confirmation_status='confirmed',
        )
        self.assertEqual(source, 'explicit_query')
        self.assertFalse(self.service._should_reconfirm_model(model_confirmation_source=source))

    def test_model_confirmation_source_is_retrieval_inferred_without_query_model(self):
        source = self.service._determine_model_confirmation_source(
            detected_query_models=[],
            retrieved_models=['MF-KZC6054'],
            model_confirmation_status='unconfirmed',
        )
        self.assertEqual(source, 'retrieval_inferred')
        self.assertTrue(self.service._should_reconfirm_model(model_confirmation_source=source))

    def test_model_confirmation_status_is_ambiguous_when_retrieved_models_conflict(self):
        status = self.service._determine_model_confirmation_status(
            detected_query_models=['MF-KZC6054'],
            retrieved_models=['MF-KZE7001'],
        )
        self.assertEqual(status, 'ambiguous')

    def test_extract_retrieved_models_ignores_shared_documents(self):
        results = [
            {'document': Document(page_content='规则', metadata={'model': 'shared'})},
            {'document': Document(page_content='规格', metadata={'model': 'MF-KZC6054'})},
            {'document': Document(page_content='说明', metadata={'model': 'MF-KZC6054'})},
        ]
        models = self.service._extract_retrieved_models(results)
        self.assertEqual(models, ['MF-KZC6054'])


if __name__ == '__main__':
    unittest.main()
