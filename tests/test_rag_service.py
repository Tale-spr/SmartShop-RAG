import re
import unittest

from langchain_core.documents import Document

from smartshop_rag.rag.rag_service import RagSummarizeService


class RagServiceModelConfirmationTestCase(unittest.TestCase):
    def setUp(self):
        self.service = RagSummarizeService.__new__(RagSummarizeService)
        self.service._model_pattern = re.compile(r"\bMF-[A-Z0-9]+\b", re.IGNORECASE)
        self.service._weak_feature_pattern = re.compile(r"\b(?:\d+(?:\.\d+)?L|\d{4})\b", re.IGNORECASE)
        self.service._weak_feature_keywords = {
            "可视窗",
            "旋钮",
            "双热源",
            "自动断电",
            "触控",
            "按键",
            "电子可视",
            "方形烤篮",
            "圆形烤篮",
        }
        self.service._manual_intent_keywords = {
            "首次使用",
            "第一次用",
            "不工作",
            "推不进去",
            "异响",
            "白烟",
            "清洁",
            "怎么检查",
            "怎么处理",
            "怎么用",
            "怎么清洗",
            "故障",
            "排查",
        }
        self.service.vector_top_k = 6
        self.service.bm25_top_k = 6
        self.service.bm25_top_k_v2 = 4
        self.service.vector_weight = 0.7
        self.service.bm25_weight = 0.3
        self.service.rrf_k = 60
        self.service.model_mismatch_penalty = 0.5
        self.service.manual_bias_boost = 1.1
        self.service.weighted_rrf_v2_bucket_conf = {
            "explicit_model": {"vector_weight": 0.85, "bm25_weight": 0.15, "rrf_k": 60},
            "weak_feature": {"vector_weight": 0.60, "bm25_weight": 0.40, "rrf_k": 60},
            "generic": {"vector_weight": 0.75, "bm25_weight": 0.25, "rrf_k": 60},
        }

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

    def test_weighted_rrf_combines_vector_and_bm25_ranks(self):
        doc_a = Document(page_content='A', metadata={'chunk_id': 'a', 'model': 'MF-KZC6054'})
        doc_b = Document(page_content='B', metadata={'chunk_id': 'b', 'model': 'MF-KZE5004'})
        vector_results = [
            {'document': doc_a, 'rank': 1, 'score': None, 'source': 'vector'},
            {'document': doc_b, 'rank': 2, 'score': None, 'source': 'vector'},
        ]
        bm25_results = [
            {'document': doc_b, 'rank': 1, 'score': 5.0, 'source': 'bm25'},
        ]

        fused = self.service._weighted_rrf_results(vector_results, bm25_results)
        self.assertEqual(fused[0]['document'].metadata['chunk_id'], 'b')
        self.assertGreater(fused[0]['rrf_score'], fused[1]['rrf_score'])
        self.assertEqual(fused[0]['source'], 'both')

    def test_weighted_rrf_handles_single_route_hits(self):
        doc_a = Document(page_content='A', metadata={'chunk_id': 'a', 'model': 'MF-KZC6054'})
        doc_b = Document(page_content='B', metadata={'chunk_id': 'b', 'model': 'MF-KZE5004'})
        vector_results = [
            {'document': doc_a, 'rank': 1, 'score': None, 'source': 'vector'},
        ]
        bm25_results = [
            {'document': doc_b, 'rank': 1, 'score': 3.0, 'source': 'bm25'},
        ]

        fused = self.service._weighted_rrf_results(vector_results, bm25_results)
        self.assertEqual(len(fused), 2)
        self.assertIsNotNone(fused[0]['rrf_score'])
        self.assertIsNotNone(fused[1]['rrf_score'])

    def test_weighted_rrf_uses_configured_weights(self):
        doc_a = Document(page_content='A', metadata={'chunk_id': 'a', 'model': 'MF-KZC6054'})
        vector_results = [{'document': doc_a, 'rank': 2, 'score': None, 'source': 'vector'}]
        bm25_results = []

        fused = self.service._weighted_rrf_results(vector_results, bm25_results)
        expected = self.service.vector_weight / (self.service.rrf_k + 2)
        self.assertAlmostEqual(fused[0]['rrf_score'], expected)

    def test_query_bucket_detects_explicit_model(self):
        bucket = self.service._determine_query_bucket('MF-KZ30E201 第一次用之前要做什么？', 'MF-KZ30E201 第一次用之前要做什么？', ['MF-KZ30E201'])
        self.assertEqual(bucket, 'explicit_model')

    def test_query_bucket_detects_weak_feature(self):
        bucket = self.service._determine_query_bucket('7L 这款适合一家几口用？', '7L 这款适合一家几口用？', [])
        self.assertEqual(bucket, 'weak_feature')

    def test_query_bucket_detects_generic(self):
        bucket = self.service._determine_query_bucket('空气炸锅适合做什么食物？', '空气炸锅适合做什么食物？', [])
        self.assertEqual(bucket, 'generic')

    def test_manual_intent_detection(self):
        self.assertTrue(self.service._is_manual_intent_query('第一次用 MF-KZ30E201 之前要先做什么？', '第一次用 MF-KZ30E201 之前要先做什么？'))
        self.assertFalse(self.service._is_manual_intent_query('MF-KZE7001 适合几个人用？', 'MF-KZE7001 适合几个人用？'))

    def test_weighted_rrf_v2_penalizes_non_matching_model_for_explicit_query(self):
        doc_match = Document(page_content='manual', metadata={'chunk_id': 'm1', 'model': 'MF-KZ30E201', 'doc_type': 'manual'})
        doc_other = Document(page_content='manual', metadata={'chunk_id': 'm2', 'model': 'MF-KZE5004', 'doc_type': 'manual'})
        vector_results = [
            {'document': doc_match, 'rank': 2, 'score': None, 'source': 'vector'},
            {'document': doc_other, 'rank': 1, 'score': None, 'source': 'vector'},
        ]
        bm25_results = []

        fused, meta = self.service._weighted_rrf_v2_results(
            query_bucket='explicit_model',
            detected_query_models=['MF-KZ30E201'],
            manual_intent=False,
            vector_results=vector_results,
            bm25_results=bm25_results,
        )
        self.assertEqual(fused[0]['document'].metadata['model'], 'MF-KZ30E201')
        self.assertTrue(meta['model_consistency_penalty_applied'])
        penalized = next(item for item in fused if item['document'].metadata['model'] == 'MF-KZE5004')
        self.assertFalse(penalized['model_match'])

    def test_weighted_rrf_v2_keeps_shared_documents_unpenalized(self):
        doc_shared = Document(page_content='policy', metadata={'chunk_id': 's1', 'model': 'shared', 'doc_type': 'policy'})
        vector_results = [{'document': doc_shared, 'rank': 1, 'score': None, 'source': 'vector'}]
        fused, meta = self.service._weighted_rrf_v2_results(
            query_bucket='explicit_model',
            detected_query_models=['MF-KZ30E201'],
            manual_intent=False,
            vector_results=vector_results,
            bm25_results=[],
        )
        self.assertEqual(fused[0]['adjusted_rrf_score'], fused[0]['base_rrf_score'])
        self.assertFalse(fused[0]['model_consistency_penalty_applied'])
        self.assertFalse(meta['model_consistency_penalty_applied'])

    def test_weighted_rrf_v2_boosts_manual_docs_for_manual_intent(self):
        doc_manual = Document(page_content='manual', metadata={'chunk_id': 'm1', 'model': 'MF-KZ30E201', 'doc_type': 'manual'})
        doc_detail = Document(page_content='detail', metadata={'chunk_id': 'd1', 'model': 'MF-KZ30E201', 'doc_type': 'detail'})
        vector_results = [
            {'document': doc_manual, 'rank': 2, 'score': None, 'source': 'vector'},
            {'document': doc_detail, 'rank': 1, 'score': None, 'source': 'vector'},
        ]
        fused, meta = self.service._weighted_rrf_v2_results(
            query_bucket='explicit_model',
            detected_query_models=['MF-KZ30E201'],
            manual_intent=True,
            vector_results=vector_results,
            bm25_results=[],
        )
        self.assertEqual(fused[0]['document'].metadata['doc_type'], 'manual')
        self.assertTrue(meta['manual_bias_applied'])
        boosted = next(item for item in fused if item['document'].metadata['doc_type'] == 'manual')
        self.assertTrue(boosted['manual_bias_applied'])

    def test_weighted_rrf_v2_uses_bucket_specific_weights(self):
        params = self.service._get_weighted_rrf_v2_params('weak_feature')
        self.assertEqual(params['vector_weight'], 0.60)
        self.assertEqual(params['bm25_weight'], 0.40)
        self.assertEqual(params['rrf_k'], 60)


if __name__ == '__main__':
    unittest.main()
