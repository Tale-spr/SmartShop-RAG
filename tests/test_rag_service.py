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
        self.service.context_postprocess_enabled = True
        self.service.query_hint_penalty = 0.70
        self.service.evidence_boost = 1.08
        self.service.entity_coverage_enabled = True
        self.service.entity_coverage_max_supplemental_docs = 3
        self.service.context_window_target_chars = 650
        self.service.context_window_neighbor_window = 1
        self.service.context_window_max_expanded_docs = 4
        self.service._chunk_by_id = None
        self.service._chunks_by_source = None
        self.service._chunks_by_model = None
        self.service._model_alias_map = None
        self.service._all_chunked_documents = None
        self.service._entity_attribute_keywords = {
            "容量",
            "功率",
            "可视窗",
            "双热源",
            "双可视",
            "旋钮",
            "触控",
            "按键",
            "清洗",
            "故障",
            "白烟",
            "异响",
            "首次使用",
            "第一次",
            "不工作",
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

    def test_bm25_index_uses_manifest_chunk_documents(self):
        docs = [
            Document(page_content='MF-KZC6054 是 5.5L 双热源空气炸锅', metadata={'chunk_id': 'manifest_chunk_1'}),
            Document(page_content='七天无理由退货规则说明', metadata={'chunk_id': 'manifest_chunk_2'}),
        ]
        self.service._all_chunked_documents = None
        self.service._bm25_index = None
        self.service.vector_store = type('FakeVectorStore', (), {'load_all_chunked_documents': lambda _: docs})()

        results = self.service._bm25_retrieve('MF-KZC6054 几升', top_k=2)

        self.assertTrue(results)
        self.assertEqual(results[0]['document'].metadata['chunk_id'], 'manifest_chunk_1')

    def test_section_role_detects_query_hint_and_evidence(self):
        query_hint = Document(page_content='问法', metadata={'heading_path': '商品详情 > 典型问法映射'})
        specs = Document(page_content='参数', metadata={'heading_path': '规格参数', 'doc_type': 'specs'})
        overview = Document(page_content='定位', metadata={'heading_path': '商品详情 > 核心定位'})

        self.assertEqual(self.service._section_role(query_hint), 'query_hint')
        self.assertEqual(self.service._section_role(specs), 'evidence')
        self.assertEqual(self.service._section_role(overview), 'overview')

    def test_section_role_weighting_penalizes_query_hint_and_boosts_evidence(self):
        query_hint = Document(page_content='问法', metadata={'heading_path': '商品详情 > 典型问法映射'})
        evidence = Document(page_content='卖点', metadata={'heading_path': '商品详情 > 核心卖点'})
        results = [
            {'document': query_hint, 'adjusted_rrf_score': 1.0, 'rrf_score': 1.0},
            {'document': evidence, 'adjusted_rrf_score': 0.8, 'rrf_score': 0.8},
        ]

        adjusted = self.service._apply_section_role_weighting(results)

        self.assertEqual(adjusted[0]['document'], evidence)
        self.assertAlmostEqual(adjusted[0]['adjusted_rrf_score'], 0.864)
        self.assertAlmostEqual(next(item for item in adjusted if item['document'] == query_hint)['adjusted_rrf_score'], 0.7)

    def test_extract_query_entities_maps_short_model_alias_and_attributes(self):
        docs = [
            Document(page_content='5089 可视窗', metadata={'chunk_id': 'c1', 'source_path': 'a', 'chunk_index': '0', 'model': 'MF-KZE5089'}),
            Document(page_content='5004 旋钮', metadata={'chunk_id': 'c2', 'source_path': 'b', 'chunk_index': '0', 'model': 'MF-KZE5004'}),
        ]
        self.service._all_chunked_documents = docs

        entities = self.service._extract_query_entities('5089 和 5004 哪个有可视窗？')

        self.assertEqual(entities['models'], ['MF-KZE5004', 'MF-KZE5089'])
        self.assertIn('可视窗', entities['attributes'])

    def test_entity_coverage_supplements_missing_model_evidence(self):
        doc_5089 = Document(
            page_content='MF-KZE5089 可视窗',
            metadata={'chunk_id': 'c1', 'source_path': 'a', 'chunk_index': '0', 'model': 'MF-KZE5089', 'heading_path': '商品详情 > 核心卖点'},
        )
        doc_5004_hint = Document(
            page_content='5004 问法',
            metadata={'chunk_id': 'c2', 'source_path': 'b', 'chunk_index': '0', 'model': 'MF-KZE5004', 'heading_path': '商品详情 > 典型问法映射'},
        )
        doc_5004_evidence = Document(
            page_content='MF-KZE5004 双旋钮操作',
            metadata={'chunk_id': 'c3', 'source_path': 'b', 'chunk_index': '1', 'model': 'MF-KZE5004', 'heading_path': '商品详情 > 核心卖点'},
        )
        self.service._all_chunked_documents = [doc_5089, doc_5004_hint, doc_5004_evidence]

        completed, supplemental_trace, entities = self.service._complete_entity_coverage(
            [doc_5089],
            query='5089 和 5004 哪个有可视窗？',
            normalized_query='5089 5004 可视窗',
        )

        self.assertEqual([doc.metadata['chunk_id'] for doc in completed], ['c1', 'c3'])
        self.assertEqual(supplemental_trace[0]['model'], 'MF-KZE5004')
        self.assertIn('MF-KZE5004', entities['models'])

    def test_context_window_expansion_adds_neighbor_chunks_in_order(self):
        docs = [
            Document(page_content='chunk 0', metadata={'chunk_id': 's#chunk_0', 'source_path': 's', 'chunk_index': '0'}),
            Document(page_content='chunk 1', metadata={'chunk_id': 's#chunk_1', 'source_path': 's', 'chunk_index': '1'}),
            Document(page_content='chunk 2', metadata={'chunk_id': 's#chunk_2', 'source_path': 's', 'chunk_index': '2'}),
            Document(page_content='chunk 3', metadata={'chunk_id': 's#chunk_3', 'source_path': 's', 'chunk_index': '3'}),
        ]
        self.service._all_chunked_documents = docs
        self.service.context_window_neighbor_window = 1
        self.service.context_window_target_chars = 100

        expanded, trace = self.service._expand_context_windows([docs[2]])

        self.assertIn('chunk 1', expanded[0].page_content)
        self.assertIn('chunk 2', expanded[0].page_content)
        self.assertIn('chunk 3', expanded[0].page_content)
        self.assertNotIn('chunk 0', expanded[0].page_content)
        self.assertEqual(trace[0]['expanded_chunk_ids'], ['s#chunk_1', 's#chunk_2', 's#chunk_3'])
        self.assertEqual(expanded[0].metadata['expanded_from_chunk_id'], 's#chunk_2')


if __name__ == '__main__':
    unittest.main()
