import unittest

from smartshop_rag.eval.analyze_ragas_results import detect_metric_keys, infer_issue_tags, summarize_by_category


class AnalyzeRagasResultsTestCase(unittest.TestCase):
    def test_detect_metric_keys_ignores_metadata(self):
        rows = [
            {
                'id': 'af_001',
                'category': 'spec',
                'user_input': 'query',
                'context_precision': 0.8,
                'faithfulness': 0.9,
                'answer_relevancy': 0.7,
            }
        ]
        self.assertEqual(detect_metric_keys(rows), ['context_precision', 'faithfulness', 'answer_relevancy'])

    def test_summarize_by_category_returns_means(self):
        rows = [
            {'category': 'spec', 'context_precision': 0.8, 'faithfulness': 0.6},
            {'category': 'spec', 'context_precision': 0.6, 'faithfulness': 0.8},
            {'category': 'scene', 'context_precision': 0.5, 'faithfulness': 0.5},
        ]
        summary = summarize_by_category(rows, ['context_precision', 'faithfulness'])
        self.assertEqual(summary[0]['category'], 'scene')
        self.assertEqual(summary[1]['category'], 'spec')
        self.assertEqual(summary[1]['context_precision'], 0.7)
        self.assertEqual(summary[1]['faithfulness'], 0.7)

    def test_infer_issue_tags(self):
        row = {
            'context_precision': 0.4,
            'faithfulness': 0.4,
            'answer_relevancy': 0.3,
            'answer_correctness': 0.4,
        }
        tags = infer_issue_tags(row)
        self.assertIn('retrieval_or_rerank', tags)
        self.assertIn('grounding_or_generation', tags)
        self.assertIn('response_focus', tags)


if __name__ == '__main__':
    unittest.main()
