import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from smartshop_rag.agent.react_agent import ReactAgent


class DummyRagService:
    def __init__(self, *, trace=None, docs=None):
        self.last_retrieved_docs = docs or [
            Document(page_content='商品支持七天无理由退货，特殊类目除外。', metadata={'source': 'faq.txt', 'model': 'shared'})
        ]
        self.last_retrieval_trace = trace or {
            'query': '这款商品支持七天无理由吗？',
            'normalized_query': '这款商品 支持 七天无理由 退货 吗',
            'vector_hit_count': 1,
            'bm25_hit_count': 1,
            'merged_candidate_count': 1,
            'rerank_selected_count': 1,
            'doc_count': '1',
            'detected_query_models': [],
            'retrieved_models': [],
            'model_confirmation_status': 'unconfirmed',
            'final_docs': [
                {
                    'chunk_id': 'faq.txt#chunk_0',
                    'source': 'both',
                    'rank': 1,
                    'doc_type': 'service',
                    'model': 'shared',
                    'score': None,
                }
            ],
        }


class DummyTool:
    name = 'rag_summarize'

    def __init__(self, *, summary=None, trace=None, docs=None):
        self._rag_service = DummyRagService(trace=trace, docs=docs)
        self._summary = summary or '商品支持七天无理由退货，但特殊类目需以页面规则为准。'

    def invoke(self, payload):
        return self._summary


class ReactAgentTestCase(unittest.TestCase):
    def test_execute_uses_rag_and_records_trace(self):
        model = Mock()
        model.invoke.return_value = Mock(content='根据当前资料，这款商品支持七天无理由，具体以页面规则为准。')
        agent = ReactAgent(model=model, tools=[DummyTool()])
        runtime_context = {
            'user_id': 'demo_user',
            'session_id': 'session_001',
            'session_summary': '',
            'recent_history': '',
            'status_events': [],
        }

        answer = agent.execute('这款商品支持七天无理由吗？', runtime_context)

        self.assertIn('七天无理由', answer)
        self.assertEqual(runtime_context['trace_tool_calls'], ['rag_summarize'])
        self.assertEqual(len(runtime_context['retrieved_docs']), 1)
        self.assertEqual(runtime_context['retrieval_trace'][0]['doc_count'], '1')
        self.assertEqual(runtime_context['retrieval_trace'][0]['normalized_query'], '这款商品 支持 七天无理由 退货 吗')

    def test_unconfirmed_model_adds_guard_prompt(self):
        model = Mock()
        model.invoke.return_value = Mock(content='目前还不能确认具体型号，建议先查看机身铭牌或订单页。')
        trace = {
            'query': '我这款容量是多少？',
            'normalized_query': '我 这 款 容量 是 多少',
            'vector_hit_count': 2,
            'bm25_hit_count': 2,
            'merged_candidate_count': 3,
            'rerank_selected_count': 2,
            'doc_count': '2',
            'detected_query_models': [],
            'retrieved_models': ['MF-KZC6054'],
            'model_confirmation_status': 'unconfirmed',
            'model_confirmation_source': 'retrieval_inferred',
            'should_reconfirm_model': True,
            'confirmed_model': '',
            'final_docs': [],
        }
        docs = [
            Document(page_content='MF-KZC6054 容量约 5.5L 至 6L。', metadata={'source': 'specs.md', 'model': 'MF-KZC6054'})
        ]
        agent = ReactAgent(model=model, tools=[DummyTool(summary='候选型号资料主要来自 MF-KZC6054。', trace=trace, docs=docs)])
        runtime_context = {
            'user_id': 'demo_user',
            'session_id': 'session_002',
            'session_summary': '',
            'recent_history': '',
            'status_events': [],
        }

        answer = agent.execute('我这款容量是多少？', runtime_context)

        self.assertIn('不能确认', answer)
        prompt_messages = model.invoke.call_args.args[0]
        prompt_text = prompt_messages[1].content
        self.assertIn('型号确认状态: unconfirmed', prompt_text)
        self.assertIn('禁止使用“您这款就是MF-XXX”', prompt_text)
        self.assertIn('MF-KZC6054', prompt_text)

    def test_confirmed_model_allows_specific_answering(self):
        model = Mock()
        model.invoke.return_value = Mock(content='MF-KZC6054 容量约 5.5L 至 6L，适合 3 到 5 人日常使用。')
        trace = {
            'query': 'MF-KZC6054 容量是多少？',
            'normalized_query': 'MF-KZC6054 容量 是 多少',
            'vector_hit_count': 2,
            'bm25_hit_count': 2,
            'merged_candidate_count': 3,
            'rerank_selected_count': 2,
            'doc_count': '2',
            'detected_query_models': ['MF-KZC6054'],
            'retrieved_models': ['MF-KZC6054'],
            'model_confirmation_status': 'confirmed',
            'model_confirmation_source': 'explicit_query',
            'should_reconfirm_model': False,
            'confirmed_model': 'MF-KZC6054',
            'final_docs': [],
        }
        docs = [
            Document(page_content='MF-KZC6054 容量约 5.5L 至 6L。', metadata={'source': 'specs.md', 'model': 'MF-KZC6054'})
        ]
        agent = ReactAgent(model=model, tools=[DummyTool(summary='MF-KZC6054 容量约 5.5L 至 6L。', trace=trace, docs=docs)])
        runtime_context = {
            'user_id': 'demo_user',
            'session_id': 'session_003',
            'session_summary': '',
            'recent_history': '',
            'status_events': [],
        }

        answer = agent.execute('MF-KZC6054 容量是多少？', runtime_context)

        self.assertIn('5.5L', answer)
        prompt_messages = model.invoke.call_args.args[0]
        prompt_text = prompt_messages[1].content
        self.assertIn('型号确认状态: confirmed', prompt_text)
        self.assertIn('确认来源: explicit_query', prompt_text)
        self.assertIn('MF-KZC6054', prompt_text)
        self.assertIn('不要再次建议用户查看铭牌', prompt_text)
        self.assertNotIn('优先提示用户查看机身铭牌', prompt_text)

    def test_conflicted_model_still_requires_reconfirmation(self):
        model = Mock()
        model.invoke.return_value = Mock(content='当前型号信息存在不一致，建议先核对铭牌。')
        trace = {
            'query': '型号是 MF-KZE7001',
            'normalized_query': '型号 是 MF-KZE7001',
            'vector_hit_count': 2,
            'bm25_hit_count': 2,
            'merged_candidate_count': 3,
            'rerank_selected_count': 2,
            'doc_count': '2',
            'detected_query_models': ['MF-KZE7001'],
            'retrieved_models': ['MF-KZC6054'],
            'model_confirmation_status': 'ambiguous',
            'model_confirmation_source': 'conflicted',
            'should_reconfirm_model': True,
            'confirmed_model': '',
            'final_docs': [],
        }
        docs = [
            Document(page_content='MF-KZC6054 容量约 5.5L 至 6L。', metadata={'source': 'specs.md', 'model': 'MF-KZC6054'})
        ]
        agent = ReactAgent(model=model, tools=[DummyTool(summary='当前证据更多来自 MF-KZC6054。', trace=trace, docs=docs)])
        runtime_context = {
            'user_id': 'demo_user',
            'session_id': 'session_004',
            'session_summary': '',
            'recent_history': '',
            'status_events': [],
        }

        agent.execute('型号是 MF-KZE7001', runtime_context)

        prompt_messages = model.invoke.call_args.args[0]
        prompt_text = prompt_messages[1].content
        self.assertIn('确认来源: conflicted', prompt_text)
        self.assertIn('优先提示用户查看机身铭牌', prompt_text)
        self.assertNotIn('不要再次建议用户查看铭牌', prompt_text)


if __name__ == '__main__':
    unittest.main()
