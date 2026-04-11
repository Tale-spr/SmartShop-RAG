import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from smart_clean_agent.agent.react_agent import ReactAgent


class DummyRagService:
    def __init__(self):
        self.last_retrieved_docs = [
            Document(page_content="商品支持七天无理由退货，特殊类目除外。", metadata={"source": "faq.txt"})
        ]


class DummyTool:
    name = "rag_summarize"

    def __init__(self):
        self._rag_service = DummyRagService()

    def invoke(self, payload):
        return "商品支持七天无理由退货，但特殊类目需以页面规则为准。"


class ReactAgentTestCase(unittest.TestCase):
    def test_execute_uses_rag_and_records_trace(self):
        model = Mock()
        model.invoke.return_value = Mock(content="根据当前资料，这款商品支持七天无理由，具体以页面规则为准。")
        agent = ReactAgent(model=model, tools=[DummyTool()])
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_001",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("这款商品支持七天无理由吗？", runtime_context)

        self.assertIn("七天无理由", answer)
        self.assertEqual(runtime_context["trace_tool_calls"], ["rag_summarize"])
        self.assertEqual(len(runtime_context["retrieved_docs"]), 1)
        self.assertEqual(runtime_context["retrieval_trace"][0]["doc_count"], "1")


if __name__ == "__main__":
    unittest.main()
