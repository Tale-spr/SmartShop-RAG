import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from smartshop_rag.agent.react_agent import ReactAgent


class FakeRagService:
    def __init__(
        self,
        *,
        retrieval_steps: list[dict],
        rewritten_query: str = "默认改写查询",
        transformed_query: str = "默认补救查询",
        summary: str = "默认知识摘要",
    ):
        self.retrieval_steps = retrieval_steps
        self.rewritten_query = rewritten_query
        self.transformed_query = transformed_query
        self.summary = summary
        self.retrieve_calls: list[dict[str, object]] = []
        self.transform_calls: list[dict[str, str]] = []
        self.summarize_calls: list[dict[str, object]] = []
        self.last_retrieved_docs = []
        self.last_retrieval_trace = {}

    def rewrite_query(self, query: str) -> str:
        return self.rewritten_query

    def transform_query(
        self,
        *,
        question: str,
        current_query: str,
        session_summary: str = "",
        recent_history: str = "",
        retrieval_summary: str = "",
    ) -> str:
        self.transform_calls.append(
            {
                "question": question,
                "current_query": current_query,
                "session_summary": session_summary,
                "recent_history": recent_history,
                "retrieval_summary": retrieval_summary,
            }
        )
        return self.transformed_query

    def retrieve_docs(self, query: str, *, rewrite: bool = True) -> list[Document]:
        call_index = len(self.retrieve_calls)
        step = self.retrieval_steps[min(call_index, len(self.retrieval_steps) - 1)]
        self.retrieve_calls.append({"query": query, "rewrite": rewrite})
        self.last_retrieved_docs = list(step["docs"])
        self.last_retrieval_trace = dict(step["trace"])
        return list(step["docs"])

    def summarize_docs(self, query: str, docs: list[Document]) -> str:
        self.summarize_calls.append({"query": query, "docs": docs})
        return self.summary


class ReactAgentWorkflowTestCase(unittest.TestCase):
    def test_execute_uses_workflow_and_records_trace(self):
        docs = [
            Document(page_content="商品支持七天无理由退货，特殊类目除外。", metadata={"source": "returns.md", "model": "shared", "doc_type": "returns"})
        ]
        trace = {
            "query": "商品支持七天无理由吗？",
            "normalized_query": "商品 支持 七天无理由 吗",
            "vector_hit_count": 1,
            "bm25_hit_count": 1,
            "merged_candidate_count": 1,
            "rerank_selected_count": 1,
            "doc_count": "1",
            "detected_query_models": [],
            "retrieved_models": [],
            "model_confirmation_status": "unconfirmed",
            "model_confirmation_source": "retrieval_inferred",
            "should_reconfirm_model": True,
            "confirmed_model": "",
            "final_docs": [],
        }
        rag_service = FakeRagService(
            retrieval_steps=[{"docs": docs, "trace": trace}],
            rewritten_query="商品 七天无理由 退货 规则",
            summary="商品支持七天无理由退货，特殊类目需以页面规则为准。",
        )
        model = Mock()
        model.invoke.return_value = Mock(content="根据当前资料，这款商品支持七天无理由，具体以页面规则为准。")
        agent = ReactAgent(model=model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_001",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("商品支持七天无理由吗？", runtime_context)

        self.assertIn("七天无理由", answer)
        self.assertEqual(rag_service.retrieve_calls[0]["query"], "商品 七天无理由 退货 规则")
        self.assertEqual(len(runtime_context["retrieved_docs"]), 1)
        self.assertEqual(runtime_context["retrieval_trace"][0]["normalized_query"], "商品 支持 七天无理由 吗")
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router", "query_rewrite", "retrieve_and_decide", "answer_node"])
        visible_event_types = [event["event_type"] for event in runtime_context["status_events"]]
        self.assertIn("stage.intent", visible_event_types)
        self.assertIn("stage.rewrite", visible_event_types)
        self.assertIn("stage.rag", visible_event_types)
        self.assertIn("stage.final", visible_event_types)

    def test_unconfirmed_model_adds_guard_prompt(self):
        docs = [
            Document(page_content="MF-KZC6054 容量约 5.5L 至 6L。", metadata={"source": "specs.md", "model": "MF-KZC6054", "doc_type": "specs"})
        ]
        trace = {
            "query": "我这款容量是多少？",
            "normalized_query": "我 这 款 容量 是 多少",
            "vector_hit_count": 2,
            "bm25_hit_count": 2,
            "merged_candidate_count": 3,
            "rerank_selected_count": 2,
            "doc_count": "2",
            "detected_query_models": [],
            "retrieved_models": ["MF-KZC6054"],
            "model_confirmation_status": "unconfirmed",
            "model_confirmation_source": "retrieval_inferred",
            "should_reconfirm_model": True,
            "confirmed_model": "",
            "final_docs": [],
        }
        rag_service = FakeRagService(
            retrieval_steps=[{"docs": docs, "trace": trace}],
            rewritten_query="空气炸锅 容量 型号 确认",
            summary="候选型号资料主要来自 MF-KZC6054。",
        )
        model = Mock()
        model.invoke.return_value = Mock(content="目前还不能确认具体型号，建议先查看机身铭牌或订单页。")
        agent = ReactAgent(model=model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_002",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("我这款容量是多少？", runtime_context)

        self.assertIn("不能确认", answer)
        prompt_messages = model.invoke.call_args.args[0]
        prompt_text = prompt_messages[1].content
        self.assertIn("型号确认状态: unconfirmed", prompt_text)
        self.assertIn("禁止使用“您这款就是MF-XXX”", prompt_text)
        self.assertIn("MF-KZC6054", prompt_text)

    def test_confirmed_model_allows_specific_answering(self):
        docs = [
            Document(page_content="MF-KZC6054 容量约 5.5L 至 6L。", metadata={"source": "specs.md", "model": "MF-KZC6054", "doc_type": "specs"})
        ]
        trace = {
            "query": "MF-KZC6054 容量是多少？",
            "normalized_query": "MF-KZC6054 容量 是 多少",
            "vector_hit_count": 2,
            "bm25_hit_count": 2,
            "merged_candidate_count": 3,
            "rerank_selected_count": 2,
            "doc_count": "2",
            "detected_query_models": ["MF-KZC6054"],
            "retrieved_models": ["MF-KZC6054"],
            "model_confirmation_status": "confirmed",
            "model_confirmation_source": "explicit_query",
            "should_reconfirm_model": False,
            "confirmed_model": "MF-KZC6054",
            "final_docs": [],
        }
        rag_service = FakeRagService(
            retrieval_steps=[{"docs": docs, "trace": trace}],
            rewritten_query="MF-KZC6054 容量 规格",
            summary="MF-KZC6054 容量约 5.5L 至 6L。",
        )
        model = Mock()
        model.invoke.return_value = Mock(content="MF-KZC6054 容量约 5.5L 至 6L，适合 3 到 5 人日常使用。")
        agent = ReactAgent(model=model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_003",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("MF-KZC6054 容量是多少？", runtime_context)

        self.assertIn("5.5L", answer)
        prompt_messages = model.invoke.call_args.args[0]
        prompt_text = prompt_messages[1].content
        self.assertIn("型号确认状态: confirmed", prompt_text)
        self.assertIn("确认来源: explicit_query", prompt_text)
        self.assertIn("MF-KZC6054", prompt_text)
        self.assertIn("不要再次建议用户查看铭牌", prompt_text)
        self.assertNotIn("优先提示用户查看机身铭牌", prompt_text)

    def test_non_domain_question_returns_polite_refusal_without_retrieval(self):
        rag_service = FakeRagService(retrieval_steps=[{"docs": [], "trace": {"doc_count": "0"}}])
        answer_model = Mock()
        router_model = Mock()
        router_model.invoke.return_value = Mock(content="non_domain")
        agent = ReactAgent(model=answer_model, tools=[], rag_service=rag_service, router_model=router_model)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_004",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("今天天气怎么样？", runtime_context)

        self.assertIn("我目前主要负责空气炸锅相关的商品、使用和售后问题", answer)
        self.assertEqual(rag_service.retrieve_calls, [])
        answer_model.invoke.assert_not_called()
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router"])

    def test_smalltalk_returns_greeting_without_retrieval(self):
        rag_service = FakeRagService(retrieval_steps=[{"docs": [], "trace": {"doc_count": "0"}}])
        answer_model = Mock()
        smalltalk_model = Mock()
        smalltalk_model.invoke.return_value = Mock(content="你好，在的。空气炸锅相关问题可以继续问我。")
        agent = ReactAgent(model=answer_model, tools=[], rag_service=rag_service, smalltalk_model=smalltalk_model)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_006",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("你好", runtime_context)

        self.assertIn("你好", answer)
        self.assertEqual(rag_service.retrieve_calls, [])
        answer_model.invoke.assert_not_called()
        smalltalk_model.invoke.assert_called_once()
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router", "smalltalk_answer"])

    def test_smalltalk_meta_question_uses_smalltalk_model_without_retrieval(self):
        rag_service = FakeRagService(retrieval_steps=[{"docs": [], "trace": {"doc_count": "0"}}])
        answer_model = Mock()
        smalltalk_model = Mock()
        smalltalk_model.invoke.return_value = Mock(content="我没法知道你的真实身份，不过你可以直接告诉我想查什么空气炸锅问题。")
        agent = ReactAgent(model=answer_model, tools=[], rag_service=rag_service, smalltalk_model=smalltalk_model)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_009",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("我是谁？", runtime_context)

        self.assertIn("没法知道你的真实身份", answer)
        self.assertEqual(rag_service.retrieve_calls, [])
        answer_model.invoke.assert_not_called()
        smalltalk_model.invoke.assert_called_once()
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router", "smalltalk_answer"])

    def test_capability_query_returns_scope_description_without_retrieval(self):
        rag_service = FakeRagService(retrieval_steps=[{"docs": [], "trace": {"doc_count": "0"}}])
        answer_model = Mock()
        agent = ReactAgent(model=answer_model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_007",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("你可以帮我做什么？", runtime_context)

        self.assertIn("商品参数", answer)
        self.assertIn("售后规则", answer)
        self.assertEqual(rag_service.retrieve_calls, [])
        answer_model.invoke.assert_not_called()
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router"])

    def test_business_question_takes_priority_over_smalltalk(self):
        docs = [
            Document(page_content="MF-KZC6054 容量约 5.5L 至 6L。", metadata={"source": "specs.md", "model": "MF-KZC6054", "doc_type": "specs"})
        ]
        trace = {
            "query": "MF-KZC6054 容量多少",
            "normalized_query": "MF-KZC6054 容量 多少",
            "vector_hit_count": 1,
            "bm25_hit_count": 1,
            "merged_candidate_count": 1,
            "rerank_selected_count": 1,
            "doc_count": "1",
            "detected_query_models": ["MF-KZC6054"],
            "retrieved_models": ["MF-KZC6054"],
            "model_confirmation_status": "confirmed",
            "model_confirmation_source": "explicit_query",
            "should_reconfirm_model": False,
            "confirmed_model": "MF-KZC6054",
            "final_docs": [],
        }
        rag_service = FakeRagService(
            retrieval_steps=[{"docs": docs, "trace": trace}],
            rewritten_query="MF-KZC6054 容量多少",
            summary="MF-KZC6054 容量约 5.5L 至 6L。",
        )
        model = Mock()
        model.invoke.return_value = Mock(content="MF-KZC6054 容量约 5.5L 至 6L。")
        agent = ReactAgent(model=model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_008",
            "session_summary": "",
            "recent_history": "",
            "status_events": [],
        }

        answer = agent.execute("你好，MF-KZC6054 容量多少？", runtime_context)

        self.assertIn("5.5L", answer)
        self.assertEqual(len(rag_service.retrieve_calls), 1)
        self.assertEqual(runtime_context["trace_tool_calls"], ["intent_router", "query_rewrite", "retrieve_and_decide", "answer_node"])

    def test_retry_once_then_answer(self):
        retry_trace = {
            "query": "这款第一次用要注意什么？",
            "normalized_query": "这 款 第一次 用 要 注意 什么",
            "vector_hit_count": 0,
            "bm25_hit_count": 0,
            "merged_candidate_count": 0,
            "rerank_selected_count": 0,
            "doc_count": "0",
            "detected_query_models": [],
            "retrieved_models": [],
            "model_confirmation_status": "unconfirmed",
            "model_confirmation_source": "retrieval_inferred",
            "should_reconfirm_model": True,
            "confirmed_model": "",
            "final_docs": [],
        }
        success_docs = [
            Document(page_content="首次使用前建议清洗炸桶和烤盘。", metadata={"source": "manual.md", "model": "MF-KZE7001", "doc_type": "manual"})
        ]
        success_trace = {
            "query": "空气炸锅 首次使用 清洗 炸桶 烤盘 注意事项",
            "normalized_query": "空气炸锅 首次使用 清洗 炸桶 烤盘 注意事项",
            "vector_hit_count": 1,
            "bm25_hit_count": 1,
            "merged_candidate_count": 1,
            "rerank_selected_count": 1,
            "doc_count": "1",
            "detected_query_models": [],
            "retrieved_models": [],
            "model_confirmation_status": "unconfirmed",
            "model_confirmation_source": "retrieval_inferred",
            "should_reconfirm_model": True,
            "confirmed_model": "",
            "final_docs": [],
        }
        rag_service = FakeRagService(
            retrieval_steps=[
                {"docs": [], "trace": retry_trace},
                {"docs": success_docs, "trace": success_trace},
            ],
            rewritten_query="这款 第一次 用 注意事项",
            transformed_query="空气炸锅 首次使用 清洗 炸桶 烤盘 注意事项",
            summary="首次使用前建议先清洗炸桶和烤盘。",
        )
        model = Mock()
        model.invoke.return_value = Mock(content="首次使用前建议先清洗炸桶和烤盘，再按说明书操作。")
        agent = ReactAgent(model=model, tools=[], rag_service=rag_service)
        runtime_context = {
            "user_id": "demo_user",
            "session_id": "session_005",
            "session_summary": "最近用户关注: 第一次使用空气炸锅",
            "recent_history": "用户: 这款第一次用要注意什么？",
            "status_events": [],
        }

        answer = agent.execute("这款第一次用要注意什么？", runtime_context)

        self.assertIn("清洗炸桶", answer)
        self.assertEqual(len(rag_service.retrieve_calls), 2)
        self.assertEqual(rag_service.retrieve_calls[0]["query"], "这款 第一次 用 注意事项")
        self.assertEqual(rag_service.retrieve_calls[1]["query"], "空气炸锅 首次使用 清洗 炸桶 烤盘 注意事项")
        self.assertEqual(len(runtime_context["retrieval_trace"]), 2)
        self.assertIn("stage.retry", [event["event_type"] for event in runtime_context["status_events"]])


if __name__ == "__main__":
    unittest.main()
