from typing import Any

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from smartshop_rag.agent.agentic_state import AgenticState
from smartshop_rag.agent.runtime_context import AgentRuntimeContext
from smartshop_rag.agent.tools.middleware import build_runtime_context_prompt
from smartshop_rag.rag.rag_service import RagSummarizeService
from smartshop_rag.services.status_event_service import record_status_event
from smartshop_rag.utils.prompt_loader import load_intent_router_prompt, load_smalltalk_answer_prompt, load_system_prompts


class ReactAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        *,
        rag_service: RagSummarizeService | None = None,
        router_model: BaseChatModel | None = None,
        smalltalk_model: BaseChatModel | None = None,
    ):
        self.chat_model = model
        self.router_model = router_model or model
        self.smalltalk_model = smalltalk_model or self.router_model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.rag_service = rag_service or self._infer_rag_service_from_tools()
        if self.rag_service is None:
            raise ValueError("未配置可用的 RagSummarizeService")
        self.intent_router_prompt = load_intent_router_prompt()
        self.smalltalk_answer_prompt = load_smalltalk_answer_prompt()
        self._workflow = self._build_workflow()

    def execute_stream(self, query: str, runtime_context: AgentRuntimeContext):
        yield self.execute(query, runtime_context)

    def execute(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        initial_state = self._build_initial_state(query, runtime_context)
        result = self._workflow.invoke(initial_state)
        self._sync_runtime_context(runtime_context, result)
        return str(result.get("final_answer") or "").strip()

    def _build_workflow(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("intent_router", self._intent_router_node)
        workflow.add_node("query_rewrite", self._query_rewrite_node)
        workflow.add_node("retrieve_and_decide", self._retrieve_and_decide_node)
        workflow.add_node("answer_node", self._answer_node)
        workflow.add_edge(START, "intent_router")
        workflow.add_conditional_edges(
            "intent_router",
            self._route_after_intent,
            {
                "query_rewrite": "query_rewrite",
                "answer_node": "answer_node",
            },
        )
        workflow.add_edge("query_rewrite", "retrieve_and_decide")
        workflow.add_edge("retrieve_and_decide", "answer_node")
        workflow.add_edge("answer_node", END)
        return workflow.compile()

    def _build_initial_state(self, query: str, runtime_context: AgentRuntimeContext) -> AgenticState:
        runtime_context.setdefault("trace_tool_calls", [])
        runtime_context.setdefault("retrieval_trace", [])
        runtime_context.setdefault("retrieved_docs", [])
        runtime_context.setdefault("status_events", [])
        return {
            "question": query,
            "user_id": runtime_context.get("user_id", ""),
            "session_id": runtime_context.get("session_id", ""),
            "session_summary": runtime_context.get("session_summary", ""),
            "recent_history": runtime_context.get("recent_history", ""),
            "documents": [],
            "retrieved_docs": list(runtime_context.get("retrieved_docs", [])),
            "retrieval_trace": list(runtime_context.get("retrieval_trace", [])),
            "retry_count": 0,
            "max_retry": 1,
            "trace_tool_calls": list(runtime_context.get("trace_tool_calls", [])),
            "status_events": runtime_context.get("status_events", []),
            "status_event_callback": runtime_context.get("status_event_callback"),
        }

    def _sync_runtime_context(self, runtime_context: AgentRuntimeContext, state: AgenticState) -> None:
        runtime_context["trace_tool_calls"] = list(state.get("trace_tool_calls") or [])
        runtime_context["retrieval_trace"] = list(state.get("retrieval_trace") or [])
        runtime_context["retrieved_docs"] = list(state.get("retrieved_docs") or [])

    def _route_after_intent(self, state: AgenticState) -> str:
        return "answer_node" if state.get("intent") in {"smalltalk", "capability_query", "non_domain"} else "query_rewrite"

    def _intent_router_node(self, state: AgenticState) -> dict[str, Any]:
        self._append_trace_call(state, "intent_router")
        record_status_event(
            state,
            event_type="stage.intent",
            title="正在判断问题类型",
            detail="正在识别是否属于空气炸锅商品知识与售后场景",
        )
        intent = self._classify_intent(state)
        return {"intent": intent, "trace_tool_calls": list(state.get("trace_tool_calls") or [])}

    def _query_rewrite_node(self, state: AgenticState) -> dict[str, Any]:
        self._append_trace_call(state, "query_rewrite")
        record_status_event(
            state,
            event_type="stage.rewrite",
            title="正在改写检索问题",
            detail="正在结合当前问题与上下文生成更适合检索的查询",
        )
        rewritten_query = self.rag_service.rewrite_query(state["question"])
        return {
            "rewritten_query": rewritten_query,
            "active_query": rewritten_query,
            "trace_tool_calls": list(state.get("trace_tool_calls") or []),
        }

    def _retrieve_and_decide_node(self, state: AgenticState) -> dict[str, Any]:
        self._append_trace_call(state, "retrieve_and_decide")
        retrieval_update = self._run_retrieval_once(
            state=state,
            query=state.get("active_query") or state["question"],
            event_title="正在检索知识库",
            event_detail="正在进行混合召回、重排与型号确认",
        )
        decision = self._determine_retrieval_decision(
            intent=state.get("intent", "product_qa"),
            documents=retrieval_update["documents"],
            trace=retrieval_update["latest_trace"],
            retry_count=int(state.get("retry_count", 0)),
            max_retry=int(state.get("max_retry", 1)),
        )
        if decision == "retry" and int(state.get("retry_count", 0)) < int(state.get("max_retry", 1)):
            record_status_event(
                state,
                event_type="stage.retry",
                title="正在进行补充检索",
                detail="第一次检索证据不足，正在调整查询后补充检索",
            )
            self._append_trace_call(state, "transform_query")
            transformed_query = self.rag_service.transform_query(
                question=state["question"],
                current_query=state.get("active_query") or state["question"],
                session_summary=state.get("session_summary", ""),
                recent_history=state.get("recent_history", ""),
                retrieval_summary=self._build_retrieval_summary(
                    retrieval_update["latest_trace"],
                    retrieval_update["documents"],
                ),
            )
            retry_count = int(state.get("retry_count", 0)) + 1
            retry_update = self._run_retrieval_once(
                state={
                    **state,
                    **retrieval_update,
                    "active_query": transformed_query,
                    "retry_count": retry_count,
                },
                query=transformed_query,
                event_title="正在进行补充检索",
                event_detail="正在使用补救查询重新检索说明书、参数和售后资料",
            )
            decision = self._determine_retrieval_decision(
                intent=state.get("intent", "product_qa"),
                documents=retry_update["documents"],
                trace=retry_update["latest_trace"],
                retry_count=retry_count,
                max_retry=int(state.get("max_retry", 1)),
            )
            if decision == "retry":
                decision = "fallback"
            return {
                **retry_update,
                "retrieval_decision": decision,
                "transformed_query": transformed_query,
                "active_query": transformed_query,
                "retry_count": retry_count,
                "trace_tool_calls": list(state.get("trace_tool_calls") or []),
            }
        if decision == "retry":
            decision = "fallback"
        return {
            **retrieval_update,
            "retrieval_decision": decision,
            "trace_tool_calls": list(state.get("trace_tool_calls") or []),
        }

    def _answer_node(self, state: AgenticState) -> dict[str, Any]:
        if state.get("intent") == "smalltalk":
            answer = self._generate_smalltalk_answer(state)
            return {
                "answer": answer,
                "final_answer": answer,
                "answer_decision": "fallback",
                "trace_tool_calls": list(state.get("trace_tool_calls") or []),
            }

        if state.get("intent") == "capability_query":
            answer = self._build_capability_answer()
            return {
                "answer": answer,
                "final_answer": answer,
                "answer_decision": "fallback",
            }

        if state.get("intent") == "non_domain":
            answer = self._build_non_domain_answer()
            return {
                "answer": answer,
                "final_answer": answer,
                "answer_decision": "fallback",
            }

        if state.get("retrieval_decision") != "enough" or not state.get("documents"):
            record_status_event(
                state,
                event_type="stage.fallback",
                title="当前资料不足，转为保守回答",
                detail="为了避免误导，系统将基于现有知识范围给出保守回复",
            )
            answer = self._build_fallback_answer(state)
            return {
                "answer": answer,
                "final_answer": answer,
                "answer_decision": "fallback",
            }

        self._append_trace_call(state, "answer_node")
        record_status_event(
            state,
            event_type="stage.final",
            title="正在生成最终回答",
            detail="正在结合检索结果、会话上下文和型号确认信息组织最终回复",
        )
        docs = [doc for doc in state.get("documents", []) if isinstance(doc, Document)]
        rag_summary = self.rag_service.summarize_docs(state["question"], docs).strip()
        answer = self._generate_answer(
            query=state["question"],
            rag_summary=rag_summary,
            docs=docs,
            runtime_context=state,
        ).strip()
        answer_decision = "clarify" if self._needs_model_clarification(state) else "answer"
        return {
            "answer": answer,
            "final_answer": answer,
            "answer_decision": answer_decision,
            "trace_tool_calls": list(state.get("trace_tool_calls") or []),
        }

    def _run_retrieval_once(
        self,
        *,
        state: AgenticState,
        query: str,
        event_title: str,
        event_detail: str,
    ) -> dict[str, Any]:
        record_status_event(
            state,
            event_type="stage.rag",
            title=event_title,
            detail=event_detail,
        )
        docs = self.rag_service.retrieve_docs(query, rewrite=False)
        trace = dict(self.rag_service.last_retrieval_trace or {"query": state["question"], "doc_count": str(len(docs))})
        retrieval_trace = list(state.get("retrieval_trace") or [])
        retrieval_trace.append(trace)
        return {
            "documents": docs,
            "retrieved_docs": [self._serialize_doc(doc) for doc in docs],
            "retrieval_trace": retrieval_trace,
            "latest_trace": trace,
            "model_confirmation_status": str(trace.get("model_confirmation_status", "")).strip() or "unconfirmed",
            "model_confirmation_source": str(trace.get("model_confirmation_source", "")).strip() or "retrieval_inferred",
            "should_reconfirm_model": bool(trace.get("should_reconfirm_model", True)),
            "confirmed_model": str(trace.get("confirmed_model", "")).strip(),
            "detected_query_models": list(trace.get("detected_query_models") or []),
            "retrieved_models": list(trace.get("retrieved_models") or []),
        }

    def _determine_retrieval_decision(
        self,
        *,
        intent: str,
        documents: list[Document],
        trace: dict[str, Any],
        retry_count: int,
        max_retry: int,
    ) -> str:
        if not documents:
            return "retry" if retry_count < max_retry else "fallback"

        if intent == "policy_qa" and not self._has_policy_evidence(documents):
            return "retry" if retry_count < max_retry else "fallback"

        return "enough"

    def _classify_intent(self, state: AgenticState) -> str:
        heuristic = self._heuristic_intent(state["question"])
        if heuristic:
            return heuristic

        prompt = self.intent_router_prompt.format(
            question=state["question"],
            session_summary=state.get("session_summary", "") or "无",
            recent_history=state.get("recent_history", "") or "无",
        )
        response = self.router_model.invoke([HumanMessage(content=prompt)])
        label = str(getattr(response, "content", "") or "").strip().lower()
        if label in {"smalltalk", "capability_query", "non_domain", "product_qa", "policy_qa", "usage_qa", "fault_qa"}:
            return label
        return "non_domain"

    def _heuristic_intent(self, question: str) -> str | None:
        normalized = (question or "").strip().lower()
        if not normalized:
            return "smalltalk"
        policy_keywords = {"保修", "退货", "退款", "发票", "配送", "运费", "售后", "质保", "换货", "七天无理由"}
        fault_keywords = {"故障", "异响", "白烟", "不工作", "推不进去", "e1", "e2", "坏了", "报警", "风扇不转"}
        usage_keywords = {"怎么用", "怎么清洗", "清洁", "首次使用", "第一次用", "步骤", "预热", "保养"}
        capability_keywords = {
            "你可以帮我做什么",
            "你能帮我做什么",
            "你支持什么",
            "你支持查什么",
            "你能查什么",
            "你会什么",
            "你可以做什么",
            "怎么使用你",
            "你能做什么",
        }
        smalltalk_keywords = {
            "你好",
            "您好",
            "hi",
            "hello",
            "在吗",
            "我是谁",
            "你认识我吗",
            "你知道我是谁吗",
            "你记得我吗",
            "你是谁",
            "谢谢",
            "多谢",
            "感谢",
            "再见",
            "拜拜",
            "辛苦了",
            "早上好",
            "晚上好",
        }
        product_keywords = {
            "空气炸锅",
            "容量",
            "功率",
            "型号",
            "可视窗",
            "双热源",
            "旋钮",
            "触控",
            "几个人",
            "参数",
            "规格",
            "midea",
            "mf-",
        }
        if any(keyword in normalized for keyword in policy_keywords):
            return "policy_qa"
        if any(keyword in normalized for keyword in fault_keywords):
            return "fault_qa"
        if any(keyword in normalized for keyword in usage_keywords):
            return "usage_qa"
        if any(keyword in normalized for keyword in product_keywords):
            return "product_qa"
        if any(keyword in normalized for keyword in capability_keywords):
            return "capability_query"
        if any(keyword in normalized for keyword in smalltalk_keywords):
            return "smalltalk"
        return None

    def _has_policy_evidence(self, documents: list[Document]) -> bool:
        policy_doc_types = {"service", "returns", "shipping", "invoice", "warranty"}
        for doc in documents:
            model = str(doc.metadata.get("model", "")).strip().lower()
            doc_type = str(doc.metadata.get("doc_type", "")).strip().lower()
            if model == "shared" or doc_type in policy_doc_types:
                return True
        return False

    def _generate_smalltalk_answer(self, state: AgenticState) -> str:
        self._append_trace_call(state, "smalltalk_answer")
        prompt = self.smalltalk_answer_prompt.format(
            question=(state.get("question") or "").strip() or "你好",
            session_summary=state.get("session_summary", "") or "无",
            recent_history=state.get("recent_history", "") or "无",
        )
        try:
            response = self.smalltalk_model.invoke([HumanMessage(content=prompt)])
        except Exception:
            return self._build_smalltalk_fallback_answer(state)

        answer = str(getattr(response, "content", "") or "").strip()
        return answer or self._build_smalltalk_fallback_answer(state)

    def _build_smalltalk_fallback_answer(self, state: AgenticState) -> str:
        question = (state.get("question") or "").strip()
        lowered = question.lower()
        if any(keyword in lowered for keyword in {"谢谢", "多谢", "感谢"}):
            return "不客气。如果你想继续了解空气炸锅的参数、使用方法、清洁保养、售后规则或型号信息，可以直接告诉我。"
        if any(keyword in lowered for keyword in {"我是谁", "你认识我吗", "你知道我是谁吗"}):
            return "我没法知道你的真实身份，不过你可以直接告诉我想查哪款空气炸锅或遇到了什么问题。"
        if any(keyword in lowered for keyword in {"再见", "拜拜"}):
            return "好的，有需要随时来问我。空气炸锅的参数、使用、清洁、售后和型号问题我都可以继续帮你看。"
        return "你好，我可以帮你查询空气炸锅的参数、使用方法、清洁保养、售后规则和型号信息。你可以直接告诉我你的问题。"

    def _build_capability_answer(self) -> str:
        return (
            "我目前主要可以帮助你查询空气炸锅相关的商品参数、功能差异、使用说明、清洁保养、常见故障、"
            "售后规则和型号确认信息。比如你可以问我容量、功率、首次使用前要做什么、能不能退换货，"
            "或者某个型号之间有什么区别。"
        )

    def _build_non_domain_answer(self) -> str:
        return (
            "我目前主要负责空气炸锅相关的商品、使用和售后问题，"
            "暂时不能准确回答这个问题。"
            "如果你想咨询空气炸锅的参数、使用、清洁、售后或型号确认，我可以继续帮你。"
        )

    def _build_fallback_answer(self, state: AgenticState) -> str:
        if self._needs_model_clarification(state):
            return (
                "根据当前知识库资料，我暂时还不能确认你提到的是哪一个具体型号。"
                "为避免误导，建议先查看机身铭牌、说明书封面或订单页中的型号信息，"
                "也可以补充颜色、旋钮/面板样式、是否带可视窗等特征后再继续确认。"
            )
        return (
            "根据当前知识库资料，我暂时无法确认这个问题的准确答案。"
            "为避免误导，我不想直接给出不可靠结论。"
            "如果你愿意，可以补充更明确的型号、问题场景，或查看说明书与订单页后再继续确认。"
        )

    def _build_retrieval_summary(self, trace: dict[str, Any], docs: list[Document]) -> str:
        doc_types = "、".join(
            sorted(
                {
                    str(doc.metadata.get("doc_type", "")).strip()
                    for doc in docs
                    if str(doc.metadata.get("doc_type", "")).strip()
                }
            )
        ) or "无明确文档类型"
        return (
            f"命中文档数: {len(docs)}；"
            f"候选文档类型: {doc_types}；"
            f"检索模式: {trace.get('mode', '') or 'unknown'}；"
            f"重排后文档数: {trace.get('rerank_selected_count', len(docs))}。"
        )

    def _append_trace_call(self, state: AgenticState, call_name: str) -> None:
        trace_tool_calls = state.setdefault("trace_tool_calls", [])
        if isinstance(trace_tool_calls, list):
            trace_tool_calls.append(call_name)

    def _needs_model_clarification(self, state: AgenticState) -> bool:
        if not bool(state.get("should_reconfirm_model", False)):
            return False
        if state.get("intent") == "policy_qa":
            return False
        retrieved_models = [str(item).strip() for item in (state.get("retrieved_models") or []) if str(item).strip()]
        return bool(retrieved_models)

    def _generate_answer(
        self,
        *,
        query: str,
        rag_summary: str,
        docs: list[Document],
        runtime_context: AgenticState,
    ) -> str:
        system_prompt = load_system_prompts()
        context_prompt = build_runtime_context_prompt(runtime_context)
        evidence_prompt = self._build_evidence_prompt(docs)
        model_guard_prompt = self._build_model_confirmation_prompt(runtime_context)
        user_prompt = (
            f"用户问题:\n{query}\n\n"
            f"{context_prompt}\n\n"
            f"知识库摘要:\n{rag_summary or '未检索到可靠摘要。'}\n\n"
            f"检索证据:\n{evidence_prompt}\n\n"
            f"型号确认约束:\n{model_guard_prompt}\n\n"
            "请基于检索到的资料直接回答用户问题。"
            "回答应自然、简洁、准确；如果资料不足以支撑明确结论，请明确说明不知道或资料不足。"
            "不要编造商品信息、政策条款、价格、库存、物流时效或售后规则。"
        ).strip()
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return getattr(response, "content", "") or ""

    def _build_evidence_prompt(self, docs: list[Document]) -> str:
        if not docs:
            return "未检索到有效参考资料。"
        lines: list[str] = []
        for index, doc in enumerate(docs[:4], start=1):
            source = str(doc.metadata.get("source", "")).strip()
            content = " ".join((doc.page_content or "").split())
            if len(content) > 220:
                content = content[:220].rstrip() + "..."
            prefix = f"资料{index}"
            if source:
                prefix += f" | 来源: {source}"
            lines.append(f"{prefix}\n{content}")
        return "\n\n".join(lines)

    def _build_model_confirmation_prompt(self, runtime_context: dict[str, Any]) -> str:
        trace = self._get_latest_retrieval_trace(runtime_context)
        if not trace:
            return "当前没有额外的型号确认信息；如资料不足，请保持保守表达。"

        status = str(trace.get("model_confirmation_status", "")).strip() or "unconfirmed"
        source = str(trace.get("model_confirmation_source", "")).strip() or "retrieval_inferred"
        should_reconfirm = bool(trace.get("should_reconfirm_model", True))
        confirmed_model = str(trace.get("confirmed_model", "")).strip()
        query_models = trace.get("detected_query_models") or []
        retrieved_models = trace.get("retrieved_models") or []
        query_models_text = "、".join(str(item) for item in query_models if str(item).strip()) or "未在问题中识别到明确型号"
        retrieved_models_text = "、".join(str(item) for item in retrieved_models if str(item).strip()) or "当前证据未形成稳定型号集合"

        if not should_reconfirm:
            confirmed_model_text = confirmed_model or query_models_text
            return (
                f"型号确认状态: {status}。确认来源: {source}。"
                f"已确认型号: {confirmed_model_text}。"
                f"当前证据涉及的型号: {retrieved_models_text}。"
                "用户已经明确提供型号，且当前证据与之不冲突。"
                "请直接基于该型号回答参数、容量、适用人数或功能差异。"
                "不要再次建议用户查看铭牌、说明书封面、订单页或再次确认型号。"
                "不要使用“建议再确认型号”“您可以再核对型号”这类重复核验话术。"
                "只有在当前回答确实涉及证据冲突时，才允许补一句“如页面信息有更新，以最新资料为准”。"
            )

        return (
            f"型号确认状态: {status}。确认来源: {source}。"
            f"问题中明确出现的型号: {query_models_text}。"
            f"当前证据涉及的候选型号: {retrieved_models_text}。"
            "当前不能把任何检索到的型号直接当作用户真实购买的型号。"
            "禁止使用“您这款就是MF-XXX”或“您买的就是MF-XXX”这类确定表述。"
            "如需引用候选型号，只能用“候选型号可能包括……”或“如果您说的是MF-XXX”这类表达。"
            "优先提示用户查看机身铭牌、说明书封面、订单页，或提供颜色、旋钮/面板样式、是否带可视窗等外观特征来辅助确认。"
        )

    def _get_latest_retrieval_trace(self, runtime_context: dict[str, Any]) -> dict[str, Any] | None:
        traces = runtime_context.get("retrieval_trace") or []
        if not isinstance(traces, list) or not traces:
            return None
        latest = traces[-1]
        return latest if isinstance(latest, dict) else None

    def _infer_rag_service_from_tools(self) -> RagSummarizeService | None:
        rag_tool = self.tool_map.get("rag_summarize")
        rag_service = getattr(rag_tool, "_rag_service", None)
        return rag_service if isinstance(rag_service, RagSummarizeService) or rag_service is not None else None

    def _serialize_doc(self, doc: Document) -> dict[str, Any]:
        content = " ".join((doc.page_content or "").split())
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        return {
            "content": content,
            "metadata": {str(key): str(value) for key, value in (doc.metadata or {}).items()},
        }
